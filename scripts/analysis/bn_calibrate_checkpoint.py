"""S4-b 第二步：用固定 train-only manifest 给一个 checkpoint 做 BN calibration，并固化 provenance。

产物（全部落 `experiments/`，禁止落 `scratchpad/`）::

    <out_dir>/<run>_calibrated.pth          # bare state_dict，多出 ego_history_encoder.1.running_*
    <out_dir>/<run>_calibration.json        # provenance：原 ckpt SHA256 + manifest hash + calibrated SHA256

阶段⑳ §2.2 要求记录的三个哈希、manifest、顺序/批量/dtype/代码版本全部写进 provenance JSON。

用法::

    python3 scripts/analysis/bn_calibrate_checkpoint.py \
        --ckpt experiments/ablation_2x2_f1/dev/B_seed42/models/best_model_default.pth \
        --manifest experiments/calibration/calib_manifest_f1_train_n1600.json \
        --out_dir experiments/calibration/calibrated/B_seed42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT),
           str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.bn_calibration import (  # noqa: E402
    ARCH_BUILD_KEYS, BN_MODULE_PATH, CalibrationError, build_model, calibrate_bn,
    collate_fixed_batches, read_architecture,
    file_sha256, state_dict_sha256, swap_bn_with_running_stats,
)
from src.shared.utils.dataset import build_graph_data_from_feature_df, get_adjacency_config  # noqa: E402
from train_magnet_imitation import move_batch_to_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FILE_PREFIX = "features_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S4-b：BN calibration + provenance 固化")
    p.add_argument("--ckpt", required=True, type=Path, help="原始 bare checkpoint（best_model_default.pth）")
    p.add_argument("--manifest", required=True, type=Path, help="train-only calibration manifest")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="Q13=(ii)：对 fix=True 训练的 checkpoint 必须加，与训练语义一致")
    p.add_argument("--dataset_config", default="default")
    p.add_argument("--device", default="cpu", help="CPU 即可；calibration 只前向 1600 个样本")
    return p.parse_args()


def _git_provenance(repo: Path) -> Dict[str, str]:
    def _run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "UNKNOWN"
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "dirty": "yes" if _run("git", "status", "--porcelain") else "no"}


def load_manifest_samples(manifest: Dict[str, Any], dataset_config: str) -> List[Any]:
    """按 manifest 顺序加载样本，走与训练 `GraphDataset.process()` 同一个 helper。

    :raises CalibrationError: 任一样本被 skip（会破坏等大小 batch 与 manifest 哈希的对应关系）。
    """
    train_root = Path(manifest["train_root"])
    get_adjacency_config(dataset_config)     # 设全局，决定 edge builder
    data_list: List[Any] = []
    for row in manifest["entries"]:
        token = row["token"]
        df = pd.read_pickle(train_root / f"{FILE_PREFIX}{token}.pkl")
        graph, skip = build_graph_data_from_feature_df(df, dataset_config)
        if skip is not None or graph is None:
            raise CalibrationError(f"token {token} 被跳过（{skip}）；manifest 必须逐条可用")
        data_list.append(graph)
    logger.info("加载 calibration 样本 %d 条（顺序 = manifest 顺序）", len(data_list))
    return data_list


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())

    # 重算 manifest 哈希，防止有人改了 entries 却没改 hash
    tokens = [str(r["token"]) for r in manifest["entries"]]
    recomputed = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
    if recomputed != manifest["manifest_hash_sha256"]:
        raise CalibrationError(f"manifest 哈希不符：文件写 {manifest['manifest_hash_sha256']}，"
                               f"实算 {recomputed}")
    if manifest["source_split"] != "train":
        raise CalibrationError(f"manifest 来源是 {manifest['source_split']}，S4-b 只允许 train split")

    device = torch.device(args.device)

    # 架构解析（C2 之后模型不止两种）：优先读源 ckpt 同目录 `train_meta.json` 的 architecture 块，
    # 没有就退回命令行 flag（老 checkpoint 的正常情况）。
    # 🔴 `ego_query_heads` 无法从权重形状反推 ⇒ 缺 sidecar 时不猜，让 strict load 抛错。
    arch = read_architecture(args.ckpt) or {}
    build_kwargs = {"gnn_pooling_fix": bool(
        arch.get("gnn_pooling_fix_effective", arch.get("gnn_pooling_fix", args.gnn_pooling_fix)))}
    for k in ARCH_BUILD_KEYS:
        if k in arch:
            build_kwargs[k] = arch[k]
    if arch:
        logger.info("架构取自源 ckpt sidecar: %s", build_kwargs)

    # 🔴 图拓扑一致性（2026-08-01 加）：`--dataset_config` 默认是 "default"，而它**静默决定 edge builder**。
    #   拿它给 `nearby_only` 训练的 checkpoint 做 calibration，等于在**完全图**上统计 BN、
    #   却把权重部署到近邻图上 —— 与 planner 侧 `_check_adjacency_consistency` 同一类失配，必须一起堵。
    #   manifest 本身只是 token 列表（config-agnostic），可跨配置复用，所以只需校验不需换 manifest。
    src_meta_path = args.ckpt.parent / "train_meta.json"
    if src_meta_path.exists():
        src_cfg = json.loads(src_meta_path.read_text(encoding="utf-8")).get("dataset_config")
        if src_cfg is not None and src_cfg != args.dataset_config:
            raise CalibrationError(
                f"图拓扑失配：源 checkpoint 训练用 dataset_config='{src_cfg}'，"
                f"本次 calibration 传的是 '{args.dataset_config}'。"
                f"改成 --dataset_config {src_cfg}")
        logger.info("图拓扑一致性校验通过: %s", args.dataset_config)

    data_list = load_manifest_samples(manifest, args.dataset_config)
    # follow_batch 必须跟**生效**的 pooling 语义走，否则 calibration 的前向与训练/部署不同路
    follow = ["gnn_x"] if build_kwargs["gnn_pooling_fix"] else None
    batches = collate_fixed_batches(data_list, manifest["batch_size"], follow_batch=follow)

    src_state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(src_state, dict) and "model_state_dict" in src_state:
        src_state = src_state["model_state_dict"]
    if f"{BN_MODULE_PATH}.running_mean" in src_state:
        raise CalibrationError(f"{args.ckpt} 已经是 calibrated checkpoint，不要重复 calibration")

    model = build_model(**build_kwargs).to(device)
    model.load_state_dict(src_state)   # strict：架构不符即抛错，不静默部分加载
    model.eval()
    logger.info("model=%s  build_kwargs=%s  device=%s",
                type(model).__name__, build_kwargs, device)

    swap_bn_with_running_stats(model)
    bn_stats = calibrate_bn(model, batches, device, move_batch_to_device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.ckpt.parent.parent.name          # .../<run>/models/best_model_default.pth
    out_ckpt = args.out_dir / f"{run_name}_calibrated.pth"
    calibrated_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(calibrated_state, out_ckpt)

    provenance = {
        "contract": "S4-b calibrated checkpoint（阶段⑳ §2.2）",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "run_name": run_name,
        "source_checkpoint": str(args.ckpt.resolve()),
        "source_checkpoint_sha256_file": file_sha256(args.ckpt),
        "source_checkpoint_sha256_state": state_dict_sha256(src_state),
        "calibration_manifest": str(args.manifest.resolve()),
        "calibration_manifest_hash_sha256": manifest["manifest_hash_sha256"],
        "calibration_n_samples": manifest["n_selected"],
        "calibration_source_split": manifest["source_split"],
        "calibrated_checkpoint": str(out_ckpt.resolve()),
        "calibrated_checkpoint_sha256_file": file_sha256(out_ckpt),
        "calibrated_checkpoint_sha256_state": state_dict_sha256(calibrated_state),
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),      # 命令行原始 flag
        # 🔴 实际用于建模的架构（C2 起命令行 flag 不足以描述模型）
        "architecture_effective": build_kwargs,
        "architecture_source": "source_ckpt_train_meta.json" if arch else "cli_flags",
        "dataset_config": args.dataset_config,
        "device": str(device),
        "dtype": "float32",
        "bn_calibration": bn_stats,
        "code_version": _git_provenance(_GNN_ROOT),
        "env": {"python": platform.python_version(), "torch": torch.__version__},
        "firewall_note": "locked test 未参与 calibration / checkpoint selection / 阈值设定",
    }
    out_json = args.out_dir / f"{run_name}_calibration.json"
    out_json.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    logger.info("calibrated ckpt -> %s", out_ckpt)
    logger.info("provenance     -> %s", out_json)
    logger.info("SHA256  src=%s  calibrated=%s",
                provenance["source_checkpoint_sha256_state"][:16],
                provenance["calibrated_checkpoint_sha256_state"][:16])


if __name__ == "__main__":
    main()
