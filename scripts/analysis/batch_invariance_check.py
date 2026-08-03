"""S4-b 第三步：batch-invariance 验收（阶段⑳ §2.3）。

同一 calibrated checkpoint、同一批样本、CPU eval，比较 `batch=1/8/32`：

| # | 判据 | 本脚本字段 |
|---|---|---|
| 1a | 增量空间 `max_abs_increment_diff ≤ 1e-5`（绝对） | `max_abs_increment_diff_vs_b1` |
| 1b | 位置空间 `max_abs_traj_diff ≤ 1e-5 × max|traj|`（相对） | `max_abs_traj_diff_vs_b1` / `traj_scale` |
| 2 | ADE/FDE 聚合差异只允许数值舍入级 | `ade_diff_vs_b1` / `fde_diff_vs_b1` |
| 3 | D0#7 per-graph pooling 同时启用 | `gnn_pooling_fix`（Q13=(ii)：正式验收必须 True） |
| 4 | 任一项失败 ⇒ S4-b 不得进入 G3 | `verdict` |

⚠️ `batch=1` 是 planner 的真实部署口径，故一律以 batch=1 为基准比较。
⚠️ `gnn_pooling_fix=False` 的权重在 batch>1 时 GNN 分支整 batch 共享一个向量（D0#7），
   即使 BN 已 calibrated 也**必然**不满足判据 1 —— 这是预期结果，不是脚本 bug。

**Q15（用户 2026-07-29 22:5x 裁定，选项 A）：判据 1 的 1e-5 锚在增量空间。**
理由：位置空间是增量 cumsum 60 步的结果，量级 ~100 m，而 float32 的相对分辨率 1.2e-7
⇒ 位置空间的数值噪声下界 ≈ 100 × 1.2e-7 ≈ 1.2e-05 **本身就超过 1e-5**。把绝对阈 1e-5
压在位置空间上，判的是 GEMM 归约顺序，不是语义 batch 依赖（实测 calibrated+poolON：
增量 7.15e-07 / 位置 2.29e-05 / ΔADE 2.4e-07 —— 三个数指向同一结论却被旧判据判 FAIL）。
故位置空间改用**相对界** `1e-5 × max|traj|`，绝对阈只锚在模型原生输出（增量）上。

用法::

    python3 scripts/analysis/batch_invariance_check.py \
        --ckpt experiments/calibration/calibrated/B_seed42/B_seed42_calibrated.pth \
        --eval_root data/processed_features_magnet_e1_none_v3_split3/val --n 384 \
        --gnn_pooling_fix \
        --out experiments/calibration/calibrated/B_seed42/B_seed42_batch_invariance.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT),
           str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.bn_calibration import (  # noqa: E402
    BN_MODULE_PATH, file_sha256, load_model_for_inference,
)
from src.shared.utils.dataset import build_graph_data_from_feature_df, get_adjacency_config  # noqa: E402
from train_magnet_imitation import NUM_FUTURE_POSES, _ade_fde, move_batch_to_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FILE_PREFIX = "features_"
#: 判据 1a 的阈值：模型原生输出（增量）空间的绝对界（阶段⑳ §2.3 + Q15=(A) 冻结）
MAX_ABS_INCREMENT_DIFF_TOL = 1e-5
#: 判据 1b 的系数：位置空间用相对界 `TOL × max|traj|`，避开 float32 在 ~100 m 量级的分辨率下界
MAX_ABS_TRAJ_DIFF_REL_TOL = 1e-5
#: 判据 2 的阈值：ADE/FDE 聚合差异只允许数值舍入级
AGG_DIFF_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S4-b batch-invariance 验收")
    p.add_argument("--ckpt", required=True, type=Path, help="calibrated checkpoint（也接受原始 ckpt 作对照）")
    p.add_argument("--eval_root", required=True, type=Path, help="评测样本目录（不参与 calibration）")
    p.add_argument("--n", type=int, default=384, help="样本数；必须能被最大 batch size 整除")
    p.add_argument("--batch_sizes", default="1,8,32")
    p.add_argument("--gnn_pooling_fix", action="store_true")
    p.add_argument("--dataset_config", default="default")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def load_samples(root: Path, n: int, dataset_config: str) -> List[Any]:
    """按文件名排序取前 n 个样本（与 `GraphDataset.process()` 的顺序一致）。"""
    get_adjacency_config(dataset_config)
    files = sorted(root.glob(f"{FILE_PREFIX}*.pkl"))[:n]
    if len(files) < n:
        raise RuntimeError(f"{root} 只有 {len(files)} 个样本，不足 {n}")
    out: List[Any] = []
    for f in files:
        graph, skip = build_graph_data_from_feature_df(pd.read_pickle(f), dataset_config)
        if skip is not None or graph is None:
            raise RuntimeError(f"{f.name} 被跳过（{skip}）")
        out.append(graph)
    return out


@torch.no_grad()
def forward_at_batch(model, data_list: List[Any], batch_size: int, device: torch.device,
                     follow_batch: List[str] | None) -> Tuple[torch.Tensor, torch.Tensor]:
    """在指定 batch size 下前向，返回 (预测增量 [N,120], 真值增量 [N,120])。"""
    from torch_geometric.loader import DataLoader  # noqa: WPS433

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, follow_batch=follow_batch)
    preds, ys = [], []
    for data in loader:
        data = move_batch_to_device(data, device)
        bs = int(data.num_graphs)
        preds.append(model(data)[:bs].cpu())
        ys.append(data.y.reshape(bs, -1).cpu())
    return torch.cat(preds), torch.cat(ys)


def to_positions(increments: torch.Tensor) -> torch.Tensor:
    """增量 [N,120] -> 轨迹位置 [N,60,2]（cumsum，与 `_ade_fde` 同一约定）。"""
    return torch.cumsum(increments.view(-1, NUM_FUTURE_POSES, 2), dim=1)


def main() -> None:
    args = parse_args()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    if 1 not in batch_sizes:
        raise ValueError("batch_sizes 必须包含 1（planner 部署口径 = 比较基准）")
    if any(args.n % b for b in batch_sizes):
        raise ValueError(f"n={args.n} 必须能被所有 batch size {batch_sizes} 整除，否则尾批大小不同")

    device = torch.device("cpu")
    data_list = load_samples(args.eval_root, args.n, args.dataset_config)
    model = load_model_for_inference(args.ckpt, args.gnn_pooling_fix, device)
    src_state = torch.load(args.ckpt, map_location="cpu")
    is_calibrated = f"{BN_MODULE_PATH}.running_mean" in (
        src_state["model_state_dict"] if isinstance(src_state, dict) and "model_state_dict" in src_state
        else src_state)
    follow = ["gnn_x"] if args.gnn_pooling_fix else None
    logger.info("ckpt=%s  calibrated=%s  gnn_pooling_fix=%s  n=%d",
                args.ckpt.name, is_calibrated, args.gnn_pooling_fix, args.n)

    results: Dict[int, Dict[str, float]] = {}
    ref_pred = ref_pos = None
    y_ref = None
    traj_scale = 0.0        # max|traj| on the batch=1 reference; 判据 1b 相对界的分母
    for bs in batch_sizes:
        pred, y = forward_at_batch(model, data_list, bs, device, follow)
        pos = to_positions(pred)
        ade, fde = _ade_fde(pred, y)
        if bs == 1:
            ref_pred, ref_pos, y_ref = pred, pos, y
            traj_scale = float(pos.abs().max())
        elif not torch.equal(y, y_ref):       # 防呆：样本顺序被 batch size 改了就没法比
            raise RuntimeError(f"batch={bs} 的真值与 batch=1 不一致，样本顺序被破坏")
        results[bs] = {
            "ade": float(ade.mean()), "fde": float(fde.mean()),
            "max_abs_increment_diff_vs_b1": float((pred - ref_pred).abs().max()),
            "max_abs_traj_diff_vs_b1": float((pos - ref_pos).abs().max()),
        }
        logger.info("batch=%-3d ADE=%.4f FDE=%.4f  max|Δ增量 vs b1|=%.3e  max|Δ位置 vs b1|=%.3e",
                    bs, results[bs]["ade"], results[bs]["fde"],
                    results[bs]["max_abs_increment_diff_vs_b1"],
                    results[bs]["max_abs_traj_diff_vs_b1"])

    ade_ref = results[1]["ade"]
    fde_ref = results[1]["fde"]
    for bs in batch_sizes:
        results[bs]["ade_diff_vs_b1"] = results[bs]["ade"] - ade_ref
        results[bs]["fde_diff_vs_b1"] = results[bs]["fde"] - fde_ref

    # 判据 1a：绝对界锚在模型原生输出（增量）空间
    crit1a = all(results[b]["max_abs_increment_diff_vs_b1"] <= MAX_ABS_INCREMENT_DIFF_TOL
                 for b in batch_sizes)
    # 判据 1b：位置空间用相对界；traj_scale 取 batch=1 参考轨迹的 max|位置|
    traj_diff_tol = MAX_ABS_TRAJ_DIFF_REL_TOL * traj_scale
    crit1b = all(results[b]["max_abs_traj_diff_vs_b1"] <= traj_diff_tol for b in batch_sizes)
    crit2 = all(abs(results[b]["ade_diff_vs_b1"]) <= AGG_DIFF_TOL
                and abs(results[b]["fde_diff_vs_b1"]) <= AGG_DIFF_TOL for b in batch_sizes)
    crit3 = bool(args.gnn_pooling_fix)
    verdict = "PASS" if (crit1a and crit1b and crit2 and crit3) else "FAIL"

    doc = {
        "contract": "S4-b batch-invariance 验收（阶段⑳ §2.3）",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256_file": file_sha256(args.ckpt),
        "is_calibrated": is_calibrated,
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),
        "eval_root": str(args.eval_root.resolve()),
        "n_samples": args.n,
        "device": "cpu",
        "batch_sizes": batch_sizes,
        "threshold_anchor": "Q15=(A)：绝对界锚在增量空间；位置空间用相对界 rel_tol×max|traj|",
        "tolerances": {
            "max_abs_increment_diff": MAX_ABS_INCREMENT_DIFF_TOL,
            "max_abs_traj_diff_rel": MAX_ABS_TRAJ_DIFF_REL_TOL,
            "max_abs_traj_diff_abs_derived": traj_diff_tol,
            "traj_scale_max_abs_position_m": traj_scale,
            "agg_diff": AGG_DIFF_TOL,
        },
        "per_batch": {str(k): v for k, v in results.items()},
        "criteria": {
            "1a_max_abs_increment_diff": crit1a,
            "1b_max_abs_trajectory_diff_relative": crit1b,
            "2_agg_ade_fde_rounding_level": crit2,
            "3_gnn_pooling_fix_enabled": crit3,
        },
        "verdict": verdict,
        "note": ("gnn_pooling_fix=False 时判据 1a/1b 必然失败：GNNEncoder 在 batch is None 分支走 "
                 "x.mean(dim=0)，整 batch 共享一个向量（D0#7），与 BN 无关。"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    logger.info("判据 1a=%s（增量 ≤%.1e） 1b=%s（位置 ≤%.3e = %.1e×%.2f m） 2=%s 3=%s -> %s",
                crit1a, MAX_ABS_INCREMENT_DIFF_TOL, crit1b, traj_diff_tol,
                MAX_ABS_TRAJ_DIFF_REL_TOL, traj_scale, crit2, crit3, verdict)
    logger.info("写出 %s", args.out)


if __name__ == "__main__":
    main()
