"""独立 test set 评测器（VT-2026-04191 阶段⑬，回应 R4-7）。

R4-7 原文批评：loss 权重按 val 选、early stopping 按 val ADE、拓扑按 val ADE 选，
最终 Table II 又在**同一个 val** 上报结果。本脚本把 headline 报数与模型选择彻底隔离。

**使用纪律（写死在流程里，不靠自觉）**：
  - 输入必须是**已冻结**的 checkpoint；本脚本不训练、不做 early stopping、不选超参、不选拓扑；
  - test 只在方法冻结后正式运行一次；
  - 报数一律用 **saved-checkpoint** 口径，**不得**使用训练曲线的独立 minima
    （阶段9 §4.3 已订正：曲线 minima 不保证来自保存的 checkpoint，仅作训练诊断）。

产出：
  test_metrics.json  汇总 ADE/FDE + 完整复现元数据（ckpt/seed/route_mode/perturbation/split/代码版本）
  test_per_scene.csv  逐样本 ADE/FDE

运行：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/eval_test_set.py \
      --checkpoint /abs/.../models/best_model_default.pth \
      --test_root  /abs/.../<data>_split3/test \
      --output_dir /abs/.../experiments/<run>/test_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
_MODELS = _DEVKIT / "models"
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_MODELS), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from transformer_predictor import TransformerPredictor  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_test_set")


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_GNN_ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    """逐样本 ADE/FDE。模型输出为 60×2 局部逐帧增量，需 cumsum 成位置后与 GT 位置比。"""
    model.eval()
    per_scene = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).reshape(-1, config.NUM_FUTURE_POSES, 2)      # [B,60,2] 增量
        gt = batch.y.reshape(-1, config.NUM_FUTURE_POSES, 2)             # [B,60,2] 增量
        pred_pos = torch.cumsum(pred, dim=1)
        gt_pos = torch.cumsum(gt, dim=1)
        d = torch.norm(pred_pos - gt_pos, dim=-1)                        # [B,60]
        ade = d.mean(dim=1).cpu().numpy()
        fde = d[:, -1].cpu().numpy()
        for a, f in zip(ade, fde):
            per_scene.append((float(a), float(f)))
    arr = np.asarray(per_scene, dtype=np.float64)
    return {
        "n_samples": int(arr.shape[0]),
        "ade_mean": float(arr[:, 0].mean()), "ade_std": float(arr[:, 0].std(ddof=1)),
        "fde_mean": float(arr[:, 1].mean()), "fde_std": float(arr[:, 1].std(ddof=1)),
        "ade_median": float(np.median(arr[:, 0])), "fde_median": float(np.median(arr[:, 1])),
        "_per_scene": per_scene,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立 test set 评测（R4-7）；只读 checkpoint，不训练")
    p.add_argument("--checkpoint", required=True, type=str, help="冻结的 bare state_dict（绝对路径）")
    p.add_argument("--test_root", required=True, type=str, help="split3 的 test leaf root（绝对路径）")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--dataset_config", default="default")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split_manifest", default=None, help="split3_manifest.json；默认取 test_root 上级")
    p.add_argument("--tag", default="", help="实验标签，如 B/R/P/RP + seed")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ckpt = Path(a.checkpoint).resolve()
    test_root = Path(a.test_root).resolve()
    out_dir = Path(a.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        logger.error("checkpoint 不存在: %s", ckpt)
        sys.exit(1)

    # 训练侧元数据（route_mode / perturbation），供报表与一致性追溯
    meta_path = ckpt.parent / "train_meta.json"
    train_meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            train_meta = json.load(f)
    else:
        logger.warning("checkpoint 无 train_meta.json sidecar：%s（route_mode 按 'none' 记录）", meta_path)

    sm_path = Path(a.split_manifest) if a.split_manifest else test_root.parent / "split3_manifest.json"
    split_meta = {}
    if sm_path.exists():
        with open(sm_path, encoding="utf-8") as f:
            sm = json.load(f)
        split_meta = {k: v for k, v in sm.items() if k != "logs"}

    device = torch.device(a.device)
    get_adjacency_config(a.dataset_config)
    ds = GraphDataset(str(test_root), transform=None)   # ★ test 绝不增广
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
    logger.info("test 集: %d 样本 (%s)", len(ds), test_root)

    model = TransformerPredictor().to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    logger.info("已加载冻结 checkpoint: %s", ckpt)

    t0 = time.time()
    res = evaluate(model, loader, device)
    per_scene = res.pop("_per_scene")

    summary = {
        "tag": a.tag,
        "protocol": "saved-checkpoint 口径；test 仅用于最终 headline，不参与任何选择（R4-7）",
        **res,
        "checkpoint": str(ckpt),
        "test_root": str(test_root),
        "dataset_config": a.dataset_config,
        "route_mode": train_meta.get("route_mode", "none"),
        "perturbation": train_meta.get("perturbation"),
        "train_seed": train_meta.get("seed"),
        "split_manifest": split_meta,
        "git_rev": _git_rev(),
        "eval_seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "test_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ade_m", "fde_m"])
        for i, (ade, fde) in enumerate(per_scene):
            w.writerow([i, f"{ade:.6f}", f"{fde:.6f}"])

    print()
    print("=" * 64)
    print(f"独立 TEST 评测{(' [' + a.tag + ']') if a.tag else ''}")
    print("=" * 64)
    print(f"样本数    : {res['n_samples']}")
    print(f"ADE       : {res['ade_mean']:.3f} ± {res['ade_std']:.3f} m  (median {res['ade_median']:.3f})")
    print(f"FDE       : {res['fde_mean']:.3f} ± {res['fde_std']:.3f} m  (median {res['fde_median']:.3f})")
    print(f"route_mode: {summary['route_mode']}   perturbation: "
          f"{(summary['perturbation'] or {}).get('enabled')}")
    print(f"产物      : {out_dir}")


if __name__ == "__main__":
    main()
