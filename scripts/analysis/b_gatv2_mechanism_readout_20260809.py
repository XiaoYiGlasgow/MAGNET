#!/usr/bin/env python3
"""B-M 机制层判据判读（预注册 `旋钮B_gatv2算子对照_预注册判据_20260808.md` §3 B-M）。

判据（冻结）：seed42 final_model，L1(新档) val 前 64 样本一个 batch，
conv1 输出的**批内节点间 spread** = max|out1_i − out1_0| > **1e-4**
（单测 [C] 同协议的真 ckpt 版；随机初始化 GATv2 实测 1.6e-01，GCN 4.5e-08）。

描述性读数（只报不判，20260808 判据文首降级决策）：完全图上 ego（节点 0，
`dataset.py:590` ego 恒为首节点）入边 attention 权重的变异系数，per-sample 分布；
alpha 含 GATv2 默认自环（如实注明）。

骨架复用 `k0_mechanism_readout_20260807.py`（load_model_for_inference 按 sidecar 建类 +
p0 的 load_tokens）；数据换**新档** val，模型类断言 MagnetGNNOperator。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN), str(_GNN.parent), str(_GNN.parent / "models"),
           str(_GNN / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.p0_branch_zero_ablation import load_tokens  # noqa: E402
from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402
from src.shared.utils.dataset import get_adjacency_config  # noqa: E402

VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
CKPT = _GNN / "experiments/knobs/b_gatv2_L1/seed42/models/final_model_default.pth"
OUT = _GNN / "scratchpad/b_gatv2_mechanism_readout_20260809.json"


def _load_model() -> torch.nn.Module:
    model = load_model_for_inference(CKPT, gnn_pooling_fix=True, device=torch.device("cpu"))
    model.eval()
    cls = type(model).__name__
    if cls != "MagnetGNNOperator":
        raise RuntimeError(f"模型类应为 MagnetGNNOperator，实际 {cls} —— sidecar 或建类路径不对")
    return model


@torch.no_grad()
def main() -> int:
    get_adjacency_config("default")
    _, data64 = load_tokens(VAL, "default", limit=64)
    if len(data64) < 64:
        raise RuntimeError(f"val 只取到 {len(data64)} 个样本，判据要求前 64")
    model = _load_model()
    conv1 = model.gnn_encoder.conv1

    # ---- B-M-1：前 64 样本一个 batch，conv1 输出批内节点间 spread ----
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(list(data64), batch_size=64, shuffle=False,
                                 follow_batch=["gnn_x"])))
    core = model._extract_core_features(batch.gnn_x)
    ei = batch.gnn_edge_index
    if ei.numel() and int(ei.max()) >= core.shape[0]:
        raise RuntimeError("gnn_edge_index 有越界边 —— 不经 encoder 过滤直接调 conv1 的前提不成立")
    out1 = conv1(core, ei)
    spread = float((out1 - out1[0]).abs().max())
    pass1 = spread > 1e-4

    # ---- 描述性（只报不判）：per-sample ego 入边 attention CV（heads=1，含自环）----
    cvs = []
    for d in data64:
        c = model._extract_core_features(d.gnn_x)
        e = d.gnn_edge_index
        if e.numel() == 0 or c.shape[0] < 2:
            continue
        _, (ei_a, alpha) = conv1(c, e, return_attention_weights=True)
        a_ego = alpha[ei_a[1] == 0].flatten()          # 目标节点 0 = ego（dataset.py:590）
        if a_ego.numel() < 2 or float(a_ego.mean()) == 0.0:
            continue
        cvs.append(float(a_ego.std(unbiased=True) / a_ego.mean()))
    cvs_np = np.array(cvs)

    report = {
        "ckpt": str(CKPT),
        "model_class": type(model).__name__,
        "n_samples": len(data64),
        "B-M-1_conv1_批内节点间spread": {
            "值": spread, "阈值": "> 1e-4",
            "锚": {"随机GATv2": 1.62e-01, "GCN对照": 4.47e-08},
            "PASS": pass1,
        },
        "描述性_ego入边attention_CV(只报不判,含自环)": {
            "n": int(cvs_np.size),
            "median": float(np.median(cvs_np)) if cvs_np.size else None,
            "min": float(cvs_np.min()) if cvs_np.size else None,
            "max": float(cvs_np.max()) if cvs_np.size else None,
        },
        "口径": "新档（batch=8+fix=True+高agent密度档）；L1 val 前 64；seed42 final_model",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"B-M-1  conv1 spread = {spread:.3e}  (>1e-4)  {'✅ PASS' if pass1 else '🔴 FAIL'}")
    if cvs_np.size:
        print(f"描述性 ego 入边 attention CV：median {np.median(cvs_np):.4f}"
              f"（min {cvs_np.min():.4f} / max {cvs_np.max():.4f}，n={cvs_np.size}，只报不判）")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
