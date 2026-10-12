#!/usr/bin/env python3
"""K0-M 机制层判据判读（预注册 `K0_poolfix纯flag对照_预注册判据_20260807.md` §3）。

两条判据（seed42 final_model，L1 val，batch=32 口径，阶段123 同协议）：

1. 前 64 样本按 batch=32 前向，batch 内 ``gnn_feat`` 的 max|x_i − x_0| > 1e-6
   （s4b fix=False 实测 0.000e+00 —— 整批平均使 batch 内逐位相同）；
2. gnn 分支场景间总方差的 batch=32 / batch=1 之比 ∈ [0.5, 2]
   （s4b fix=False 实测 27× 断层：0.0204@b32 vs 0.5590@b1；协议同
   `recon_gnn_variance_20260807.py`，300 token）。

🔴 与 recon 脚本的关键差异：K0 模型是 ``MagnetImitationPooled``，DataLoader 必须
``follow_batch=["gnn_x"]`` 产出 ``gnn_x_batch``，否则父类**静默回退** mean(0).repeat
（即被修的 bug 路径），测出来的还是旧行为。脚本对模型类名与 batch 属性均做断言。
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

from scripts.analysis.p0_branch_zero_ablation import SLICES, BranchTap, load_tokens  # noqa: E402
from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402
from src.shared.utils.dataset import get_adjacency_config  # noqa: E402
from train_magnet_imitation import move_batch_to_device  # noqa: E402

VAL = _GNN / "data/processed_features_scaling_L1_logsplit/val"
CKPT = _GNN / "experiments/knobs/k0_poolfix_L1/seed42/models/final_model_default.pth"
OUT = _GNN / "scratchpad/k0_mechanism_readout_20260807.json"


def _load_model() -> torch.nn.Module:
    model = load_model_for_inference(CKPT, gnn_pooling_fix=True, device=torch.device("cpu"))
    model.eval()
    cls = type(model).__name__
    if cls != "MagnetImitationPooled":
        raise RuntimeError(f"模型类应为 MagnetImitationPooled，实际 {cls} —— sidecar 或建类路径不对")
    return model


@torch.no_grad()
def _forward_capture(model, data_list, batch_size: int) -> np.ndarray:
    """带 follow_batch 前向，返回 fusion 入口特征 [N, 384]。"""
    from torch_geometric.loader import DataLoader

    tap = BranchTap(model)
    feats = []
    try:
        for batch in DataLoader(list(data_list), batch_size=batch_size, shuffle=False,
                                follow_batch=["gnn_x"]):
            if not hasattr(batch, "gnn_x_batch"):
                raise RuntimeError("batch 无 gnn_x_batch —— follow_batch 未生效，父类会静默回退整批平均")
            batch = move_batch_to_device(batch, torch.device("cpu"))
            bs = int(batch.num_graphs)
            model(batch)
            feats.append(tap.captured[:bs].cpu().numpy())
    finally:
        tap.close()
    return np.concatenate(feats)


def main() -> int:
    get_adjacency_config("default")
    _, data_300 = load_tokens(VAL, "default", limit=300)
    model = _load_model()
    lo, hi = SLICES["gnn"]

    # ---- 判据 1：前 64 样本、batch=32，batch 内 gnn_feat max|x_i − x_0| ----
    fusion64 = _forward_capture(model, data_300[:64], 32)
    per_batch_spread = []
    for b0 in range(0, 64, 32):
        blk = fusion64[b0:b0 + 32, lo:hi]
        per_batch_spread.append(float(np.abs(blk - blk[0:1]).max()))
    m1 = max(per_batch_spread)
    pass1 = m1 > 1e-6

    # ---- 判据 2：300 token，gnn 场景间总方差 b32 / b1 之比 ----
    var = {}
    for bs in (32, 1):
        fusion = _forward_capture(model, data_300, bs)
        var[bs] = {name: float(fusion[:, a:b].var(axis=0, ddof=1).sum())
                   for name, (a, b) in SLICES.items()}
    ratio = var[32]["gnn"] / var[1]["gnn"]
    pass2 = 0.5 <= ratio <= 2.0

    report = {
        "ckpt": str(CKPT),
        "model_class": type(model).__name__,
        "判据1_batch内gnn_max_abs_dev": {"值": m1, "逐batch": per_batch_spread,
                                          "阈值": "> 1e-6", "s4b对照": 0.0, "PASS": pass1},
        "判据2_gnn方差_b32_b1": {"b32": var[32]["gnn"], "b1": var[1]["gnn"], "之比": ratio,
                                  "区间": [0.5, 2.0], "s4b对照之比": 0.0204 / 0.5590, "PASS": pass2},
        "旁证_ctx_ego方差_b32": {k: var[32][k] for k in ("ctx", "ego")},
        "旁证_ctx_ego方差_b1": {k: var[1][k] for k in ("ctx", "ego")},
        "K0-M": bool(pass1 and pass2),
        "口径": "低 agent 密度档（20 槽+建图前删静止）；L1 val；seed42 final_model；fix=True",
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"判据1  max|Δ|@batch32 = {m1:.3e}  (>1e-6)  {'✅' if pass1 else '🔴'}")
    print(f"判据2  gnn var b32={var[32]['gnn']:.4f}  b1={var[1]['gnn']:.4f}  "
          f"比={ratio:.3f} ∈[0.5,2]  {'✅' if pass2 else '🔴'}")
    print(f"K0-M = {'PASS ✅' if report['K0-M'] else 'FAIL 🔴'}")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
