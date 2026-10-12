#!/usr/bin/env python3
"""新档（full_retention）训练显存探针（2026-08-07）。

背景：冒烟 v1(batch=32)/v2(batch=16) 均 OOM，崩点 = context_transformer 的
per-sample 注意力（`transformer_predictor.py:238`）。data.x 是 **point-level**
（每 polyline 点一行），注意力显存 ∝ N²，且 per-sample 循环把整个 batch 的
激活图同时攥着 ⇒ 总量 ∝ B·N²。旧档 N≈2400 × b32 ≈ 38–42 G（K0 实测吻合）；
新档 N 中位约 ×1.5、最坏 ~4.8k ⇒ b16 也爆。

本探针不再猜：取 train 集 data.x 点数最大的 B 个样本拼「最坏 batch」，
真实 forward+backward 一次，量 `torch.cuda.max_memory_allocated`，
从候选 batch 里选「最坏 batch 峰值 ≤ 40 GB（留 ~15% 余量）」的最大档。
顺带落盘 x 点数分布（median/p99/max），供排期与写作引用。
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

from src.shared.utils.bn_calibration import build_model  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402
from train_magnet_imitation import move_batch_to_device  # noqa: E402

TRAIN_ROOT = _GNN / "data/processed_features_fullret_L1_logsplit/train"
OUT = _GNN / "scratchpad/fr_mem_probe_20260807.json"
DEVICE = torch.device("cuda:0")
CAND_BS = (16, 12, 8, 6, 4)
BUDGET_GB = 40.0  # 47.5 总量留 ~15% 余量


def main() -> int:
    get_adjacency_config("default")
    ds = GraphDataset(str(TRAIN_ROOT), transform=None)
    n_x = np.array([int(d.x.size(0)) for d in ds])
    dist = {"n": int(n_x.size), "median": float(np.median(n_x)),
            "p99": float(np.percentile(n_x, 99)), "max": int(n_x.max())}
    print(f"data.x 点数分布：median {dist['median']:.0f} / p99 {dist['p99']:.0f} / max {dist['max']}")

    worst_idx = np.argsort(-n_x)
    model = build_model(gnn_pooling_fix=True).to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = torch.nn.MSELoss()

    from torch_geometric.loader import DataLoader

    rows = []
    chosen = None
    for bs in CAND_BS:
        batch = next(iter(DataLoader([ds[int(i)] for i in worst_idx[:bs]],
                                     batch_size=bs, follow_batch=["gnn_x"])))
        batch = move_batch_to_device(batch, DEVICE)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        try:
            opt.zero_grad()
            out = model(batch)
            loss = crit(out, torch.zeros_like(out))
            loss.backward()
            opt.step()
            peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1024 ** 3
            ok = peak_gb <= BUDGET_GB
            rows.append({"batch": bs, "peak_gb": round(peak_gb, 2), "fits_budget": ok})
            print(f"batch={bs:<3} 最坏batch峰值 {peak_gb:.2f} GB  {'✅ ≤40' if ok else '🔴 >40'}")
            if ok and chosen is None:
                chosen = bs
                break  # 从大到小试，第一个过预算的就是答案
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "peak_gb": None, "fits_budget": False, "oom": True})
            print(f"batch={bs:<3} 🔴 OOM")
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps({"x_points_dist": dist, "budget_gb": BUDGET_GB,
                               "rows": rows, "chosen_batch": chosen}, indent=2, ensure_ascii=False))
    print(f"→ 选定 batch = {chosen}（最坏 batch 口径，预算 {BUDGET_GB} GB）")
    print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
