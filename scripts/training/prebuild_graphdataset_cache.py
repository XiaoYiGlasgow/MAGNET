"""E1 · 预建 GraphDataset .pt cache（单进程），供多 seed 并行训练共享、避免并发 process race。

多个训练进程同时 GraphDataset(root) 会各自触发 process() 并写同一 processed/config_default/dataset_default.pt
→ race/损坏。本脚本先单进程把 train/val 两个 root 的 .pt 建好，训练进程再直接加载。

运行（conda nuplan env）：
  PYTHONPATH=GNN_Trans:nuplan-devkit /home/team/anaconda3/envs/nuplan/bin/python \
      GNN_Trans/scripts/training/prebuild_graphdataset_cache.py \
      --train_root /abs/.../train --val_root /abs/.../val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
GNN_TRANS_ROOT = _THIS.parents[2]
DEVKIT_ROOT = GNN_TRANS_ROOT.parent
for _p in (str(GNN_TRANS_ROOT), str(DEVKIT_ROOT / "models"), str(DEVKIT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="预建 train/val GraphDataset .pt cache")
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--val_root", required=True)
    ap.add_argument("--dataset_config", default="default")
    a = ap.parse_args()

    get_adjacency_config(a.dataset_config)
    tr = GraphDataset(a.train_root)
    print(f"[prebuild] train cache OK: {len(tr)} samples @ {a.train_root}")
    va = GraphDataset(a.val_root)
    print(f"[prebuild] val   cache OK: {len(va)} samples @ {a.val_root}")


if __name__ == "__main__":
    main()
