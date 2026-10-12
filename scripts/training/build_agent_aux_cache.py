"""闸⑥ —— 离线构建 agent aux 监督缓存（与 `GraphDataset` 的 `.pt` 逐样本对齐）。

**为什么需要这一步**

`train_magnet_imitation.py` 消费的是 `GraphDataset` 的离线 `.pt`，里面**没有** `AGENT_FUTURES`
（那是原始 feature pkl 里的字段）。而 `attach_agent_aux()` 需要原始 `row` 才能算目标。
两条路：改 `dataset.py` 让 `.pt` 里多带三个字段（**受管风险 + 要重建全部 `.pt`**），
或离线算一份按 dataset 下标对齐的缓存（本脚本）。选后者，理由与阶段76 一致：
`dataset.py` 正被在跑的训练消费，且重建 `.pt` 比本缓存贵得多。

**对齐是怎么保证的（不是靠"应该一样"）**

`GraphDataset.process()` 的样本顺序 = `os.walk(root)` 收集全部 `*.pkl` → `sorted()` →
逐个 `build_graph_data_from_feature_df()`，**跳过** `empty_traj` / `no_polylines` / 抛异常的。
本脚本**逐字复刻**这套顺序与跳过逻辑，因此第 i 条缓存对应第 i 个 dataset 样本。

但"复刻"不等于"证明"。所以每条缓存还存两个指纹：`n_nodes` 与 `gnn_x_sum`，
消费侧（`AgentAuxGraphDataset.get()`）**逐样本断言**指纹一致 ——
错位会在第一个样本上当场炸，而不是静默产出错监督。

**失败率闸门**：位置 join 失败的 agent 节点不参与 aux loss。
若失败率超过 `--max_join_fail_rate`（默认 1%）直接**拒绝落盘** —— 大面积失败会让
aux 退化成"几乎没有监督"，训练结果落到判据"不升"档，再被读成
「瓶颈不在监督」这一**错误结论**（与 `magnet_agent_aux.py` 坑 1 同一个陷阱）。

🔴 **闸门只数 join 失败（`slot < 0`），不数「整槽被 mask 掉」**。
两者都表现为该 agent 全程不参与 loss，但成因完全不同：
前者是**接线错误**，后者是**已知数据性质**（`AGENT_FUTURES` 尾部零填充，实测 21% 的槽受影响、
最坏 98.3%）。08-04 首版把两者合并计数，闸门在完全正常的数据上以 3.25% 误触发 ——
**质量闸门本身也会有 bug，且它的 bug 长得就像"数据有问题"**。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/training/build_agent_aux_cache.py \
        --root /abs/.../processed_features_scaling_L1_logsplit/train \
        --adjacency_config default
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from magnet_agent_aux_wiring import build_agent_aux_fields  # noqa: E402
from src.shared.utils.dataset import (  # noqa: E402
    build_graph_data_from_feature_df,
    get_adjacency_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("build_agent_aux_cache")

FUTURE_FRAMES = 60

#: 缓存文件名 —— 与 `dataset_{cfg}.pt` 同目录、同 config 维度，避免跨 config 误用。
CACHE_NAME = "agent_aux_{cfg}.pt"


def cache_path(root: str, adjacency_config: str) -> Path:
    """缓存落盘位置：与 `dataset_{cfg}.pt` 同目录。"""
    return Path(root) / "processed" / f"config_{adjacency_config}" / CACHE_NAME.format(
        cfg=adjacency_config)


def _pkl_list(root: str) -> List[str]:
    """与 `GraphDataset.process()` 逐字相同的文件发现与排序。"""
    abs_root = os.path.abspath(root)
    out: List[str] = []
    for r, _dirs, files in os.walk(abs_root):
        for f in files:
            if f.endswith("pkl"):
                out.append(os.path.join(r, f))
    return sorted(out)


def build(root: str, adjacency_config: str, max_join_fail_rate: float) -> Dict:
    """构建缓存并返回统计。"""
    get_adjacency_config(adjacency_config)          # 设全局，与 trainer 侧一致
    paths = _pkl_list(root)
    logger.info("发现 %d 个 pkl（root=%s）", len(paths), root)

    entries: List[Optional[Dict]] = []
    stats = {"total_files": len(paths), "processed": 0, "skipped_empty_traj": 0,
             "skipped_no_polylines": 0, "skipped_errors": 0,
             "aux_none": 0, "aux_no_agent": 0,
             # 闸门口径：join 失败 = slot < 0（接线错误）
             "n_agent_nodes": 0, "n_join_failed_nodes": 0, "n_samples_with_join_fail": 0,
             # 仅供参考：整槽零填充（已知数据性质，**不进闸门**）
             "n_all_zero_target_nodes": 0, "n_masked_frames": 0, "n_total_frames": 0}
    t0 = time.time()

    for i, p in enumerate(paths):
        try:
            df = pd.read_pickle(p)
            graph_data, skip_reason = build_graph_data_from_feature_df(df, adjacency_config)
            if skip_reason == "empty_traj":
                stats["skipped_empty_traj"] += 1
                continue
            if skip_reason == "no_polylines":
                stats["skipped_no_polylines"] += 1
                continue
        except Exception as e:                      # noqa: BLE001 与 dataset.py 同款兜底
            stats["skipped_errors"] += 1
            logger.warning("跳过 %s -> %s", os.path.basename(p), e)
            continue

        stats["processed"] += 1
        n_nodes = int(graph_data.gnn_x.shape[0]) if getattr(graph_data, "gnn_x", None) is not None else 0
        fp = float(graph_data.gnn_x.sum()) if n_nodes else 0.0

        # strict=False：单样本 join 失败不炸整批，但**计数**并在末尾判闸门
        out = None
        try:
            out = build_agent_aux_fields(df.iloc[0], adjacency_config,
                                         strict=False, return_slots=True)
        except Exception as e:                      # noqa: BLE001
            logger.warning("aux 构建异常 %s -> %s", os.path.basename(p), e)

        if out is None:
            stats["aux_none"] += 1
            entries.append({"n_nodes": n_nodes, "gnn_x_sum": fp,
                            "is_agent": torch.zeros(n_nodes, dtype=torch.bool),
                            "tgt": torch.zeros(0, FUTURE_FRAMES, 2),
                            "mask": torch.zeros(0, FUTURE_FRAMES, dtype=torch.bool)})
        else:
            is_agent, tgt, mask, slots = out
            if tgt.shape[0] == 0:
                stats["aux_no_agent"] += 1
            else:
                stats["n_agent_nodes"] += int(tgt.shape[0])
                # 闸门口径：位置 join 失败（接线错误）
                n_fail = int((slots < 0).sum())
                stats["n_join_failed_nodes"] += n_fail
                if n_fail:
                    stats["n_samples_with_join_fail"] += 1
                # 参考口径：join 成功但整槽零填充（已知数据性质，不进闸门）
                dead_ok = int((~mask.any(dim=1) & torch.from_numpy(slots >= 0)).sum())
                stats["n_all_zero_target_nodes"] += dead_ok
                stats["n_masked_frames"] += int((~mask).sum())
                stats["n_total_frames"] += int(mask.numel())
            entries.append({"n_nodes": n_nodes, "gnn_x_sum": fp,
                            "is_agent": is_agent, "tgt": tgt, "mask": mask})

        if (i + 1) % 2000 == 0:
            logger.info("进度 %d/%d，用时 %.1f min", i + 1, len(paths), (time.time() - t0) / 60)

    n_nodes_tot = max(stats["n_agent_nodes"], 1)
    fail_rate = stats["n_join_failed_nodes"] / n_nodes_tot
    stats["join_fail_rate"] = round(fail_rate, 6)
    stats["all_zero_target_rate"] = round(stats["n_all_zero_target_nodes"] / n_nodes_tot, 6)
    stats["masked_frame_rate"] = round(
        stats["n_masked_frames"] / max(stats["n_total_frames"], 1), 6)
    stats["wall_minutes"] = round((time.time() - t0) / 60, 2)

    if fail_rate > max_join_fail_rate:
        raise RuntimeError(
            f"位置 join 失败率 {fail_rate:.2%} > 闸门 {max_join_fail_rate:.2%}"
            f"（{stats['n_join_failed_nodes']}/{n_nodes_tot} 个 agent 节点无法唯一匹配槽位）。"
            f"⚠️ 拒绝落盘：大面积失败会让 aux 退化成几乎无监督，"
            f"训练结果落到判据「不升」档并被误读成「瓶颈不在监督」。"
        )

    out_path = cache_path(root, adjacency_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"root": os.path.abspath(root), "adjacency_config": adjacency_config,
                "n_entries": len(entries), "stats": stats, "entries": entries}, out_path)
    logger.info("已写出 %s（%d 条）", out_path, len(entries))
    logger.info("统计：%s", json.dumps(stats, ensure_ascii=False))
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="闸⑥ agent aux 监督缓存构建")
    p.add_argument("--root", required=True, type=str, help="split leaf root（绝对路径）")
    p.add_argument("--adjacency_config", default="default",
                   help="必须与训练时的 --dataset_config 一致，否则节点顺序对不上")
    p.add_argument("--max_join_fail_rate", type=float, default=0.01,
                   help="位置 join 失败率闸门；超过即拒绝落盘（默认 1%%）")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(a.root, a.adjacency_config, a.max_join_fail_rate)
