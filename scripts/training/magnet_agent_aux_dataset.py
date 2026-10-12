"""闸⑥ —— 把离线 aux 缓存挂回 `GraphDataset`（**不改 `dataset.py`**）。

`GraphDataset` 的 `.pt` 里没有 `AGENT_FUTURES`，缓存由 `build_agent_aux_cache.py` 离线算好、
按 dataset 下标对齐。本类只做一件事：`get(idx)` 时把第 idx 条缓存挂到 `GraphData` 上。

🔴 **对齐不靠约定，靠断言**。缓存每条带 `n_nodes` 与 `gnn_x_sum` 两个指纹，
本类逐样本核对；不一致立刻抛 `AgentAuxAlignmentError`。
理由：错位不会报错，只会产出**系统性错误的监督信号**，而后果是
「闸⑥ 判据不成立 ⇒ 瓶颈不在监督」这一漂亮但错误的结论（阶段75 已在别处踩过一次）。

⚠️ PyG 批处理行为（已核 `GraphData.__inc__` / `__cat_dim__`，`dataset.py:260-276`）：
三个新字段名里都不含 `index` ⇒ 走 `super().__inc__` 返回 0（不加偏移）、`cat_dim=0`（沿 0 维拼接）。
`gnn_is_agent` 与 `gnn_x` 同为逐节点、同按 graph 顺序拼接 ⇒ 对齐保持；
`agent_future_tgt` 的行序 = 各 graph 内 agent 节点的出现顺序，与
`nonzero(gnn_is_agent)` 的结果**同序** ⇒ loss 侧可直接按行配对。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_agent_aux_cache import cache_path  # noqa: E402
from src.shared.utils.dataset import GraphDataset  # noqa: E402

logger = logging.getLogger(__name__)

FUTURE_FRAMES = 60

#: 指纹比对容差 —— `gnn_x_sum` 是 float32 求和，跨进程/跨版本可能有末位差异。
#: 取相对容差 1e-4：足以放过浮点噪声，又远小于"换了一个样本"会造成的差异。
FINGERPRINT_RTOL = 1e-4


class AgentAuxAlignmentError(RuntimeError):
    """aux 缓存与 dataset 样本对不上 —— 必须当场炸，不得静默。"""


class AgentAuxGraphDataset(GraphDataset):
    """`GraphDataset` + 三个 aux 字段（`gnn_is_agent` / `agent_future_tgt` / `agent_future_mask`）。"""

    def __init__(self, root, transform=None, pre_transform=None,
                 adjacency_config: str = "default") -> None:
        super().__init__(root, transform, pre_transform)
        cp = cache_path(root, adjacency_config)
        if not cp.exists():
            raise FileNotFoundError(
                f"缺 aux 缓存 {cp} —— 先跑 "
                f"`build_agent_aux_cache.py --root {root} --adjacency_config {adjacency_config}`")
        payload = torch.load(cp)
        self._aux: List[Dict] = payload["entries"]
        self._aux_stats = payload.get("stats", {})
        n_self = super().len()
        if len(self._aux) != n_self:
            raise AgentAuxAlignmentError(
                f"aux 缓存条数 {len(self._aux)} ≠ dataset 样本数 {n_self} —— "
                f"缓存与 `.pt` 不是同一批数据（root 或 adjacency_config 不一致？）")
        logger.info("AgentAuxGraphDataset: %d 样本，aux 缓存 %s，join 失败率 %.4f%%",
                    n_self, cp.name, 100 * float(self._aux_stats.get("join_fail_rate", 0.0)))

    def get(self, idx):                                   # noqa: D102  见类 docstring
        data = super().get(idx)
        e = self._aux[idx]

        n_nodes = int(data.gnn_x.shape[0]) if getattr(data, "gnn_x", None) is not None else 0
        if n_nodes != int(e["n_nodes"]):
            raise AgentAuxAlignmentError(
                f"样本 {idx} 节点数对不上：dataset {n_nodes} vs 缓存 {e['n_nodes']} ⇒ 缓存错位")
        if n_nodes:
            fp = float(data.gnn_x.sum())
            ref = float(e["gnn_x_sum"])
            if abs(fp - ref) > FINGERPRINT_RTOL * max(abs(ref), 1.0):
                raise AgentAuxAlignmentError(
                    f"样本 {idx} 指纹对不上：dataset gnn_x.sum()={fp:.6f} vs 缓存 {ref:.6f} ⇒ 缓存错位")

        data.gnn_is_agent = e["is_agent"]
        data.agent_future_tgt = e["tgt"]
        data.agent_future_mask = e["mask"]
        return data


__all__ = ["AgentAuxAlignmentError", "AgentAuxGraphDataset"]
