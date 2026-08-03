"""Context 分支 polyline 池化（VectorNet 式 encode-then-pool），**默认关闭、flag 门控**。

背景（判据 0 实测，`scratchpad/branch_cost_probe_20260730.py`，20 个 val 场景）：

    分支            MFLOPs      占比
    ego              11.2       0.1%
    context        7973.0      99.8%      <- M 中位数 = 2036 个 segment token
    gnn               1.1       0.014%    <- 用的是已池化的 44 个 polyline 节点
    fusion            0.1       0.0%

即：`gnn_x` 走的是 `subgraph_pooling()` 的 polyline 级表示（44 节点），而 context 分支
对 **2036 个未池化 segment token** 做 3 层 O(M²) 全局自注意力。`cluster`（col9=subgraph_ID）
在 `GraphData` 里**已经算好并存着**，只是 context 分支从不使用它。

本模块按 VectorNet(CVPR 2020) 的层级顺序补上这一步：
`input_proj(9->128) + type_embed` 之后、进 `context_transformer` 之前，按 polyline 池化。

🔴 **必须处理的地雷（D0#7 同族）**：`GraphData.__inc__` 对 `'cluster'` **返回 0**
（`src/shared/utils/dataset.py:266-267`）⇒ PyG 批处理时 cluster ID **不做样本间偏移**，
样本 0 的 cluster 5 与样本 1 的 cluster 5 是同一个值。若直接 `global_max_pool(x, cluster)`，
**跨样本的 polyline 会被合并**，产生与 D0#7 完全同型的静默串样本缺陷。
本实现按 `(sample_idx, cluster)` 组合出全局唯一 ID，并对每次调用做**严格的行数校验**，
不一致即抛错而非静默退化。

实现方式：**不修改 devkit vendor 的 `transformer_predictor.py`**。父类在
`transformer_predictor.py:234-248` 逐样本调用 `self.context_transformer(sample_x)`，
本模块把该子模块换成一个带池化的包装器，因此前向语义的其余部分与父类逐字对齐。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parents[0]
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_geometric.nn import global_max_pool, global_mean_pool  # noqa: E402

from magnet_imitation_pooled import MagnetImitationPooled  # noqa: E402

__all__ = [
    "ClusterCoverageError",
    "ContextPoolAlignmentError",
    "MagnetContextPooled",
    "CONTEXT_POOLING_TAG",
]

CONTEXT_POOLING_TAG = "context_pooling_v1"

_POOL_FNS = {"max": global_max_pool, "mean": global_mean_pool}


class ClusterCoverageError(RuntimeError):
    """`cluster` 缺失、长度不符或含负值 —— 拒绝执行，不静默退化。"""


class ContextPoolAlignmentError(RuntimeError):
    """包装器收到的 token 行数与预期样本不符 —— 说明消费顺序被打乱，拒绝执行。"""


def _per_sample_clusters(data) -> List[torch.Tensor]:
    """按父类的遍历顺序切出每个样本的 cluster 向量。

    父类 `transformer_predictor.py:234` 是 `for i in range(batch_size)` 且
    `mask = data.batch == i`，故这里必须用完全相同的顺序与掩码。
    """
    cluster = getattr(data, "cluster", None)
    if cluster is None:
        raise ClusterCoverageError("data.cluster 不存在，无法做 context 池化")
    cluster = cluster.reshape(-1)
    if cluster.numel() != data.x.size(0):
        raise ClusterCoverageError(
            f"cluster 长度 {cluster.numel()} != x 行数 {data.x.size(0)}"
        )
    if int(cluster.min()) < 0:
        raise ClusterCoverageError(f"cluster 含负值 min={int(cluster.min())}")

    batch = getattr(data, "batch", None)
    if batch is None:
        return [cluster]
    n = int(batch.max().item()) + 1
    return [cluster[batch == i] for i in range(n)]


class _PooledContextTransformer(nn.Module):
    """包装父类的 `context_transformer`：先按 polyline 池化，再进真正的 Transformer。

    父类每次前向会按样本顺序调用本模块 `batch_size` 次；`reset(...)` 在每次
    `MagnetContextPooled.forward` 开始时装入该批的 per-sample cluster 列表。
    """

    def __init__(self, inner: nn.Module, mode: str = "max") -> None:
        super().__init__()
        if mode not in _POOL_FNS:
            raise ValueError(f"unknown pooling mode: {mode}")
        self.inner = inner
        self.mode = mode
        self._queue: Optional[List[torch.Tensor]] = None
        self._cursor = 0

    def reset(self, clusters: Optional[List[torch.Tensor]]) -> None:
        self._queue = clusters
        self._cursor = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [1, M_i, 128]
        if self._queue is None:  # 未启用：与父类逐字节一致
            return self.inner(x)

        if self._cursor >= len(self._queue):
            raise ContextPoolAlignmentError(
                f"context_transformer 第 {self._cursor + 1} 次调用，但本批只有 "
                f"{len(self._queue)} 个样本的 cluster"
            )
        cl = self._queue[self._cursor]
        self._cursor += 1

        if x.dim() != 3 or x.size(0) != 1:
            raise ContextPoolAlignmentError(f"预期 [1, M, D]，收到 {tuple(x.shape)}")
        if x.size(1) != cl.numel():
            raise ContextPoolAlignmentError(
                f"第 {self._cursor} 个样本 token 数 {x.size(1)} != cluster 数 {cl.numel()}；"
                "消费顺序与父类循环不一致，拒绝执行"
            )

        # 单样本内 cluster 已唯一，压实成 0..P-1（PyG pool 要求连续下标）
        _, inv = torch.unique(cl, sorted=True, return_inverse=True)
        pooled = _POOL_FNS[self.mode](x[0], inv.to(x.device))  # [P, D]
        return self.inner(pooled.unsqueeze(0))


class MagnetContextPooled(MagnetImitationPooled):
    """父类（含 D0#7 GNN 逐样本池化）+ context 分支 polyline 池化。

    `context_pooling=False` 时与 `MagnetImitationPooled` **逐字节一致**。
    """

    def __init__(
        self,
        strict_coverage: bool = True,
        context_pooling: bool = False,
        context_pool_mode: str = "max",
    ) -> None:
        super().__init__(strict_coverage=strict_coverage)
        self.context_pooling = bool(context_pooling)
        self.context_pool_mode = context_pool_mode
        # 只包装、不替换权重：inner 仍是父类构造的那个 TransformerEncoder
        self.context_transformer = _PooledContextTransformer(
            self.context_transformer, mode=context_pool_mode
        )

    def _prepare(self, data):
        self.context_transformer.reset(
            _per_sample_clusters(data) if self.context_pooling else None
        )
        return data

    def forward(self, data):  # type: ignore[override]
        return super().forward(self._prepare(data))

    def get_encoder_features(self, data):  # type: ignore[override]
        return super().get_encoder_features(self._prepare(data))
