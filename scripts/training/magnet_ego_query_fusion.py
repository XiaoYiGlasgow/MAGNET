"""C2 靶向修复：**以 ego 表征为 query 的注意力池化**，替代两处 mean pooling。**默认关闭、flag 门控**。

诊断（2026-07-31 阶段㊵，三条**互相独立**的证据指向同一机制）
-------------------------------------------------------------
父类 `transformer_predictor.py` 把三条分支各自 pool 成一个 128-d 向量再 concat：

* `:53`   `GNNEncoder.forward` 末尾 `global_mean_pool(x, batch)`
* `:239`  context 分支 `sample_out.mean(dim=1)`

⇒ **整张图（ego + agents + lane）被平均成一个与 ego 无关的场景向量。**
代入 G19 实测的池规模（lane 恒为 40，agent 中位数 2，合计中位数 43）：

    agent 在 mean pool 里的权重 ≈ 2/43 = 4.7%

三条证据：

1. **G19 绝对量级**：删掉**全部** agent，轨迹只移动 **0.72%**（0.143 m / base_scale 19.81 m）；
2. **G19 剂量–响应**：1 个 agent 0.45% → ≥8 个 3.67%，**单调** ⇒ 不是「学会忽略」，是**被稀释**；
3. **G16 图拓扑**：30 m 近邻图 ≡ 完全图（ΔADE 0.119 m < σ_rep 0.283 m，且逐 seed 符号不一致）
   ⇒ **无论边怎么连，最后都过同一个 mean pool，拓扑差异在 pooling 处被抹掉。**

⚠️ **归属**：`阶段11` §4.2.1 的①诊断**早已写明「瓶颈在 route 缺失 + mean-pool」**。
本模块对应的实验是**量化并三方互证后的靶向修复**，写作时不得当作新发现。

修法与它答的审稿意见
--------------------
以 `ego_feat` 为 query、对**逐节点嵌入**做 multi-head attention pooling：
ego 不再只能看到一个场景平均向量，而是**逐 agent 学出权重**。

* **R1-1 验收路径①**（*"Without a learning mechanism … for topology selection"*）——
  注意力权重就是**学出来的、逐场景的、ego 条件化的软邻接**，拓扑不再是手选的
  `default`/`nearby_only`/`selective`。R3-M1、R4-2 是同一诉求。
* **R3-m1**（*"Does global mean pooling … preserve interaction effects? …may be washed out
  in the scene average"*）—— 审稿人原话问的就是这件事。
* 副产物：`last_attn_weights` 可直接作为**可解释性证据**（AE-3 / AE-5）。

🔴 **§7 写作硬约束**：attention 融合**不是新东西**（VectorNet / HDGT / PlanTF 已占）。
论文里必须**显式引用占据者**并定位为「**按审稿意见实施的修复**」，
**不得写成本文提出的方法**。novelty 落在「诊断决定了修什么 + 闭环验证了这条因果链」。

实现约束（照 `magnet_context_pooled.py` 的既定模式）
--------------------------------------------------
* **不修改 devkit vendor 的 `transformer_predictor.py`**；
* `ego_query_fusion=False` 时与父类 **逐字节一致**（判据 A，见单测）；
* 对齐不上就**抛错，不静默退化**；
* GNN 分支用 **forward hook 抓 `conv2` 的逐节点输出**，从而**完整复用父类的边界检查与卷积路径**，
  只把末尾的 `global_mean_pool` 换掉 —— 不复制父类逻辑，避免日后分叉。

一个必须知道的接口技巧
----------------------
父类在 `:238-239` 是 `sample_out = self.context_transformer(sample_x)` 然后
`sample_pooled = sample_out.mean(dim=1)`。本模块的包装器**返回 `[1, 1, D]`**（已池化成单 token），
于是父类那句 `.mean(dim=1)` 退化为恒等操作 —— 无需改父类一行。
"""
from __future__ import annotations

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

from magnet_context_pooled import MagnetContextPooled  # noqa: E402

__all__ = [
    "EgoQueryAlignmentError",
    "EgoQueryPool",
    "MagnetEgoQueryFusion",
    "EGO_QUERY_FUSION_TAG",
]

EGO_QUERY_FUSION_TAG = "ego_query_fusion_v1"


class EgoQueryAlignmentError(RuntimeError):
    """ego query 缺失、样本数对不上或消费顺序被打乱 —— 拒绝执行，不静默退化。"""


class EgoQueryPool(nn.Module):
    """以 ego 表征为 query 的注意力池化：`[1, N, D] × [1, D] -> [1, 1, D]`。

    与 mean pooling 的唯一区别是权重由数据决定而非恒为 `1/N` ——
    这正是 R1-1 要的「learning mechanism for topology selection」。
    """

    def __init__(self, dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        """`x`: `[1, N, D]` 逐节点嵌入；`ego`: `[1, D]` ego 表征。返回 `[1, 1, D]`。"""
        if x.dim() != 3 or x.size(0) != 1:
            raise EgoQueryAlignmentError(f"预期 x 为 [1, N, D]，收到 {tuple(x.shape)}")
        if ego.dim() != 2 or ego.size(0) != 1:
            raise EgoQueryAlignmentError(f"预期 ego 为 [1, D]，收到 {tuple(ego.shape)}")
        q = self.q_proj(ego).unsqueeze(1)                      # [1, 1, D]
        out, w = self.attn(q, x, x, need_weights=True)         # out [1,1,D], w [1,1,N]
        self.last_attn_weights = w.detach()                    # 可解释性证据，勿参与梯度
        return self.norm(out)


class _EgoAwareContextTransformer(nn.Module):
    """包装父类的 `context_transformer`：先跑真正的 Transformer，再做 ego-query 池化。

    父类每次前向按样本顺序调用本模块 `batch_size` 次，故用与
    `_PooledContextTransformer` 相同的游标模式对齐。
    返回 `[1, 1, D]` ⇒ 父类 `:239` 的 `.mean(dim=1)` 成为恒等操作。
    """

    def __init__(self, inner: nn.Module, pool: EgoQueryPool) -> None:
        super().__init__()
        self.inner = inner
        self.pool = pool
        self._ego: Optional[torch.Tensor] = None
        self._cursor = 0

    def reset(self, clusters) -> None:
        """🔴 **必须原样转发给 `inner`** —— 父类 `MagnetContextPooled._prepare` 会调
        `self.context_transformer.reset(<per-sample cluster 列表>)`。本类插在中间，
        若把这个 `reset` 据为己有，polyline 池化的 cluster 队列就永远装不进去
        （首次实现时就踩了这个坑：`AttributeError: 'list' object has no attribute 'size'`）。
        本类自己的状态走 `reset_ego`。"""
        self.inner.reset(clusters)

    def reset_ego(self, ego) -> None:
        self._ego = ego
        self._cursor = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        if self._ego is None:            # 未启用：与父类逐字节一致
            return out
        if self._cursor >= self._ego.size(0):
            raise EgoQueryAlignmentError(
                f"context_transformer 第 {self._cursor + 1} 次调用，"
                f"但本批只有 {self._ego.size(0)} 个 ego query"
            )
        ego_i = self._ego[self._cursor:self._cursor + 1]
        self._cursor += 1
        return self.pool(out, ego_i)


class _EgoAwareGNNEncoder(nn.Module):
    """包装父类的 `gnn_encoder`：用 forward hook 抓 `conv2` 的逐节点输出，再做 ego-query 池化。

    **不复制父类的边界检查逻辑** —— `inner` 照常执行（含它对空图 / 越界边的全部处理），
    我们只是把它末尾的 `global_mean_pool` 结果丢掉，改用 hook 抓到的逐节点嵌入重新池化。
    若 `inner` 因空图 / 无有效边提前返回（`conv2` 未执行 ⇒ hook 未触发），
    则**原样返回 `inner` 的输出**，保持父类的退化语义。
    """

    def __init__(self, inner: nn.Module, pool: EgoQueryPool) -> None:
        super().__init__()
        self.inner = inner
        self.pool = pool
        self._ego: Optional[torch.Tensor] = None
        self._node_out: Optional[torch.Tensor] = None
        inner.conv2.register_forward_hook(self._capture)

    def _capture(self, _module, _inp, out) -> None:
        self._node_out = out

    def reset_ego(self, ego) -> None:
        self._ego = ego

    def forward(self, x, edge_index, batch=None):
        self._node_out = None
        fallback = self.inner(x, edge_index, batch)
        if self._ego is None or self._node_out is None:
            return fallback

        node_out = self._node_out
        if batch is None:
            return self.pool(node_out.unsqueeze(0), self._ego[0:1]).squeeze(1)

        n = int(batch.max().item()) + 1
        if n != self._ego.size(0):
            raise EgoQueryAlignmentError(
                f"GNN batch 内样本数 {n} != ego query 数 {self._ego.size(0)}"
            )
        pooled = []
        for i in range(n):
            mask = batch == i
            if int(mask.sum()) == 0:
                raise EgoQueryAlignmentError(f"GNN batch 第 {i} 个样本没有任何节点")
            xi = node_out[mask].unsqueeze(0)                    # [1, N_i, D]
            pooled.append(self.pool(xi, self._ego[i:i + 1]).squeeze(1))
        return torch.cat(pooled, dim=0)                         # [B, D]


class MagnetEgoQueryFusion(MagnetContextPooled):
    """父类（D0#7 GNN 逐样本池化 + 判据0 context polyline 池化）+ **C2 ego-query 融合**。

    `ego_query_fusion=False` 时与 `MagnetContextPooled` **逐字节一致**。
    """

    def __init__(
        self,
        strict_coverage: bool = True,
        context_pooling: bool = False,
        context_pool_mode: str = "max",
        ego_query_fusion: bool = False,
        ego_query_heads: int = 4,
    ) -> None:
        super().__init__(
            strict_coverage=strict_coverage,
            context_pooling=context_pooling,
            context_pool_mode=context_pool_mode,
        )
        self.ego_query_fusion = bool(ego_query_fusion)
        self.ego_query_heads = int(ego_query_heads)

        # 两条分支各自一份池化器（不共享参数：context 是 polyline/segment 语义，
        # gnn 是 polyline 图节点语义，共享会把两种语义强行对齐）
        self.context_pool = EgoQueryPool(128, ego_query_heads)
        self.gnn_pool = EgoQueryPool(128, ego_query_heads)

        self.context_transformer = _EgoAwareContextTransformer(
            self.context_transformer, self.context_pool
        )
        self.gnn_encoder = _EgoAwareGNNEncoder(self.gnn_encoder, self.gnn_pool)

        # ego 分支的输出要在两条分支之前拿到 ⇒ 挂 hook 抓 `ego_transformer` 的输出。
        # 父类 `:210-211` 是 `ego_feat_seq = self.ego_transformer(...)` 后 `.mean(dim=1)`，
        # 这里复刻同一个归约，得到的就是父类真正用作 ego_feat 的那个张量。
        self._ego_feat: Optional[torch.Tensor] = None
        self.ego_transformer.register_forward_hook(self._capture_ego)

    def _capture_ego(self, _module, _inp, out) -> None:
        self._ego_feat = out.mean(dim=1)          # [B, 128]，与父类 :211 一致

    def _prepare_ego_query(self, data):
        """跑一次 ego 分支拿 query。**必须在 context/gnn 分支之前**。"""
        if not self.ego_query_fusion:
            self.context_transformer.reset_ego(None)
            self.gnn_encoder.reset_ego(None)
            return
        self._ego_feat = None
        # 父类 forward 里 ego 分支在最前，hook 会在其中触发；这里只需先清空，
        # 由 `forward` 的父类调用负责填充 —— 但 context 分支紧随其后，
        # 所以必须在父类 forward 之前把 reset 做好，填充值由 hook 在同一次前向内写入。
        self.context_transformer.reset_ego(_LazyEgo(self))
        self.gnn_encoder.reset_ego(_LazyEgo(self))

    def forward(self, data):  # type: ignore[override]
        self._prepare_ego_query(data)
        return super().forward(data)

    def get_encoder_features(self, data):  # type: ignore[override]
        self._prepare_ego_query(data)
        return super().get_encoder_features(data)


class _LazyEgo:
    """惰性代理：ego 分支在父类 forward 内部才跑完，池化器取值时才解析。

    直接传张量不行 —— `reset` 发生在父类 forward **之前**，那时 `_ego_feat` 还是 None。
    本代理把 `size(0)` 与切片都转发到宿主模型当次前向真正算出的 `_ego_feat`。
    """

    def __init__(self, host: "MagnetEgoQueryFusion") -> None:
        self._host = host

    def _resolve(self) -> torch.Tensor:
        t = self._host._ego_feat
        if t is None:
            raise EgoQueryAlignmentError(
                "ego_query_fusion 已启用，但本次前向没有捕获到 ego_transformer 的输出。"
                "父类 transformer_predictor.py:186-190 存在一条不经过 ego_transformer 的分支"
                "（ego_history 形状为 [batch_size, >=2] 时），该分支下无法构造 ego query。"
                "请检查 EGO_HISTORY 是否为约定的 [20, 2]。"
            )
        return t

    def size(self, dim: int) -> int:
        return self._resolve().size(dim)

    def __getitem__(self, item):
        return self._resolve()[item]
