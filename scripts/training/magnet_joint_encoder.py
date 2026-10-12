"""路线 B 新结构 S1 第 1 件 —— **实体编码器**（Q110 §3.2）。

## 它取代了什么

旧结构在到达 decoder 之前有**三次连续均值池化**（Q110 §1.1）：
`subgraph_pooling` mean → 完全图 `GCNConv` ≡ `MLP(mean)` → `global_mean_pool`。
本模块取代**第一次**，并让后两次失去存在理由 —— 它直接吃**未池化的逐向量输入** `data.x`，
按 subgraph 分组，用 **learned-query attention pooling** 产出**每个实体一个 token**。

🔴 **P1**：产出的是 `[n_entity, dim]`，不是 `[1, dim]`。预测路径上不再有任何 global pooling。
🔴 **P2**：`dim` 默认 **128，本批不加宽**。这一步把「整个场景 128 维」换成「每个实体 128 维」，
有效容量提高约 n_entity 倍而**编码器参数量不增**（token 之间权重共享）——这就是不需要加宽的原因。

## 🔴 四个已实测的输入契约（错一个就全盘错，故全部写成断言）

============  ===================================================================  ==============
契约          实测                                                                 本模块的用法
============  ===================================================================  ==============
**A 编号**    `subgraph_id ≡ gnn_x 的行号`；80/80 场景 `list(节点) == range(n)`，   按 ``x[:,9]`` 分组
              且 ``x[:,9]`` 的 id 集 **≡** ``gnn_x`` 节点集（60/60 场景）           即得 gnn_x 行号
**B 批处理**  🔴 ``GraphData.__inc__`` 对 ``cluster`` 返回 **0**（不偏移）。         必须自己加
              实测 4 样本批：``gnn_edge_index`` max=351（正确偏移）而               ``offsets[batch]``
              ``cluster`` max=93 ⇒ **直接分组会静默合并跨样本实体**                 （见 V-12）
**C 类型**    `_create_feature_mask` 启发式（time 列或 speed 列非零）在**实体级     **按实体投票**后
              100% 准确**（6770 实体/80 场景，混淆矩阵对角线满）；但 **15.27%       广播回它的向量
              的实体内部逐向量判定不一致**                                          （不得逐向量用）
**D ego**     **ego 恒为节点 0**（120/120 场景）。🔴 备选判据「离原点最近」          ``offsets[sample]``
              **不可用** —— 大量 padding 节点位置也是 (0,0) 会撞                    即该样本的 ego 行号
============  ===================================================================  ==============

## 🔴 P8：padding 全程不参与

实测（60 场景）：**44.4% 的图节点、57.8% 的 `data.x` 向量是纯 padding**，而旧结构**三处都没 mask**
（context transformer 没传 `src_key_padding_mask`、随后 mean、完全图与 `global_mean_pool` 也含空槽）
⇒ 所有均值的分母有 44% 是空气。

判据：**「除 col9（子图 id）外整行全零」** —— 2266/2266 个零位置 lane 节点全部命中，无假阳。

本模块的处理：**保留行、置零、并把 `entity_valid` 一路带下去**。
不删行是因为 `gnn_edge_index` 的索引空间绑定在 `gnn_x` 的行号上（契约 A），删行要重编号、
会牵动干预算子与 `resolve_agent_slots`。⇒ **行在、值为零、mask 为 False**，下游一律按 mask 排除。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import softmax as segment_softmax

#: `POLYLINE_FEATURES` 的前 9 维是送进编码器的 core 特征（与旧结构 `_extract_core_features` 同源）。
CORE_DIM: int = 9

#: col9 = subgraph_ID（`dataset.py:357` 注明「原 -1=col11 end_time 取错列」，勿再改回去）。
SUBGRAPH_ID_COL: int = 9

#: 判 padding 时要排除的列（只有 col9，它对 padding 行也是有效的 id）。
_NON_ID_COLS = [c for c in range(12) if c != SUBGRAPH_ID_COL]

#: 类型编码：0 = lane，1 = agent（含 ego）。与 `_create_feature_mask` 的 `is_agent.long()` 同向。
TYPE_LANE, TYPE_AGENT = 0, 1

#: 分段 softmax 里给无效位用的有限大负数。🔴 不用 `-inf`：
#: padding 实体的**全部**向量都无效，`-inf` 会让该段 softmax 得 NaN 并污染整个 batch。
_NEG = -1e9


class EntityIndexError(RuntimeError):
    """实体索引契约不成立 —— 必须暴露，不得静默降级成错误的分组。"""


@dataclass(frozen=True)
class EntityIndex:
    """把逐向量输入组织成逐实体所需的全部索引（不可变；只读结构）。

    :ivar global_eid: `[n_vec]` 每个向量所属实体的**全局**行号（= 批内 `gnn_x` 的行号）。
    :ivar n_entity: 批内实体总数 = `gnn_x.size(0)`。
    :ivar entity_batch: `[n_entity]` 每个实体属于哪个样本。
    :ivar entity_valid: `[n_entity]` bool，False = 纯 padding（P8）。
    :ivar entity_is_agent: `[n_entity]` bool，按契约 C 投票所得。
    :ivar ego_index: `[n_graphs]` 每个样本的 ego 实体全局行号（契约 D）。
    :ivar vec_valid: `[n_vec]` bool，False = 纯 padding 向量。
    :ivar vec_type: `[n_vec]` long，实体类型广播回向量（契约 C：**不是**逐向量判定）。
    """

    global_eid: torch.Tensor
    n_entity: int
    entity_batch: torch.Tensor
    entity_valid: torch.Tensor
    entity_is_agent: torch.Tensor
    ego_index: torch.Tensor
    vec_valid: torch.Tensor
    vec_type: torch.Tensor


def _vector_batch(data, n_vec: int, device: torch.device) -> torch.Tensor:
    """取逐向量的样本号；单样本时 PyG 的 `batch` 为 None，退化成全 0。"""
    b = getattr(data, "batch", None)
    return torch.zeros(n_vec, dtype=torch.long, device=device) if b is None else b


def _entity_batch_and_offsets(data, n_entity: int, n_graphs: int,
                              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """逐实体的样本号，以及每个样本在 `gnn_x` 里的起始行号（契约 B 的关键）。

    优先读 `gnn_x_batch`（DataLoader 的 `follow_batch=['gnn_x']` 产物，
    与 `magnet_imitation_pooled.attach_gnn_batch` 同源）；单样本时它不存在，退化成全 0。
    """
    eb = getattr(data, "gnn_x_batch", None)
    if eb is None:
        eb = getattr(data, "gnn_batch", None)
    if eb is None:
        if n_graphs != 1:
            raise EntityIndexError(
                f"批内有 {n_graphs} 个样本但缺 `gnn_x_batch` —— DataLoader 必须传 "
                f"`follow_batch=['gnn_x']`，否则实体无法按样本切分（契约 B）")
        eb = torch.zeros(n_entity, dtype=torch.long, device=device)
    counts = torch.bincount(eb, minlength=n_graphs)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return eb, offsets


@torch.no_grad()
def build_entity_index(data) -> EntityIndex:
    """从一个 `GraphData` / `Batch` 造出全部逐实体索引。

    :raises EntityIndexError: 四个契约中任意一个不成立。**不猜、不降级。**
    """
    x = data.x
    gnn_x = data.gnn_x
    device = x.device
    n_vec = int(x.shape[0])
    n_entity = int(gnn_x.shape[0])

    vec_batch = _vector_batch(data, n_vec, device)
    n_graphs = int(vec_batch.max().item()) + 1 if n_vec else 1
    entity_batch, offsets = _entity_batch_and_offsets(data, n_entity, n_graphs, device)

    # ---- 契约 A + B：局部 subgraph_id → 全局实体行号 ------------------------------------
    local_id = x[:, SUBGRAPH_ID_COL].round().long()
    if int(local_id.min()) < 0:
        raise EntityIndexError(f"x[:, {SUBGRAPH_ID_COL}] 出现负的 subgraph_id（min={int(local_id.min())}）")
    global_eid = local_id + offsets[vec_batch]
    if int(global_eid.max()) >= n_entity:
        raise EntityIndexError(
            f"global_eid 越界：max={int(global_eid.max())} >= n_entity={n_entity}。"
            f"这通常意味着 `cluster`/`x[:,9]` 与 `gnn_x` 的编号空间不一致（契约 A 被破坏）")
    covered = torch.zeros(n_entity, dtype=torch.bool, device=device)
    covered[global_eid] = True
    if not bool(covered.all()):
        raise EntityIndexError(
            f"{int((~covered).sum())}/{n_entity} 个实体在 `data.x` 里没有任何向量 —— "
            f"契约 A 要求二者是同一批实体；此时按实体分组会产出空 token")

    # ---- P8：padding 判据（除 col9 外整行全零）------------------------------------------
    entity_valid = gnn_x[:, _NON_ID_COLS].abs().sum(dim=1) > 0
    vec_valid = x[:, _NON_ID_COLS].abs().sum(dim=1) > 0

    # ---- 契约 C：逐向量 is_agent → **按实体投票** → 广播回向量 ---------------------------
    # 逐向量判据与 `transformer_predictor._create_feature_mask` 逐字同源（time 列或 speed 列非零）。
    vec_is_agent = ((x[:, 6] != 0) | (x[:, 7] != 0) | (x[:, 8] != 0)).float()
    vote_sum = torch.zeros(n_entity, device=device).index_add_(0, global_eid, vec_is_agent)
    vote_cnt = torch.zeros(n_entity, device=device).index_add_(
        0, global_eid, torch.ones_like(vec_is_agent))
    entity_is_agent = (vote_sum / vote_cnt.clamp(min=1.0)) > 0.5
    vec_type = torch.where(entity_is_agent[global_eid],
                           torch.full_like(local_id, TYPE_AGENT),
                           torch.full_like(local_id, TYPE_LANE))

    # ---- 契约 D：ego 恒为每个样本的节点 0 -------------------------------------------------
    ego_index = offsets.clone()

    return EntityIndex(global_eid=global_eid, n_entity=n_entity, entity_batch=entity_batch,
                       entity_valid=entity_valid, entity_is_agent=entity_is_agent,
                       ego_index=ego_index, vec_valid=vec_valid, vec_type=vec_type)


class EntityEncoder(nn.Module):
    """逐向量 → 逐实体 token。**取代 `subgraph_pooling` 的 mean**（Q110 §3.2）。

    为什么用 learned-query attention pooling 而不是 mean/max：
    * `mean` 是 D-2 的病灶（agent 位移衰减 19×，残留 5.26% = 1/(T−1)，数学必然）；
    * `max` 已被 A-max 格实测证明属「**四次干预抬高使用度而精度不动**」那一组（T3）⇒ 不是解药；
    * learned query 让「这条 polyline 里哪几个向量重要」变成**可学**的，且保留实体内部的形状信息。

    🔴 P3：输出投影一律 `Linear → LayerNorm → ReLU → Linear`（前置归一化）。
    """

    def __init__(self, dim: int = 128, core_dim: int = CORE_DIM, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.vec_norm = nn.LayerNorm(core_dim)
        self.vec_proj = nn.Linear(core_dim, dim)
        self.type_emb = nn.Embedding(2, dim)          # 0=lane, 1=agent
        self.embed_norm = nn.LayerNorm(dim)
        self.pool_query = nn.Parameter(torch.empty(dim))
        self.pool_key = nn.Linear(dim, dim)
        self.pool_val = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(                # P3：前置归一化
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """与旧结构**实际生效**的初始化一致（xavier + zero bias），不复制被覆盖的死代码。"""
        nn.init.normal_(self.pool_query, std=self.dim ** -0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, idx: EntityIndex) -> torch.Tensor:
        """:param x: `[n_vec, 12]` 原始 polyline 向量（本方法自己取前 9 维）。
        :param idx: :func:`build_entity_index` 的产物。
        :returns: `[n_entity, dim]`；**padding 实体的行恒为 0**（P8）。
        """
        h = self.vec_proj(self.vec_norm(x[:, :CORE_DIM]))
        h = self.embed_norm(h + self.type_emb(idx.vec_type))

        score = (self.pool_key(h) * self.pool_query).sum(dim=-1) / math.sqrt(self.dim)
        # 🔴 padding 向量不参与实体内聚合；用有限大负数而非 -inf（见模块常量 `_NEG` 的说明）
        score = torch.where(idx.vec_valid, score, torch.full_like(score, _NEG))
        alpha = segment_softmax(score, idx.global_eid, num_nodes=idx.n_entity)

        tok = torch.zeros(idx.n_entity, self.dim, dtype=h.dtype, device=h.device)
        tok.index_add_(0, idx.global_eid, alpha.unsqueeze(-1) * self.pool_val(h))
        tok = self.out_proj(tok)
        # 🔴 P8：padding 实体整行置零 —— 它的 softmax 段全是 `_NEG`，结果虽有限但无意义
        return tok * idx.entity_valid.unsqueeze(-1).to(tok.dtype)


__all__ = ["CORE_DIM", "SUBGRAPH_ID_COL", "TYPE_AGENT", "TYPE_LANE",
           "EntityIndex", "EntityIndexError", "EntityEncoder", "build_entity_index"]
