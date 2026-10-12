"""路线 B 新结构 S1 第 3 件 —— **joint 多模态 decoder**（Q110 §3.5，含 D-f 定案）。

## 它取代了什么

旧结构在这里做**第三次**均值池化：`global_mean_pool` 把 83 个节点压成 **1 个 128 维向量**，
再由 `output_head` 吐出**一条** ego 轨迹。⇒ 路线 B 要的 ≈14,400 维联合输出**写不出来**。

本模块改成 **per-entity query 解码**：每个预测目标 i（ego + 全部 agent）、每个模态 m 各出一条 query，
去 cross-attend **关系增强后的全部实体 token**（🔴 **不是**一个场景向量）。
P1 因此在整条预测路径上成立：**从输入到输出没有任何 global pooling**。

## 🔴 D-f（已定案 = (1)）：decoder 必须受拓扑管辖

按朴素写法，query 的 cross-attention 读**全部**实体 token，**不受 `w_{j→i}` 约束** ——
即便 `w=0` 掐断了 message passing，j 的信息仍可经 decoder 直达 i 的预测。
后果不是测试问题而是**主张问题**：「学到的拓扑控制信息流」结构上不成立、G3 的 faithfulness 被削弱、
G5 的效应量被压扁（两臂共享同一条不受管辖的旁路）。

定案形式**写死**为::

    attn_logit_{i,j}  +=  log(w_{j→i} + EPS_GATE)          EPS_GATE = 1e-6

且 **与 message passing 共用同一套 `w`**（由 `JointInteractionStack` 返回）——
用第二个打分头会让论文里出现**两个拓扑**，"learned topology" 指哪个说不清。
选它还因为它是 attention masking 的标准形式，与 PlanTF 的 `attn_mask` 同构，
X 探针已实测那条通路可用 ⇒ 跨仪器对照将来能直接接上。

🔴🔴 **必须同时写明的限制：softmax 会重新归一化。**
给目标 i 的**所有**边同时乘一个常数，attention 完全不变 —— 只有**相对**权重有意义。

* ✅ 对 **faithfulness 无害**：切**一条** `j→i` 时其余边不动，该边注意力归零、质量重分配，正是 G3 要测的；
* 🔴 对**稀疏性主张有害**：该机制**永远无法把拓扑整体关掉**
  ⇒ **正文不得声称「模型学会只用少数几条边」这类全局稀疏性**。

⚠️ 因为是「压到 1e-6」而不是「恰好 0」，**V-2 的严格隔离必须另给 `key_block`**（硬 mask 掉那个 token），
只靠 `w=0` 拿不到逐位的 0。这不是实现瑕疵，是 softmax 重归一化的直接后果，已如实写进上面的限制。

## 🔴 图里没有的边 = 不可见（不是「不受约束」）

稠密边权矩阵**初值为 0**（⇒ bias = log(1e-6)），只有 `edge_index` 里真实存在的边才被填上 `w`；
对角线（自注意力）对**有效**实体置 1。⇒ `nearby_only` 之类的稀疏拓扑臂，其 decoder 也同样受限，
四个 topology 臂的语义因此一致（Q109 §6.1 要求同容量/同 mode 数/同预算，语义也得同）。

## 🔴 输出口径

输出 = **绝对 ego-centric 位置**，与 `trajectory_contract_joint.build_joint_targets` **逐位同口径**
（Q109 §3.2 强制契约；Q106 已查实增量域会奖励恒定输出）。
实现上锚定在实体**当前位置**上（`traj = geom.pos[i] + Δ`）：agent 到原点的距离中位数实测 **20–40 m**、
max **76 m**，不锚定等于让网络先学会「把自己搬过去」。
⚠️ 锚定的是**位置**不是**逐帧增量** ——「原地不动」在位置域仍要吃满全部位移误差，
不会重蹈 Q106 查实的「增量域奖励恒定输出」。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from magnet_joint_encoder import EntityIndex
from magnet_joint_layers import EPS_GATE, EdgeContext

#: 预测帧数（6 s @ 10 Hz），与 `trajectory_contract_joint.FUTURE_FRAMES` 一致。
FUTURE_FRAMES: int = 60

#: 模态数 M。与 `magnet_multimodal.py` 现有默认一致（Q110 §3.5）。
DEFAULT_NUM_MODES: int = 6

#: softmax 里给不可见 key 用的有限大负数（不用 `-inf`：整行全屏蔽会得 NaN 并污染整个 batch）。
_NEG = -1e9


@dataclass(frozen=True)
class JointPrediction:
    """联合多模态预测结果。**全部按 `gnn_x` 的行序**（= 实体全局行号）。

    :ivar traj: `[n_entity, M, 60, 2]` 绝对 ego-centric 位置；padding 实体的行恒为 0。
    :ivar logits: `[n_entity, M]` 模态 logit（未归一化）；padding 实体的行恒为 0。
    :ivar attn: `[B, N, N]` 逐目标对逐 key 的注意力（模态平均后），供 faithfulness 探针用；
        `None` 表示未请求。
    :ivar edge_logit: `[E]` edge scorer 的 pre-sigmoid 分数，供 S5 的边监督项用；
        固定臂恒为 `None`。
    """

    traj: torch.Tensor
    logits: torch.Tensor
    attn: Optional[torch.Tensor] = None
    edge_logit: Optional[torch.Tensor] = None


def dense_edge_weight(w: torch.Tensor, ectx: EdgeContext, idx: EntityIndex,
                      dense_mask: torch.Tensor) -> torch.Tensor:
    """把 `[E]` 的边权散成 `[B, N, N]` 的稠密矩阵（`[b, i, j] = w_{j→i}`）。

    :param dense_mask: `to_dense_batch` 的 `[B, N]` 有效位（**含** padding 实体，它们在这里也算 slot）。
    :returns: `[B, N, N]`；图里没有的边保持 **0**（⇒ 不可见），对角线对有效实体置 1（自注意力）。
    """
    b_size, n_max = dense_mask.shape
    counts = torch.bincount(idx.entity_batch, minlength=b_size)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])

    src, dst = ectx.edge_index[0], ectx.edge_index[1]
    gb = idx.entity_batch[dst]                       # 边不跨样本，取任一端即可
    wd = w.new_zeros(b_size, n_max, n_max)
    wd[gb, dst - offsets[gb], src - offsets[gb]] = w
    diag = torch.arange(n_max, device=w.device)
    wd[:, diag, diag] = dense_mask.to(w.dtype)       # 自注意力恒开（padding slot 除外）
    return wd


class JointModeDecoder(nn.Module):
    """per-entity × per-mode query 的 cross-attention decoder（Q110 §3.5）。

    🔴 **D-a**：ego 与周车**共用同一套权重**，只由 `is_ego` type embedding 区分 ——
    各一套会让 ego 分支退化回旧模型，且与「预测任何一个交通参与者」的联合叙事相悖。
    """

    def __init__(self, dim: int = 128, num_modes: int = DEFAULT_NUM_MODES,
                 n_heads: int = 4, dropout: float = 0.1,
                 future_frames: int = FUTURE_FRAMES) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim={dim} 不能被 n_heads={n_heads} 整除")
        self.dim, self.num_modes, self.n_heads = dim, num_modes, n_heads
        self.head_dim, self.future_frames = dim // n_heads, future_frames

        self.mode_emb = nn.Embedding(num_modes, dim)
        self.is_ego_emb = nn.Embedding(2, dim)        # D-a：ego / 周车共用权重，靠它区分
        self.q_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.z_norm = nn.LayerNorm(dim)
        # P3：前置归一化。与 PlanTF `trajectory_decoder` 的 loc 头同构。
        self.traj_head = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, future_frames * 2))
        self.mode_head = nn.Linear(dim, 1)

    def active_head(self) -> nn.Sequential:
        """给塌陷/margin 探针的显式挂点（与 `MagnetImitationMultiModal` / `MagnetHeadNorm` 同接口）。"""
        return self.traj_head

    def forward(self, tokens: torch.Tensor, w: torch.Tensor, ectx: EdgeContext,
                idx: EntityIndex, anchor: torch.Tensor,
                key_block: Optional[torch.Tensor] = None,
                need_attn: bool = False) -> JointPrediction:
        """:param tokens: `[n_entity, dim]` 关系增强后的实体 token。
        :param w: `[E]` **与 MP 共用的**边权（D-f）。
        :param anchor: `[n_entity, 2]` 每个实体的当前位置（输出锚点）。
        :param key_block: `[n_entity]` bool，True = **硬屏蔽**该 token 的注意力。
            🔴 V-2 的严格隔离要它 —— 只靠 `w=0` 得到的是 1e-6 而不是逐位 0（见模块文档）。
        :param need_attn: True 时一并返回模态平均后的注意力（faithfulness 探针用）。
        """
        h_d, mask = to_dense_batch(tokens, idx.entity_batch)          # [B, N, D], [B, N]
        b_size, n_max, dim = h_d.shape
        m, heads, hd = self.num_modes, self.n_heads, self.head_dim

        visible = mask & to_dense_batch(idx.entity_valid, idx.entity_batch,
                                        fill_value=False)[0]          # P8：padding 不进注意力
        if key_block is not None:
            visible = visible & ~to_dense_batch(key_block, idx.entity_batch, fill_value=False)[0]

        # ---- query：h_i + mode_emb[m] + is_ego_emb（D-a） ----------------------------------
        is_ego = torch.zeros(idx.n_entity, dtype=torch.long, device=tokens.device)
        is_ego[idx.ego_index] = 1
        ego_d = to_dense_batch(self.is_ego_emb(is_ego), idx.entity_batch)[0]      # [B, N, D]
        q = h_d.unsqueeze(2) + ego_d.unsqueeze(2) + self.mode_emb.weight.view(1, 1, m, dim)
        q = self.q_proj(self.q_norm(q)).reshape(b_size, n_max * m, heads, hd)

        k = self.k_proj(h_d).reshape(b_size, n_max, heads, hd)
        v = self.v_proj(h_d).reshape(b_size, n_max, heads, hd)

        # ---- 🔴 D-f：attn_logit += log(w + ε)，w 与 MP 共用 --------------------------------
        bias = torch.log(dense_edge_weight(w, ectx, idx, mask) + EPS_GATE)        # [B, N, N]
        bias = bias.repeat_interleave(m, dim=1)                                   # [B, N*M, N]

        logit = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd) + bias.unsqueeze(1)
        logit = logit.masked_fill(~visible[:, None, None, :], _NEG)
        attn = torch.softmax(logit, dim=-1)
        z = torch.einsum("bhts,bshd->bthd", attn, v).reshape(b_size, n_max * m, dim)
        z = self.z_norm(q.reshape(b_size, n_max * m, dim) + self.o_proj(z))       # 残差

        traj = self.traj_head(z).view(b_size, n_max, m, self.future_frames, 2)
        logits = self.mode_head(z).view(b_size, n_max, m)

        # ---- 稠密 → 扁平（按 `gnn_x` 行序），并锚定到实体当前位置 --------------------------
        keep = idx.entity_valid.unsqueeze(-1).to(traj.dtype)
        flat_traj = traj[mask] + anchor.view(idx.n_entity, 1, 1, 2)
        flat_traj = flat_traj * keep.view(idx.n_entity, 1, 1, 1)
        flat_logits = logits[mask] * keep
        out_attn = attn.mean(dim=1).view(b_size, n_max, m, n_max).mean(dim=2) if need_attn else None
        return JointPrediction(traj=flat_traj, logits=flat_logits, attn=out_attn)


def select_mode_argmax(pred: JointPrediction) -> Tuple[torch.Tensor, torch.Tensor]:
    """🔴 **D-i**：闭环模态选择 = `argmax π`。

    **绝不做模态平均** —— 平均会造出物理上不存在的中间轨迹（两条绕行方案的均值是直穿障碍物）。
    与 PlanTF `planning_model.py:135` 的 `best_mode = probability.argmax(dim=-1)` 同口径。

    :returns: `(traj [n_entity, 60, 2], mode_index [n_entity])`
    """
    best = pred.logits.argmax(dim=-1)
    sel = pred.traj[torch.arange(pred.traj.shape[0], device=best.device), best]
    return sel, best


__all__ = ["FUTURE_FRAMES", "DEFAULT_NUM_MODES", "JointPrediction", "JointModeDecoder",
           "dense_edge_weight", "select_mode_argmax"]
