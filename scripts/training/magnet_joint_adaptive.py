"""路线 B 新结构 S1 第 4 件 —— **模型类 `MagnetJointAdaptive`**（Q110 决策 D-e）。

把前三件串成一个模型：::

    data.x [n_vec,12] ──EntityEncoder──> h⁰ [n_entity,128]      ← 取代 subgraph_pooling 的 mean
    EGO_HISTORY [B,20,2] ──EgoHistoryEncoder──> 融进**节点 0**   ← 🔴 不新增 token（§3.2 修订）
    h⁰ ──JointInteractionStack（2 层 relation-aware MP + 单个 scorer）──> h', w
    h', w ──JointModeDecoder（per-entity × per-mode query，attn_logit += log(w+ε)）──> [n_entity,M,60,2]

🔴 **整条预测路径上没有任何 global pooling（P1）**，输出是 `[n_entity, M, 60, 2]` 而不是 `[1,60,2]`。

## 🔴 D-e：注册成**顶层分支**，不是旋钮

它不继承 `TransformerPredictor` 家族的任何一个类 —— 三条旧分支（ego / context / gnn）在这里
被**合并**成一套 per-entity token，旧旋钮（`context_pooling` / `ego_query_fusion` / `agent_aux` /
`head_norm` / `gnn_operator`）在新结构里**没有对应物**。
⇒ `bn_calibration.build_model` 必须把 `joint_adaptive` 放在**最前面**判断，并对任何组合**显式拦截**
（照 `multimodal` / `temporal_graph` / `agent_aux` 的既有写法），不得静默择一。

## 🔴 P4：ego 编码器的 BatchNorm 换成 LayerNorm

父类 `ego_history_encoder` 是 `Linear(2,64) → BatchNorm1d(64, track_running_stats=False) → ReLU → Linear(64,128)`。
那个 BN 是**batch 依赖的唯一来源**（D0#13：开环 b=32 得 2.194、闭环 b=1 得 4.495；
新档 test 也因此把 batch 冻死在 8）。本类换成 `LayerNorm(64)`
⇒ **BN calibration 这一整步连同它的口径纠纷一并消失**，V-7（batch 不变性）应当逐位成立。

## 🔴 输出与 target 的对接

输出是**绝对 ego-centric 位置**，与 `trajectory_contract_joint.build_joint_targets` 的
`ego_tgt` / `agent_tgt` **同口径**；行序 = `gnn_x` 行序 = `is_agent` 的行序
⇒ 周车侧用 `is_agent` 取子集即可与 `agent_tgt` 对齐（槽位 join 已由 `resolve_agent_slots()` 解决）。
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn as nn

from magnet_joint_decoder import (DEFAULT_NUM_MODES, FUTURE_FRAMES, JointModeDecoder,
                                  JointPrediction, select_mode_argmax)
from magnet_joint_encoder import EntityEncoder, EntityIndexError, build_entity_index
from magnet_joint_layers import (JointInteractionStack, build_edge_context,
                                 build_entity_geometry, isolate_edge_gate)

#: `EGO_HISTORY` 的帧数（2 s @ 10 Hz）。🔴 **不是 4 s** —— 前 20 帧曾是零占位，
#: 论文里不得写 "4 s history"（`feature_schema_v2.md` §3-4）。
EGO_HISTORY_FRAMES: int = 20

#: 本类的版本串，写进 sidecar 的 `architecture.class` 旁边，便于日后区分实现代次。
JOINT_ADAPTIVE_VERSION = "MAGNET_JOINT_ADAPTIVE_v1_20260825"


class EgoHistoryEncoder(nn.Module):
    """ego 历史（20 帧逐帧增量）→ 一个 128 维向量。

    🔴 **P4**：与父类逐层同构，唯一差别 = `BatchNorm1d(64, track_running_stats=False)` → `LayerNorm(64)`。
    另加两处新结构才有的东西：**帧位置嵌入**（否则 20 帧是无序集合）与
    **learned-query attention pooling**（与 `EntityEncoder` 同一套做法，不用 mean）。
    """

    def __init__(self, dim: int = 128, hidden: int = 64,
                 frames: int = EGO_HISTORY_FRAMES) -> None:
        super().__init__()
        self.dim, self.frames = dim, frames
        self.frame_mlp = nn.Sequential(
            nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.frame_pos = nn.Embedding(frames, dim)
        self.pool_query = nn.Parameter(torch.empty(dim))
        self.pool_key = nn.Linear(dim, dim)
        self.pool_val = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, ego_history: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """:param ego_history: `[B*20, 2]`（PyG 批处理后的形态）或 `[20, 2]`（单样本）。
        :returns: `[B, dim]`
        :raises EntityIndexError: 形状不符 —— **不猜、不 pad**，静默降级会把 ego 分支变成噪声。
        """
        eh = ego_history
        if eh.dim() == 3:                                    # 少数调用方已经 reshape 过
            eh = eh.reshape(-1, eh.shape[-1])
        if eh.dim() != 2 or eh.shape[-1] < 2:
            raise EntityIndexError(f"ego_history 形状 {tuple(ego_history.shape)} 不可用")
        if eh.shape[0] != n_graphs * self.frames:
            raise EntityIndexError(
                f"ego_history 有 {eh.shape[0]} 行，但批内 {n_graphs} 个样本 × {self.frames} 帧 "
                f"= {n_graphs * self.frames} —— 形状契约不成立")
        seq = eh[:, :2].view(n_graphs, self.frames, 2)

        h = self.frame_mlp(seq) + self.frame_pos.weight.unsqueeze(0)
        score = (self.pool_key(h) * self.pool_query).sum(dim=-1) / math.sqrt(self.dim)
        alpha = torch.softmax(score, dim=1).unsqueeze(-1)
        return self.out_norm((alpha * self.pool_val(h)).sum(dim=1))


class MagnetJointAdaptive(nn.Module):
    """联合多智能体多模态轨迹预测 + 反事实边效用监督的自适应交互拓扑（Q108 路线 B）。

    :param num_modes: 模态数 M（Q110 §3.5 定 6）。
    :param learn_topology: False = **固定/完全图对照臂**（`w ≡ 1`）。
        🔴 scorer 仍被实例化 ⇒ 两臂参数量逐位相同（Q110 §4-1）。
    :param n_mp_layers: message passing 层数（Q110 §3.3 定 2）。
    """

    def __init__(self, num_modes: int = DEFAULT_NUM_MODES, dim: int = 128,
                 n_mp_layers: int = 2, n_heads: int = 4, learn_topology: bool = True,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.dim, self.num_modes = dim, num_modes
        self.learn_topology = learn_topology
        self.version = JOINT_ADAPTIVE_VERSION

        self.encoder = EntityEncoder(dim=dim, dropout=dropout)
        self.ego_encoder = EgoHistoryEncoder(dim=dim)
        # §3.2：ego 的历史编码**融进节点 0**，token 数不变（新增 token 会让 n_entity 与
        # gnn_edge_index 的索引空间错位、图直接错）。V-13 断言 `n_entity == gnn_x.size(0)`。
        self.ego_fuse = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Linear(dim, dim))
        self.ego_fuse_norm = nn.LayerNorm(dim)
        self.stack = JointInteractionStack(dim=dim, n_layers=n_mp_layers, dropout=dropout,
                                           learn_topology=learn_topology)
        self.decoder = JointModeDecoder(dim=dim, num_modes=num_modes, n_heads=n_heads,
                                        dropout=dropout)
        self._reset_head_parameters()

    def _reset_head_parameters(self) -> None:
        """与 `EntityEncoder` / `JointInteractionStack` 同一套（xavier + zero bias）。"""
        for mod in (self.ego_encoder, self.ego_fuse, self.decoder):
            for m in mod.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.ego_encoder.pool_query, std=self.dim ** -0.5)

    def active_head(self) -> nn.Sequential:
        """给塌陷 / margin 探针的挂点（P5）。与既有子类同接口。"""
        return self.decoder.active_head()

    def forward(self, data, *, isolate: Optional[torch.Tensor] = None,
                edge_gate: Optional[torch.Tensor] = None,
                need_attn: bool = False) -> JointPrediction:
        """:param isolate: 要**完全孤立**的实体全局行号 `[k]`。
            🔴 它同时切边（两个方向）**并**硬屏蔽 decoder 对该 token 的注意力 ——
            两者缺一不可：只切边会走二阶泄漏 `j → lane → decoder → i`（D-b 的已知代价），
            只屏蔽注意力则 message passing 仍在传。**V-2 测的就是这个组合。**
        :param edge_gate: `[E]` 额外的逐边干预门（切边/剂量实验）。与 `isolate` 可叠加。
        """
        idx = build_entity_index(data)
        geom = build_entity_geometry(data.x, idx)
        ectx = build_edge_context(data.gnn_edge_index, idx, geom)
        n_graphs = int(idx.entity_batch.max().item()) + 1

        h = self.encoder(data.x, idx)
        h = self._fuse_ego(h, data, idx, n_graphs)

        gate, block = edge_gate, None
        if isolate is not None and isolate.numel():
            iso_gate = isolate_edge_gate(ectx, isolate)
            gate = iso_gate if gate is None else gate * iso_gate
            block = torch.zeros(idx.n_entity, dtype=torch.bool, device=h.device)
            block[isolate] = True

        out = self.stack(h, ectx, idx, edge_gate=gate)
        pred = self.decoder(out.tokens, out.edge_weight, ectx, idx, anchor=geom.pos,
                            key_block=block, need_attn=need_attn)
        # 🔴 透传 scorer 的 pre-sigmoid 分数供 S5 的边监督项用；固定臂是 None ⇒ 逐位不变。
        return dataclasses.replace(pred, edge_logit=out.edge_logit)

    def _fuse_ego(self, h: torch.Tensor, data, idx, n_graphs: int) -> torch.Tensor:
        """把 ego 历史编码融进**节点 0**（每个样本一个），token 数不变。"""
        eh = getattr(data, "ego_history", None)
        if eh is None or eh.numel() == 0:
            raise EntityIndexError("样本缺 `ego_history` —— 新结构的 ego 分支没有输入可用")
        ego_feat = self.ego_encoder(eh, n_graphs)                    # [B, dim]
        ego_rows = idx.ego_index                                     # [B]，契约 D：每样本节点 0
        fused = self.ego_fuse_norm(
            h[ego_rows] + self.ego_fuse(torch.cat([h[ego_rows], ego_feat], dim=-1)))
        return h.index_copy(0, ego_rows, fused)

    @torch.no_grad()
    def predict_closed_loop(self, data) -> torch.Tensor:
        """闭环用的单条轨迹：🔴 **D-i = `argmax π`，绝不做模态平均**。

        :returns: `[n_entity, 60, 2]` 绝对 ego-centric 位置；ego 的那一行是 planner 要的轨迹。
        """
        return select_mode_argmax(self.forward(data))[0]


__all__ = ["EGO_HISTORY_FRAMES", "FUTURE_FRAMES", "JOINT_ADAPTIVE_VERSION",
           "EgoHistoryEncoder", "MagnetJointAdaptive"]
