"""R1 时序场景图模型 —— Q76 重构的模型侧。

设计意图（用户原话）：「对于每一个时刻的多个 agent、ego、lanes 有一个图结构，
然后在时序上有很多个相同的图结构串联用 Transformer。每一个时刻是一个点，
在时序上串成一条线，但每个点都是一个多元素组成的图。」

现有 `TransformerPredictor` **没有实现这个** —— `subgraph_pooling` 在建图之前就把
19 帧沿时间轴 `np.mean` 掉了，只剩一个「20 帧平均场景」的静态图。

本模型只改 GNN 分支，**ego 与 context 两条分支逐字复用父类**，
保证 R1 的唯一变量是「GNN 分支是否逐帧」::

    父类 GNN 分支： gnn_x[N, 9] --GCN--> global_mean_pool --> gnn_feat[128]
    本类 GNN 分支： tg_x[T*N, 9] --GCN--> 按(样本,帧)池化 --> [B, T, 128]
                    --+时间位置编码--> 时序 Transformer --> 取末帧 --> gnn_feat[128]

🔑 **即使每帧内 GCN 在完全图上仍然恒等，h_1..h_T 的序列依然携带信息**
（场景如何演化）—— 信息从「图内结构」转移到「帧间变化」。
这是 R1 在 `full` 拓扑下也值得跑的理由；`knn30` 拓扑则让边本身逐帧变化。

读出用**末帧**而非 mean —— `global_mean_pool` 的 1/N 稀释是 R3 要修的东西，
不在 R1 里引入第二个池化。末帧 t=T-1 也是时间上最接近预测起点的一帧。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from transformer_predictor import TransformerPredictor

logger = logging.getLogger(__name__)

#: 与 `src/shared/utils/temporal_graph.NUM_FRAMES` 必须一致
NUM_FRAMES: int = 19


class TemporalPredictor(TransformerPredictor):
    """逐帧场景图 + 跨帧时序 Transformer。

    父类的 `ego_history_encoder` / `ego_transformer` / `input_proj` / `type_embedding` /
    `context_transformer` / `gnn_encoder` / `feature_fusion` / `output_head` 全部复用，
    ⇒ 旧 checkpoint 的这些权重可直接 `load_state_dict(strict=False)` 加载，
    只有本类新增的两个模块是随机初始化的。

    Args:
        num_frames: 时间步数，须与建图侧一致。
        temporal_layers: 时序 Transformer 层数。
        temporal_heads: 时序 Transformer 注意力头数。
    """

    def __init__(
        self,
        num_frames: int = NUM_FRAMES,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames

        # 🔴 时间位置编码是必须的 —— 现有 context 分支正是因为没有位置/时间编码，
        # 才退化成「对无序 token 集合做平均」。
        self.frame_pos_embedding = nn.Embedding(num_frames, 128)
        nn.init.normal_(self.frame_pos_embedding.weight, mean=0.0, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=temporal_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers
        )

        logger.info(
            "[R1] TemporalPredictor: T=%d, 时序 Transformer %d 层 %d 头, 读出=末帧",
            num_frames, temporal_layers, temporal_heads,
        )

    def _encode_temporal_gnn(self, data, batch_size: int, device) -> torch.Tensor:
        """逐帧图 -> GCN -> 按(样本,帧)池化 -> 时序 Transformer -> 末帧。

        Returns:
            [batch_size, 128] 的 GNN 分支特征；数据缺失时返回零向量（与父类同构）。
        """
        tg_x = getattr(data, "tg_x", None)
        tg_edge_index = getattr(data, "tg_edge_index", None)
        tg_frame = getattr(data, "tg_frame", None)
        if tg_x is None or tg_edge_index is None or tg_frame is None or tg_x.numel() == 0:
            # 🔴 部署侧（planner）必须把这个兜底变成硬失败：`tg_x` 缺失时本分支恒零，
            #    闭环会 rc=0、有产物、分数低，**与真实结果长得一模一样**（第十条）。
            #    训练/审计侧保持原行为（strict_temporal 默认 False），逐位不变。
            if getattr(self, "strict_temporal", False):
                raise RuntimeError(
                    "[R1] strict_temporal=True 但样本没有可用的 tg_x/tg_edge_index/tg_frame"
                    f"（tg_x={'None' if tg_x is None else f'numel={tg_x.numel()}'}）⇒ "
                    "GNN 分支会恒零。调用方必须先挂逐帧图"
                    "（`src.shared.utils.temporal_graph.AttachTemporalGraph`）。"
                )
            return torch.zeros(batch_size, 128, device=device)

        # 样本索引：DataLoader(follow_batch=['tg_x']) 产出 tg_x_batch
        tg_sample = getattr(data, "tg_x_batch", None)
        if tg_sample is None:
            # 单样本推理路径（planner batch=1）
            tg_sample = torch.zeros(tg_x.size(0), dtype=torch.long, device=tg_x.device)

        node_feat = self._extract_core_features(tg_x)
        # 一次前向算完全部 (样本 × 帧) 的图；边已按帧、按样本偏移过，不跨帧不跨样本
        node_embed = self.gnn_encoder.conv1(node_feat, tg_edge_index)
        node_embed = torch.relu(node_embed)
        node_embed = self.gnn_encoder.dropout(node_embed)
        node_embed = self.gnn_encoder.conv2(node_embed, tg_edge_index)

        # (样本, 帧) 双重分组 -> 唯一组 id；每个样本恒有 num_frames 帧 ⇒ 组数 = B*T
        group = tg_sample * self.num_frames + tg_frame
        frame_embed = global_mean_pool(node_embed, group)  # [B*T, 128]

        n_groups = batch_size * self.num_frames
        if frame_embed.size(0) != n_groups:
            logger.warning(
                "[R1] 分组数 %d ≠ 期望 %d，回退零向量（勿静默当成「干预了但没效果」）",
                frame_embed.size(0), n_groups,
            )
            return torch.zeros(batch_size, 128, device=device)

        seq = frame_embed.view(batch_size, self.num_frames, 128)

        frame_ids = torch.arange(self.num_frames, device=seq.device)
        seq = seq + self.frame_pos_embedding(frame_ids).unsqueeze(0)

        seq = self.temporal_transformer(seq)  # [B, T, 128]
        return seq[:, -1, :]  # 末帧读出（不再引入 mean pool，那是 R3 的事）

    def forward(self, data) -> torch.Tensor:
        """与父类同签名、同输出（[batch, 120] 的 60×2 增量）。"""
        if hasattr(data, "batch") and data.batch is not None:
            batch_size = int(data.batch.max().item()) + 1
        elif hasattr(data, "y") and data.y is not None:
            batch_size = data.y.shape[0] // 60 if data.y.dim() == 2 else data.y.shape[0]
        else:
            batch_size = 1

        device = data.x.device if hasattr(data, "x") else torch.device("cpu")

        ego_feat = torch.zeros(batch_size, 128, device=device)
        context_feat = torch.zeros(batch_size, 128, device=device)

        # ── 分支 1/2：逐字复用父类逻辑（R1 不动它们）──────────────────
        if hasattr(data, "ego_history") and data.ego_history.numel() > 0:
            ego_history_seq = self._reshape_ego_history(data, batch_size, device)
            if ego_history_seq is not None:
                ego_encoded = self.ego_history_encoder(ego_history_seq.view(-1, 2))
                ego_encoded = ego_encoded.view(batch_size, 20, 128)
                ego_feat = self.ego_transformer(ego_encoded).mean(dim=1)

        if hasattr(data, "x") and data.x.size(1) >= 9:
            x_features = self._extract_core_features(data.x)
            type_mask = self._create_feature_mask(data.x)
            x_embed = self.embed_norm(self.input_proj(x_features) + self.type_embedding(type_mask))

            if hasattr(data, "batch") and data.batch is not None:
                outs = []
                for i in range(batch_size):
                    mask = data.batch == i
                    if mask.sum() > 0:
                        outs.append(self.context_transformer(x_embed[mask].unsqueeze(0)).mean(dim=1))
                if outs:
                    context_feat = torch.cat(outs, dim=0)
            else:
                ctx = self.context_transformer(x_embed.unsqueeze(0)).mean(dim=1)
                context_feat = ctx if ctx.size(0) == batch_size else ctx.repeat(batch_size, 1)

        # ── 分支 3：R1 的改动点 ─────────────────────────────────────
        gnn_feat = self._encode_temporal_gnn(data, batch_size, device)

        fused = self.feature_fusion(torch.cat([ego_feat, context_feat, gnn_feat], dim=1))
        return self.output_head(fused)

    def _reshape_ego_history(self, data, batch_size: int, device) -> Optional[torch.Tensor]:
        """把 ego_history 归一成 [batch, 20, 2]；无法归一时返回 None。"""
        eh = data.ego_history
        if eh.dim() == 3:
            return eh
        if eh.dim() == 2:
            if eh.shape[0] == batch_size * 20:
                return eh.view(batch_size, 20, 2)
            if eh.shape[0] == 20:
                return eh.unsqueeze(0).repeat(batch_size, 1, 1)
        if eh.numel() == batch_size * 20 * 2:
            return eh.view(batch_size, 20, 2)
        return None
