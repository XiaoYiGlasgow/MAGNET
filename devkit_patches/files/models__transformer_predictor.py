import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

# 使用更兼容的config导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GNN_Trans")))

import config


class GNNEncoder(nn.Module):
    """增强的GNN编码器，具备更强的边界检查和错误处理"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        # 增强的边界检查和错误处理
        if x.size(0) == 0 or edge_index.numel() == 0:
            return torch.zeros(1, self.conv2.out_channels, device=x.device)
            
        # 静默处理边索引验证，防止CUDA设备端断言错误
        if edge_index.numel() > 0:
            max_node_idx = x.size(0) - 1
            if edge_index.max() > max_node_idx:
                # 静默过滤无效边，不输出警告
                valid_edges = (edge_index[0] <= max_node_idx) & (edge_index[1] <= max_node_idx)
                if valid_edges.sum() == 0:
                    if batch is not None:
                        return global_mean_pool(x, batch)
                    else:
                        return x.mean(dim=0, keepdim=True)
                edge_index = edge_index[:, valid_edges]
                
        # 如果没有有效边，返回零初始化输出
        if edge_index.numel() == 0:
            return torch.zeros(x.size(0), self.conv2.out_channels, device=x.device)
            
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return x


class TransformerPredictor(nn.Module):
    """整合版TransformerPredictor - 结合两个版本的优势"""

    def __init__(self):
        super(TransformerPredictor, self).__init__()

        # Ego历史编码器 - 使用更复杂的结构但保持检查点兼容性
        self.ego_history_encoder = nn.Sequential(
            nn.Linear(2, 64),           # 第一层：2 -> 64
            nn.BatchNorm1d(64, track_running_stats=False),  # BatchNorm for better training
            nn.ReLU(),
            nn.Linear(64, 128)          # 第二层：64 -> 128
        )

        # 输入投影和归一化层
        self.input_proj = nn.Linear(9, 128)
        self.feature_norm = nn.LayerNorm(9)
        self.type_embedding = nn.Embedding(2, 128)  # 二值掩码嵌入（0/1，非语义类型；决策B 删 typed-edge 后仅 ego-vs-rest 二值，论文 P4 对齐）
        self.embed_norm = nn.LayerNorm(128)

        # 双Transformer架构 - 分别处理ego和context
        ego_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.ego_transformer = nn.TransformerEncoder(ego_encoder_layer, num_layers=2)

        context_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.context_transformer = nn.TransformerEncoder(context_encoder_layer, num_layers=3)

        # GNN编码器用于空间关系建模
        self.gnn_encoder = GNNEncoder(
            in_dim=9,
            hidden_dim=64,
            out_dim=128
        )

        # 特征融合层
        self.feature_fusion = nn.Linear(384, 128)

        # 更复杂的输出头结构
        self.output_head = nn.Sequential(
            nn.Linear(128, 128),        # 第一层扩展
            nn.ReLU(),
            nn.LayerNorm(128),          # 中间归一化
            nn.Dropout(0.1),            # 防过拟合
            nn.Linear(128, 128),        # 第二层处理
            nn.ReLU(),
            nn.Linear(128, 120)         # 最终输出层
        )
        
        # 关键修复：初始化输出层权重，确保合理的预测幅度
        with torch.no_grad():
            # 输出层使用更大的初始化标准差
            nn.init.normal_(self.output_head[-1].weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.output_head[-1].bias)
            
            # 其他线性层也调整初始化
            for module in self.output_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        print(f"[整合模型] 架构已更新匹配检查点")
        print(f"[整合模型] Ego历史: 2->64->128 (with BatchNorm)")
        print(f"[整合模型] 双Transformer: ego(2层) + context(3层)")
        print(f"[整合模型] GNN: 9->64->128")
        print(f"[整合模型] 特征融合: 384->128")
        print(f"[整合模型] 输出头: 128->128->128->120 (增强版)")

    def _create_feature_mask(self, features: torch.Tensor) -> torch.Tensor:
        """创建特征掩码，标识哪些是agent轨迹（有时间信息）哪些是道路（0填充）"""
        time_nonzero = features[:, 6] != 0
        speed_nonzero = (features[:, 7] != 0) | (features[:, 8] != 0)
        is_agent = time_nonzero | speed_nonzero
        return is_agent.long()

    def _extract_core_features(self, features: torch.Tensor) -> torch.Tensor:
        """提取核心的9维特征，并进行标准化"""
        core_features = features[:, :9]
        return self.feature_norm(core_features)

    def forward(self, data) -> torch.Tensor:
        """增强的前向传播 - 结合两个版本的优点"""
        
        # 更鲁棒的批次大小计算
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
        elif hasattr(data, 'y') and data.y is not None:
            if data.y.dim() == 2:
                batch_size = data.y.shape[0] // 60
            elif data.y.dim() == 3:
                batch_size = data.y.shape[0]
            else:
                batch_size = 1
        else:
            batch_size = 1

        device = data.x.device if hasattr(data, 'x') else torch.device('cpu')

        # 初始化特征容器
        ego_feat = torch.zeros(batch_size, 128, device=device)
        context_feat = torch.zeros(batch_size, 128, device=device)
        gnn_feat = torch.zeros(batch_size, 128, device=device)

        # 1. 增强的Ego历史特征处理
        if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
            try:
                # 处理不同的batch格式 - 更复杂的维度处理
                if data.ego_history.dim() == 2:
                    # 批量数据被拼接成 [batch_size*20, 2] 格式
                    if data.ego_history.shape[0] == batch_size * 20:
                        ego_history_seq = data.ego_history.view(batch_size, 20, 2)
                    elif data.ego_history.shape[0] == 20:
                        # 单个样本，扩展到batch
                        ego_history_seq = data.ego_history.unsqueeze(0).repeat(batch_size, 1, 1)
                    elif data.ego_history.shape[0] == batch_size:
                        # 简化的2D输入
                        ego_input = data.ego_history[:, :2] if data.ego_history.size(1) >= 2 else data.ego_history
                        ego_feat = self.ego_history_encoder(ego_input)
                        ego_history_seq = None
                    else:
                        # 意外格式，使用零填充
                        ego_history_seq = torch.zeros(batch_size, 20, 2, device=device)
                elif data.ego_history.dim() == 3:
                    # 已经是正确格式 [batch_size, 20, 2]
                    ego_history_seq = data.ego_history
                else:
                    # 其他格式，尝试重塑
                    total_elements = data.ego_history.numel()
                    if total_elements == batch_size * 20 * 2:
                        ego_history_seq = data.ego_history.view(batch_size, 20, 2)
                    else:
                        ego_history_seq = torch.zeros(batch_size, 20, 2, device=device)

                # 如果有时序数据，使用Transformer处理
                if ego_history_seq is not None and ego_history_seq.shape[1] == 20:
                    # 编码ego历史轨迹序列
                    ego_encoded = self.ego_history_encoder(ego_history_seq.view(-1, 2))  # [batch*20, 2] -> [batch*20, 128]
                    ego_encoded = ego_encoded.view(batch_size, 20, 128)  # [batch, 20, 128]
                    ego_feat_seq = self.ego_transformer(ego_encoded)  # [batch_size, 20, 128]
                    ego_feat = ego_feat_seq.mean(dim=1)  # [batch_size, 128] - 时间维度池化

            except Exception:
                # 决策D/E0：EGO_HISTORY 已修为 [20,2]，ego 分支不应再失败；暴露而非静默零向量死分支
                raise

        # 2. 增强的Context特征处理
        if hasattr(data, 'x') and data.x.size(1) >= 9:
            try:
                # 使用更复杂的特征提取和掩码创建
                x_features = self._extract_core_features(data.x)
                type_mask = self._create_feature_mask(data.x)
                
                # 输入投影和类型嵌入
                x_embed = self.input_proj(x_features)
                type_embed = self.type_embedding(type_mask)
                x_embed = self.embed_norm(x_embed + type_embed)

                # 按batch分组处理context transformer
                # E0：需显式判 data.batch is not None（与上方 batch_size 计算一致）；
                # 单样本 PyG Data 仍有 batch 属性但为 None，None==i 得 bool 会令 mask.sum() 崩 → 落 else 单样本路径
                if hasattr(data, 'batch') and data.batch is not None:
                    context_outputs = []
                    for i in range(batch_size):
                        mask = data.batch == i
                        if mask.sum() > 0:
                            sample_x = x_embed[mask].unsqueeze(0)  # [1, num_nodes_i, 128]
                            sample_out = self.context_transformer(sample_x)  # [1, num_nodes_i, 128]
                            sample_pooled = sample_out.mean(dim=1)  # [1, 128]
                            context_outputs.append(sample_pooled)

                    if context_outputs:
                        context_feat = torch.cat(context_outputs, dim=0)  # [batch_size, 128]
                else:
                    # 单样本处理
                    x_embed = x_embed.unsqueeze(0)  # [1, num_nodes, 128]
                    context_feat = self.context_transformer(x_embed)  # [1, num_nodes, 128]
                    context_feat = context_feat.mean(dim=1)  # [1, 128]
                    if context_feat.size(0) != batch_size:
                        context_feat = context_feat.repeat(batch_size, 1)

            except Exception:
                raise  # E0：context 分支失败暴露，不再静默零向量

        # 3. 增强的GNN特征处理
        if hasattr(data, 'gnn_x') and hasattr(data, 'gnn_edge_index'):
            if data.gnn_x.numel() > 0 and data.gnn_edge_index.numel() > 0:
                try:
                    # 提取核心特征并进行归一化
                    gnn_core_features = self._extract_core_features(data.gnn_x)
                    
                    # 使用增强的GNN编码器
                    gnn_batch = getattr(data, 'gnn_batch', None)
                    gnn_out = self.gnn_encoder(gnn_core_features, data.gnn_edge_index, gnn_batch)

                    # 确保输出维度匹配batch_size
                    if gnn_out.dim() == 2:
                        if gnn_out.size(0) == batch_size:
                            gnn_feat = gnn_out
                        elif gnn_out.size(0) == 1:
                            gnn_feat = gnn_out.repeat(batch_size, 1)
                        else:
                            gnn_pooled = gnn_out.mean(dim=0, keepdim=True)
                            gnn_feat = gnn_pooled.repeat(batch_size, 1)
                    elif gnn_out.dim() == 1:
                        gnn_feat = gnn_out.unsqueeze(0).repeat(batch_size, 1)

                except Exception:
                    raise  # E0：gnn 分支失败暴露，不再静默零向量

        # 4. 特征融合 - 拼接所有特征
        combined_feat = torch.cat([ego_feat, context_feat, gnn_feat], dim=1)  # [batch_size, 384]
        fused_feat = self.feature_fusion(combined_feat)  # [batch_size, 128]

        # 5. 输出预测 - 使用增强的输出头
        output = self.output_head(fused_feat)

        return output

    def get_encoder_features(self, data) -> torch.Tensor:
        """
        提取encoder特征用于Diffusion生成

        返回拼接的encoder特征 [batch, 384] = ego_feat(128) + context_feat(128) + gnn_feat(128)

        Args:
            data: 输入数据对象

        Returns:
            [batch, 384] encoder特征向量
        """
        # 更鲁棒的批次大小计算
        if hasattr(data, 'batch') and data.batch is not None:
            batch_size = data.batch.max().item() + 1
        elif hasattr(data, 'y') and data.y is not None:
            if data.y.dim() == 2:
                batch_size = data.y.shape[0] // 60
            elif data.y.dim() == 3:
                batch_size = data.y.shape[0]
            else:
                batch_size = 1
        else:
            batch_size = 1

        device = data.x.device if hasattr(data, 'x') else torch.device('cpu')

        # 初始化特征容器
        ego_feat = torch.zeros(batch_size, 128, device=device)
        context_feat = torch.zeros(batch_size, 128, device=device)
        gnn_feat = torch.zeros(batch_size, 128, device=device)

        # 1. Ego历史特征处理（与forward相同）
        if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
            try:
                if data.ego_history.dim() == 2:
                    if data.ego_history.shape[0] == batch_size * 20:
                        ego_history_seq = data.ego_history.view(batch_size, 20, 2)
                    elif data.ego_history.shape[0] == 20:
                        ego_history_seq = data.ego_history.unsqueeze(0).repeat(batch_size, 1, 1)
                    elif data.ego_history.shape[0] == batch_size:
                        ego_input = data.ego_history[:, :2] if data.ego_history.size(1) >= 2 else data.ego_history
                        ego_feat = self.ego_history_encoder(ego_input)
                        ego_history_seq = None
                    else:
                        ego_history_seq = torch.zeros(batch_size, 20, 2, device=device)
                elif data.ego_history.dim() == 3:
                    ego_history_seq = data.ego_history
                else:
                    total_elements = data.ego_history.numel()
                    if total_elements == batch_size * 20 * 2:
                        ego_history_seq = data.ego_history.view(batch_size, 20, 2)
                    else:
                        ego_history_seq = torch.zeros(batch_size, 20, 2, device=device)

                if ego_history_seq is not None and ego_history_seq.shape[1] == 20:
                    ego_encoded = self.ego_history_encoder(ego_history_seq.view(-1, 2))
                    ego_encoded = ego_encoded.view(batch_size, 20, 128)
                    ego_feat_seq = self.ego_transformer(ego_encoded)
                    ego_feat = ego_feat_seq.mean(dim=1)
            except Exception:
                pass

        # 2. Context特征处理（与forward相同）
        if hasattr(data, 'x') and data.x.size(1) >= 9:
            try:
                x_features = self._extract_core_features(data.x)
                type_mask = self._create_feature_mask(data.x)

                x_embed = self.input_proj(x_features)
                type_embed = self.type_embedding(type_mask)
                x_embed = self.embed_norm(x_embed + type_embed)

                if hasattr(data, 'batch'):
                    context_outputs = []
                    for i in range(batch_size):
                        mask = data.batch == i
                        if mask.sum() > 0:
                            sample_x = x_embed[mask].unsqueeze(0)
                            sample_out = self.context_transformer(sample_x)
                            sample_pooled = sample_out.mean(dim=1)
                            context_outputs.append(sample_pooled)

                    if context_outputs:
                        context_feat = torch.cat(context_outputs, dim=0)
                else:
                    x_embed = x_embed.unsqueeze(0)
                    context_feat = self.context_transformer(x_embed)
                    context_feat = context_feat.mean(dim=1)
                    if context_feat.size(0) != batch_size:
                        context_feat = context_feat.repeat(batch_size, 1)
            except Exception:
                pass

        # 3. GNN特征处理（与forward相同）
        if hasattr(data, 'gnn_x') and hasattr(data, 'gnn_edge_index'):
            if data.gnn_x.numel() > 0 and data.gnn_edge_index.numel() > 0:
                try:
                    gnn_core_features = self._extract_core_features(data.gnn_x)
                    gnn_batch = getattr(data, 'gnn_batch', None)
                    gnn_out = self.gnn_encoder(gnn_core_features, data.gnn_edge_index, gnn_batch)

                    if gnn_out.dim() == 2:
                        if gnn_out.size(0) == batch_size:
                            gnn_feat = gnn_out
                        elif gnn_out.size(0) == 1:
                            gnn_feat = gnn_out.repeat(batch_size, 1)
                        else:
                            gnn_pooled = gnn_out.mean(dim=0, keepdim=True)
                            gnn_feat = gnn_pooled.repeat(batch_size, 1)
                    elif gnn_out.dim() == 1:
                        gnn_feat = gnn_out.unsqueeze(0).repeat(batch_size, 1)
                except Exception:
                    pass

        # 4. 拼接所有encoder特征 [batch, 384]
        encoder_features = torch.cat([ego_feat, context_feat, gnn_feat], dim=1)

        return encoder_features