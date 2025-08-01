import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import sys, os
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

# ✅ 添加项目根目录路径，确保 config.py 可导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import config  # ✅ 统一模块式导入 config


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch=None):
        # ✅ 使用图卷积网络处理图结构数据
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)

        # ✅ 如果有batch信息，使用全局池化；否则使用平均池化
        if batch is not None:
            x = global_mean_pool(x, batch)  # [batch_size, out_dim]
        else:
            x = x.mean(dim=0, keepdim=True)  # [1, out_dim]

        return x


class TransformerPredictor(nn.Module):
    def __init__(self):
        super(TransformerPredictor, self).__init__()

        # ✅ 简化输入处理：直接使用前9维特征
        self.input_proj = nn.Linear(9, config.EMBED_DIM)  # 使用前9维特征

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBED_DIM,
            nhead=config.NUM_HEADS,
            dim_feedforward=config.FF_DIM,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

        # ✅ GNN编码器用于处理图结构
        self.gnn_encoder = GNNEncoder(
            in_dim=12,  # 完整的12维特征
            hidden_dim=config.HIDDEN_DIM,
            out_dim=config.EMBED_DIM
        )

        # ✅ 特征融合层
        self.feature_fusion = nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.EMBED_DIM, config.OUT_CHANNELS)
        )

    def forward(self, data):
        """
        data: PyG Data对象，包含：
            - x: [num_nodes, 12] 节点特征（包含polyline特征）
            - edge_index: [2, num_edges] 边连接
            - batch: [num_nodes] 批次信息
            - gnn_x: [num_subgraphs, 12] GNN节点特征
            - gnn_edge_index: [2, num_gnn_edges] GNN边连接
        """
        batch_size = data.y.shape[0] if hasattr(data, 'y') else 1

        # ✅ 方法1：使用传统的polyline特征处理
        if hasattr(data, 'x') and data.x.size(1) >= 9:
            # 使用前9维特征进行Transformer编码
            x_features = data.x[:, :9]  # [num_nodes, 9]
            x_embed = self.input_proj(x_features)  # [num_nodes, embed_dim]

            # ✅ 按batch分组处理
            if hasattr(data, 'batch'):
                # 为每个样本分别处理
                transformer_outputs = []
                for i in range(batch_size):
                    mask = data.batch == i
                    if torch.is_tensor(mask) and mask.sum() > 0:
                        sample_x = x_embed[mask].unsqueeze(0)  # [1, num_nodes_i, embed_dim]
                        sample_out = self.transformer(sample_x)  # [1, num_nodes_i, embed_dim]
                        sample_pooled = sample_out.mean(dim=1)  # [1, embed_dim]
                        transformer_outputs.append(sample_pooled)

                if transformer_outputs:
                    transformer_feat = torch.cat(transformer_outputs, dim=0)  # [batch_size, embed_dim]
                else:
                    transformer_feat = torch.zeros(batch_size, config.EMBED_DIM, device=data.x.device)
            else:
                # 单样本处理
                x_embed = x_embed.unsqueeze(0)  # [1, num_nodes, embed_dim]
                transformer_feat = self.transformer(x_embed)  # [1, num_nodes, embed_dim]
                transformer_feat = transformer_feat.mean(dim=1)  # [1, embed_dim]
                transformer_feat = transformer_feat.repeat(batch_size, 1)  # [batch_size, embed_dim]
        else:
            transformer_feat = torch.zeros(batch_size, config.EMBED_DIM, device=data.x.device)

        # ✅ 方法2：使用GNN处理图结构特征
        gnn_feat = torch.zeros(batch_size, config.EMBED_DIM, device=data.x.device)
        if hasattr(data, 'gnn_x') and hasattr(data, 'gnn_edge_index'):
            if data.gnn_x.numel() > 0 and data.gnn_edge_index.numel() > 0:
                # 使用GNN编码器
                gnn_out = self.gnn_encoder(data.gnn_x, data.gnn_edge_index)  # [1, embed_dim]
                gnn_feat = gnn_out.repeat(batch_size, 1)  # [batch_size, embed_dim]

        # ✅ 特征融合
        if transformer_feat.shape[1] > 0 and gnn_feat.shape[1] > 0:
            combined_feat = torch.cat([transformer_feat, gnn_feat], dim=1)  # [batch_size, embed_dim*2]
            fused_feat = self.feature_fusion(combined_feat)  # [batch_size, embed_dim]
        else:
            fused_feat = transformer_feat + gnn_feat  # [batch_size, embed_dim]

        # ✅ 输出预测
        output = self.output_head(fused_feat)  # [batch_size, OUT_CHANNELS]

        return output