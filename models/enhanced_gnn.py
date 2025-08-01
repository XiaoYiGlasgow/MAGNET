import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

class MultiHeadGNNEncoder(nn.Module):
    """多头图神经网络编码器，结合多种图卷积方法"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # 多种图卷积层
        self.gcn_conv1 = GCNConv(in_dim, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, out_dim // 4)
        
        self.gat_conv1 = GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=0.1)
        self.gat_conv2 = GATConv(hidden_dim, out_dim // 4, heads=1, dropout=0.1)
        
        self.graph_conv1 = GraphConv(in_dim, hidden_dim)
        self.graph_conv2 = GraphConv(hidden_dim, out_dim // 4)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(out_dim // 4 * 3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 时空注意力机制
        self.spatial_attention = SpatialAttention(out_dim)
        self.temporal_attention = TemporalAttention(out_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(out_dim // 4)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GCN分支
        gcn_out = F.relu(self.gcn_conv1(x, edge_index))
        gcn_out = self.norm1(gcn_out)
        gcn_out = self.dropout(gcn_out)
        gcn_out = self.gcn_conv2(gcn_out, edge_index)
        gcn_out = self.norm2(gcn_out)
        
        # GAT分支 (图注意力网络)
        gat_out = F.relu(self.gat_conv1(x, edge_index))
        gat_out = self.dropout(gat_out)
        gat_out = self.gat_conv2(gat_out, edge_index)
        
        # GraphConv分支
        graph_out = F.relu(self.graph_conv1(x, edge_index))
        graph_out = self.norm1(graph_out)
        graph_out = self.dropout(graph_out)
        graph_out = self.graph_conv2(graph_out, edge_index)
        graph_out = self.norm2(graph_out)
        
        # 特征融合
        combined_features = torch.cat([gcn_out, gat_out, graph_out], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 应用时空注意力
        fused_features = self.spatial_attention(fused_features, edge_index)
        fused_features = self.temporal_attention(fused_features)
        
        # 全局池化
        if batch is not None:
            # 组合多种池化方法
            mean_pool = global_mean_pool(fused_features, batch)
            max_pool = global_max_pool(fused_features, batch)
            output = (mean_pool + max_pool) / 2
        else:
            output = fused_features.mean(dim=0, keepdim=True)

        return output

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index):
        # 计算节点间的注意力权重
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        attention_weights = self.attention_layer(edge_features)
        
        # 聚合邻居特征
        weighted_features = x.clone()
        for i in range(len(row)):
            weighted_features[col[i]] += attention_weights[i] * x[row[i]]
            
        return weighted_features

class TemporalAttention(nn.Module):
    """时间注意力机制"""
    def __init__(self, feature_dim):
        super().__init__()
        self.temporal_weight = nn.Parameter(torch.randn(feature_dim))
        self.temporal_bias = nn.Parameter(torch.zeros(feature_dim))
        
    def forward(self, x):
        # 简单的时间加权
        temporal_weights = torch.sigmoid(self.temporal_weight * x + self.temporal_bias)
        return x * temporal_weights

class HierarchicalGNNEncoder(nn.Module):
    """分层图神经网络编码器，处理不同层级的交通要素"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        
        # 不同类型要素的编码器
        self.agent_encoder = MultiHeadGNNEncoder(in_dim, hidden_dim, out_dim // 3)
        self.lane_encoder = MultiHeadGNNEncoder(in_dim, hidden_dim, out_dim // 3) 
        self.map_encoder = MultiHeadGNNEncoder(in_dim, hidden_dim, out_dim // 3)
        
        # 跨层级交互
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=8, batch_first=True
        )
        
        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, agent_data, lane_data, map_data, batch=None):
        # 分别编码不同类型的要素
        agent_feat = self.agent_encoder(
            agent_data.x, agent_data.edge_index, batch=batch
        )
        lane_feat = self.lane_encoder(
            lane_data.x, lane_data.edge_index, batch=batch
        )
        map_feat = self.map_encoder(
            map_data.x, map_data.edge_index, batch=batch
        )
        
        # 拼接所有特征
        all_features = torch.cat([agent_feat, lane_feat, map_feat], dim=1)
        
        # 跨层级注意力交互
        all_features = all_features.unsqueeze(0)  # [1, batch_size, feature_dim]
        attended_features, _ = self.cross_level_attention(
            all_features, all_features, all_features
        )
        attended_features = attended_features.squeeze(0)
        
        # 最终融合
        output = self.final_fusion(attended_features)
        
        return output
