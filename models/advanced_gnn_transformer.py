import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

from .enhanced_gnn import MultiHeadGNNEncoder, HierarchicalGNNEncoder
from .enhanced_transformer import TrajectoryTransformer, AdaptiveTransformerEncoder

class AdvancedGNNTransformer(nn.Module):
    """
    高级GNN-Transformer融合模型
    结合了多头图神经网络和自适应Transformer的轨迹预测模型
    """
    def __init__(self):
        super().__init__()
        
        # === 配置参数 ===
        self.input_dim = 9  # Transformer输入维度
        self.gnn_input_dim = 12  # GNN输入维度
        self.d_model = config.EMBED_DIM
        self.num_heads = config.NUM_HEADS
        self.num_layers = config.NUM_LAYERS
        
        # === GNN分支 ===
        # 多头GNN编码器
        self.multi_head_gnn = MultiHeadGNNEncoder(
            in_dim=self.gnn_input_dim,
            hidden_dim=config.HIDDEN_DIM,
            out_dim=self.d_model,
            num_heads=4
        )
        
        # 分层GNN编码器（用于处理不同类型的交通要素）
        self.hierarchical_gnn = HierarchicalGNNEncoder(
            in_dim=self.gnn_input_dim,
            hidden_dim=config.HIDDEN_DIM,
            out_dim=self.d_model
        )
        
        # === Transformer分支 ===
        # 轨迹专用Transformer
        self.trajectory_transformer = TrajectoryTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )
        
        # 自适应Transformer编码器
        self.adaptive_transformer = AdaptiveTransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers // 2
        )
        
        # === 跨模态融合模块 ===
        self.cross_modal_fusion = CrossModalFusion(self.d_model)
        
        # === 时空融合注意力 ===
        self.spatiotemporal_attention = SpatioTemporalAttention(self.d_model)
        
        # === 多任务输出头 ===
        self.trajectory_head = TrajectoryPredictionHead(
            input_dim=self.d_model,
            output_dim=config.OUT_CHANNELS,
            num_future_steps=config.NUM_FUTURE_POSES
        )
        
        # === 辅助任务头（可选） ===
        self.velocity_head = nn.Linear(self.d_model, config.NUM_FUTURE_POSES)
        self.heading_head = nn.Linear(self.d_model, config.NUM_FUTURE_POSES)
        
    def forward(self, data, return_attention=False):
        """
        前向传播
        Args:
            data: 包含图数据和序列数据的输入
            return_attention: 是否返回注意力权重
        """
        batch_size = data.y.shape[0] if hasattr(data, 'y') else 1
        device = data.x.device
        
        # === GNN特征提取 ===
        gnn_features = self._extract_gnn_features(data, batch_size, device)
        
        # === Transformer特征提取 ===
        transformer_features, attention_weights = self._extract_transformer_features(
            data, batch_size, device
        )
        
        # === 跨模态特征融合 ===
        fused_features = self.cross_modal_fusion(gnn_features, transformer_features)
        
        # === 时空注意力增强 ===
        enhanced_features = self.spatiotemporal_attention(fused_features)
        
        # === 轨迹预测 ===
        trajectory_prediction = self.trajectory_head(enhanced_features)
        
        # === 辅助预测（可选） ===
        velocity_prediction = self.velocity_head(enhanced_features)
        heading_prediction = self.heading_head(enhanced_features)
        
        # 组织输出
        output = {
            'trajectory': trajectory_prediction,
            'velocity': velocity_prediction,
            'heading': heading_prediction
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
            
        return trajectory_prediction  # 主要输出，保持与原接口兼容
    
    def _extract_gnn_features(self, data, batch_size, device):
        """提取GNN特征"""
        gnn_features = torch.zeros(batch_size, self.d_model, device=device)
        
        if hasattr(data, 'gnn_x') and hasattr(data, 'gnn_edge_index'):
            if data.gnn_x.numel() > 0 and data.gnn_edge_index.numel() > 0:
                # 使用多头GNN编码器
                multi_head_feat = self.multi_head_gnn(
                    data.gnn_x, data.gnn_edge_index, 
                    batch=getattr(data, 'batch', None)
                )
                
                # 如果有分层数据，使用分层GNN
                if hasattr(data, 'agent_data') and hasattr(data, 'lane_data') and hasattr(data, 'map_data'):
                    hierarchical_feat = self.hierarchical_gnn(
                        data.agent_data, data.lane_data, data.map_data,
                        batch=getattr(data, 'batch', None)
                    )
                    # 融合多头和分层特征
                    gnn_features = (multi_head_feat + hierarchical_feat) / 2
                else:
                    gnn_features = multi_head_feat.repeat(batch_size, 1) if multi_head_feat.size(0) == 1 else multi_head_feat
        
        return gnn_features
    
    def _extract_transformer_features(self, data, batch_size, device):
        """提取Transformer特征"""
        transformer_features = torch.zeros(batch_size, self.d_model, device=device)
        attention_weights = None
        
        if hasattr(data, 'x') and data.x.size(1) >= self.input_dim:
            x_features = data.x[:, :self.input_dim]
            
            if hasattr(data, 'batch'):
                # 批处理模式
                transformer_outputs = []
                batch_attention_weights = []
                
                for i in range(batch_size):
                    mask = data.batch == i
                    if torch.is_tensor(mask) and mask.sum() > 0:
                        sample_x = x_features[mask].unsqueeze(0)
                        
                        # 提取空间位置（如果可用）
                        spatial_pos = None
                        if data.x.size(1) >= 11:  # 假设位置信息在后面的维度
                            spatial_pos = data.x[mask, -2:].unsqueeze(0)
                        
                        # 轨迹Transformer
                        traj_feat, attn_w = self.trajectory_transformer(
                            sample_x, spatial_pos
                        )
                        
                        # 自适应Transformer进一步处理
                        adaptive_feat = self.adaptive_transformer(sample_x)
                        adaptive_pooled = adaptive_feat.mean(dim=1)
                        
                        # 融合两个Transformer的输出
                        combined_feat = (traj_feat + adaptive_pooled) / 2
                        
                        transformer_outputs.append(combined_feat)
                        batch_attention_weights.append(attn_w)
                
                if transformer_outputs:
                    transformer_features = torch.cat(transformer_outputs, dim=0)
                    attention_weights = batch_attention_weights
            else:
                # 单样本模式
                x_features = x_features.unsqueeze(0)
                spatial_pos = None
                if data.x.size(1) >= 11:
                    spatial_pos = data.x[:, -2:].unsqueeze(0)
                
                traj_feat, attn_w = self.trajectory_transformer(x_features, spatial_pos)
                adaptive_feat = self.adaptive_transformer(x_features)
                adaptive_pooled = adaptive_feat.mean(dim=1)
                
                transformer_features = ((traj_feat + adaptive_pooled) / 2).repeat(batch_size, 1)
                attention_weights = [attn_w]
        
        return transformer_features, attention_weights

class CrossModalFusion(nn.Module):
    """跨模态特征融合模块"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征对齐层
        self.gnn_align = nn.Linear(feature_dim, feature_dim)
        self.transformer_align = nn.Linear(feature_dim, feature_dim)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, gnn_features, transformer_features):
        # 特征对齐
        gnn_aligned = self.gnn_align(gnn_features)
        transformer_aligned = self.transformer_align(transformer_features)
        
        # 交叉注意力：让GNN特征关注Transformer特征
        gnn_enhanced, _ = self.cross_attention(
            gnn_aligned.unsqueeze(1),
            transformer_aligned.unsqueeze(1),
            transformer_aligned.unsqueeze(1)
        )
        gnn_enhanced = gnn_enhanced.squeeze(1)
        
        # 特征拼接和融合
        concatenated = torch.cat([gnn_enhanced, transformer_aligned], dim=1)
        fused_features = self.fusion_layer(concatenated)
        
        return fused_features

class SpatioTemporalAttention(nn.Module):
    """时空注意力模块"""
    def __init__(self, feature_dim):
        super().__init__()
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 时间注意力
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # 空间注意力权重
        spatial_weights = self.spatial_attention(features)
        
        # 时间注意力权重
        temporal_weights = self.temporal_attention(features)
        
        # 应用注意力
        enhanced_features = features * spatial_weights * temporal_weights
        
        return enhanced_features

class TrajectoryPredictionHead(nn.Module):
    """轨迹预测头，支持多种预测模式"""
    def __init__(self, input_dim, output_dim, num_future_steps):
        super().__init__()
        self.num_future_steps = num_future_steps
        
        # 递进式预测层
        self.progressive_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim)
        )
        
        # 不确定性估计（可选）
        self.uncertainty_head = nn.Linear(input_dim, output_dim)
        
    def forward(self, features):
        # 递进式特征提取
        x = features
        for layer in self.progressive_layers:
            x = layer(x) + x  # 残差连接
        
        # 主要轨迹预测
        trajectory = self.output_layer(x)
        
        # 不确定性估计
        uncertainty = torch.sigmoid(self.uncertainty_head(x))
        
        return trajectory
