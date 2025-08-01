import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

class PositionalEncoding(nn.Module):
    """增强的位置编码，支持2D空间位置"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 标准时序位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 2D空间位置编码投影层
        self.spatial_proj = nn.Linear(2, d_model // 4)
        
    def forward(self, x, spatial_pos=None):
        # x: [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
        if x.dim() == 3 and x.size(1) != self.d_model:  # batch_first
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].unsqueeze(0)
        else:
            seq_len = x.size(0)
            x = x + self.pe[:seq_len].unsqueeze(1)
            
        # 添加空间位置编码
        if spatial_pos is not None:
            spatial_encoding = self.spatial_proj(spatial_pos)
            # 将空间编码重复到完整的特征维度
            spatial_encoding = spatial_encoding.repeat(1, 1, 4)  # 假设d_model能被4整除
            x = x + spatial_encoding
            
        return x

class MultiScaleAttention(nn.Module):
    """多尺度注意力机制，捕获不同时间范围的依赖关系"""
    def __init__(self, d_model, num_heads, scales=[1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        
        # 不同尺度的卷积层
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=scale, 
                     padding=scale//2, groups=d_model//num_heads)
            for scale in scales
        ])
        
        # 尺度融合
        self.scale_fusion = nn.Linear(d_model * len(scales), d_model)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # 多尺度特征提取
        x_conv = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        scale_features = []
        
        for conv in self.scale_convs:
            scale_feat = conv(x_conv)  # [batch_size, d_model, seq_len]
            scale_features.append(scale_feat.transpose(1, 2))  # [batch_size, seq_len, d_model]
        
        # 拼接多尺度特征
        multi_scale_x = torch.cat(scale_features, dim=-1)  # [batch_size, seq_len, d_model*scales]
        multi_scale_x = self.scale_fusion(multi_scale_x)  # [batch_size, seq_len, d_model]
        
        # 残差连接
        multi_scale_x = self.norm1(x + multi_scale_x)
        
        # 多头注意力
        attn_out, attn_weights = self.attention(
            multi_scale_x, multi_scale_x, multi_scale_x, 
            attn_mask=mask
        )
        
        # 残差连接和层归一化
        x = self.norm1(multi_scale_x + attn_out)
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, attn_weights

class TrajectoryTransformer(nn.Module):
    """专门用于轨迹预测的Transformer架构"""
    def __init__(self, input_dim=9, d_model=128, num_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 多尺度注意力层
        self.attention_layers = nn.ModuleList([
            MultiScaleAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        # 时间感知池化
        self.temporal_pooling = TemporalPooling(d_model)
        
        # 轨迹特征提取
        self.trajectory_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x, spatial_positions=None, attention_mask=None):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoding(x, spatial_positions)
        
        # 多层多尺度注意力
        attention_weights = []
        for attention_layer in self.attention_layers:
            x, attn_w = attention_layer(x, attention_mask)
            attention_weights.append(attn_w)
        
        # 时间感知池化
        pooled_features = self.temporal_pooling(x)  # [batch_size, d_model]
        
        # 轨迹特征提取
        trajectory_features = self.trajectory_head(pooled_features)
        
        return trajectory_features, attention_weights

class TemporalPooling(nn.Module):
    """时间感知池化层，根据时间重要性加权池化"""
    def __init__(self, d_model):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 计算每个时间步的重要性权重
        weights = self.attention_weights(x)  # [batch_size, seq_len, 1]
        
        # 加权平均池化
        weighted_features = x * weights
        pooled = weighted_features.sum(dim=1)  # [batch_size, d_model]
        
        return pooled

class AdaptiveTransformerEncoder(nn.Module):
    """自适应Transformer编码器，可以根据输入调整注意力模式"""
    def __init__(self, d_model=128, num_heads=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # 标准Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # 自适应注意力控制
        self.attention_controller = nn.Sequential(
            nn.Linear(d_model, num_heads),
            nn.Sigmoid()
        )
        
        # 层间连接权重
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(self, x, src_mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        layer_outputs = []
        current_x = x
        
        for i, layer in enumerate(self.transformer_layers):
            # 计算自适应注意力权重
            attention_control = self.attention_controller(current_x.mean(dim=1))  # [batch_size, num_heads]
            
            # 通过Transformer层
            layer_out = layer(current_x, src_mask=src_mask)
            layer_outputs.append(layer_out)
            current_x = layer_out
        
        # 加权融合所有层的输出
        stacked_outputs = torch.stack(layer_outputs, dim=0)  # [num_layers, batch_size, seq_len, d_model]
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        final_output = (stacked_outputs * weights).sum(dim=0)
        
        return final_output
