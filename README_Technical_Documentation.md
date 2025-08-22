# GNN_Trans: 基于图神经网络和Transformer的轨迹预测系统

## 技术文档

**版本**: 1.0  
**日期**: 2025年8月22日  
**作者**: Xiao Yi  

---

## 1. 系统概述

### 1.1 项目简介
GNN_Trans是一个基于NuPlan数据集的自动驾驶轨迹预测系统，采用图神经网络(GNN)和Transformer的双分支融合架构，用于预测自动驾驶车辆在复杂交通场景中的未来轨迹。

### 1.2 核心特性
- **双模态特征处理**: 结合序列建模(Transformer)和图结构建模(GNN)
- **可配置邻接矩阵**: 支持5种不同的空间连接模式
- **端到端训练**: 从原始场景数据到轨迹预测的完整流程
- **NuPlan集成**: 完全兼容NuPlan数据格式和评估框架

---

## 2. 系统架构

### 2.1 整体架构
```
原始场景数据 → 数据预处理 → 双分支网络 → 特征融合 → 轨迹预测
                          ↓
                    ┌─────────────┐
                    │ Transformer │ (序列特征)
                    │   分支      │      ↓
                    └─────────────┘  特征融合
                    ┌─────────────┐      ↓
                    │    GNN      │ (图特征)
                    │   分支      │      ↓
                    └─────────────┘  轨迹输出
```

### 2.2 核心组件

#### A. 数据预处理模块 (`utils/dataset.py`)
- **功能**: 处理NuPlan原始场景数据
- **输入**: NuPlan场景文件(.db格式)
- **输出**: PyTorch Geometric图数据结构
- **特征提取**:
  - Ego车辆历史轨迹 (4秒, 40个时间步)
  - 周围车辆轨迹 (最多15个智能体)
  - 地图元素 (车道线、人行横道、路线车道)

#### B. Transformer分支
```python
输入: polyline特征 [num_nodes, 9]
     ↓
线性投影: 9 → 128维
     ↓
多头自注意力: 8个注意力头, 3层
     ↓
全局池化: [batch_size, 128]
```

#### C. GNN分支  
```python
输入: 子图特征 [num_subgraphs, 12]
     ↓
GCN层1: 12 → 64维
     ↓
GCN层2: 64 → 128维
     ↓
全局池化: [batch_size, 128]
```

#### D. 特征融合模块
```python
Transformer特征 [128] + GNN特征 [128]
     ↓
拼接: [256]
     ↓
线性层: 256 → 128
     ↓
输出预测头: 128 → 60 (30步×2坐标)
```

---

## 3. 配置参数

### 3.1 数据配置
```python
# 时间配置
PAST_TIME_HORIZON = 4秒     # 历史轨迹长度
FUTURE_TIME_HORIZON = 3秒   # 预测轨迹长度
DISCRETE_SIZE = 10Hz        # 采样频率

# 空间配置  
NUM_AGENTS = 15            # 最大智能体数量
LANE_NUM = 30             # 车道线数量
QUERY_RADIUS = 30米        # 特征提取半径
```

### 3.2 模型配置
```python
# Transformer参数
EMBED_DIM = 128           # 嵌入维度
NUM_HEADS = 8             # 注意力头数
NUM_LAYERS = 3            # 编码器层数
FF_DIM = 256              # 前馈网络维度

# GNN参数
HIDDEN_DIM = 64           # GNN隐藏层维度

# 训练参数
BATCH_SIZE = 4            # 批次大小
LEARNING_RATE = 1e-3      # 学习率
EPOCHS = 100              # 训练轮数
```

---

## 4. 邻接矩阵配置系统

### 4.1 可用连接模式

| 模式 | 描述 | 特点 |
|------|------|------|
| `default` | 全连接模式 | 30米内所有元素相互连接 |
| `ego_only` | ego为中心 | 只有ego连接其他元素 |
| `no_ego_road` | ego不连道路 | ego不连接车道线和道路元素 |
| `nearby_only` | 近距离连接 | 15米内连接，减少长距离连接 |
| `selective` | 选择性连接 | 可排除特定干扰车辆 |

### 4.2 配置方法
```python
# 在adjacency_configs.py中选择模式
from adjacency_configs import get_adjacency_config

# 获取特定配置
builder = get_adjacency_config("ego_only")

# 自定义连接规则
builder.set_connection_rule('ego_to_vehicle', True)
builder.set_connection_rule('vehicle_to_vehicle', False)
builder.set_connection_rule('distance_threshold', 25.0)
```

---

## 5. 使用指南

### 5.1 快速开始

#### 数据准备
```bash
# 设置数据路径
export DATA_PATH="/path/to/nuplan/dataset"
export MAP_PATH="/path/to/nuplan/maps"
```

#### 运行快速Demo
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python quick_demo.py

# 选择连接模式
# 1. default - 默认全连接模式
# 2. ego_only - ego为中心的星型连接  
# 3. no_ego_road - 禁止ego连接道路元素
# 4. nearby_only - 只连接15米内的元素
```

#### 完整训练
```bash
python train.py
```

### 5.2 nuBoard可视化
```bash
# 启动nuBoard进行可视化评估
python demo_nuboard.py

# 选择模式:
# 1. 运行新的仿真并启动nuboard
# 2. 直接启动nuboard (使用现有结果)
```

---

## 6. 输入/输出格式

### 6.1 输入数据结构
```python
# PyTorch Geometric Data对象
data = {
    'x': torch.tensor([num_nodes, 12]),          # polyline节点特征
    'edge_index': torch.tensor([2, num_edges]),  # polyline内部连接
    'gnn_x': torch.tensor([num_subgraphs, 12]),  # 子图特征
    'gnn_edge_index': torch.tensor([2, num_gnn_edges]), # 子图间连接
    'batch': torch.tensor([num_nodes]),          # 批次信息
    'y': torch.tensor([batch_size, 60])          # 目标轨迹
}
```

### 6.2 特征维度说明
```python
# Polyline特征 (12维)
[x, y, heading, velocity_x, velocity_y, acceleration_x, acceleration_y, 
 yaw_rate, length, width, height, type_encoding]

# Transformer使用前9维：
[x, y, heading, velocity_x, velocity_y, acceleration_x, acceleration_y, 
 yaw_rate, length]

# GNN使用完整12维特征
```

### 6.3 输出格式
```python
# 输出维度: [batch_size, 60]
# 60 = 30个时间步 × 2个坐标(x, y)
# 预测时长: 3秒
# 时间间隔: 0.1秒
# 坐标格式: 相对位移增量
```

---

## 7. 评估指标

### 7.1 轨迹预测指标
- **ADE (Average Displacement Error)**: 平均位移误差
- **FDE (Final Displacement Error)**: 最终位移误差

### 7.2 nuPlan评估指标
- **ego_progress_along_expert_route**: 路径跟踪精度
- **ego_is_collision_free**: 碰撞检测
- **ego_expert_l2_error_statistics**: L2误差统计
- **ego_jerk_statistics**: 急动度统计
- **ego_acceleration_statistics**: 加速度统计

---

## 8. 文件结构

```
GNN_Trans/
├── config.py                    # 全局配置文件
├── adjacency_configs.py         # 邻接矩阵配置
├── quick_demo.py               # 快速Demo脚本
├── demo_nuboard.py             # nuBoard演示脚本
├── train.py                    # 完整训练脚本
├── models/
│   └── transformer_predictor.py # 核心模型
├── utils/
│   ├── dataset.py              # 数据处理
│   ├── data_utils.py          # 数据工具函数
│   └── gnn_module.py          # GNN模块
├── planner/
│   └── gnn_transformer_planner.py # nuPlan集成
└── Nuboard/
    └── gnn_trans_planner.py    # nuBoard可视化
```

---

## 9. 技术特点与创新

### 9.1 双分支架构优势
- **Transformer分支**: 擅长建模时序依赖和长程关系
- **GNN分支**: 擅长建模空间拓扑结构和局部交互
- **特征融合**: 结合两种建模方式的优势

### 9.2 可配置邻接矩阵
- 支持多种空间连接策略
- 便于消融实验和性能优化
- 适应不同交通场景的特点

### 9.3 NuPlan完全兼容
- 支持标准NuPlan数据格式
- 集成nuBoard可视化系统
- 兼容NuPlan评估指标

---

## 10. 性能优化建议

### 10.1 训练优化
- 使用GPU加速训练
- 调整批次大小以平衡内存和性能
- 使用学习率衰减策略

### 10.2 邻接矩阵选择
- `default`: 信息最丰富，计算量大
- `ego_only`: 计算效率高，适合实时应用
- `nearby_only`: 平衡性能和效率

### 10.3 数据预处理
- 首次运行会处理全部场景数据(约7万个)
- 后续训练直接使用缓存数据
- 建议使用SSD存储加速数据加载

---

## 11. 故障排除

### 11.1 常见问题
1. **内存不足**: 减少BATCH_SIZE或NUM_AGENTS
2. **CUDA错误**: 检查GPU内存，调整模型参数
3. **导入错误**: 确保PYTHONPATH包含项目根目录

### 11.2 调试建议
- 使用`quick_demo.py`进行快速测试
- 检查数据预处理输出
- 验证邻接矩阵配置

---

## 12. 联系信息

**开发者**: Xiao Yi  
**项目路径**: `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans`  
**技术栈**: PyTorch, PyTorch Geometric, NuPlan, Transformer  

---

*此文档基于GNN_Trans v1.0版本，最后更新于2025年8月22日*
