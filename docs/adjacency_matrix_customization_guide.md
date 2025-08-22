# GNN_Trans 邻接矩阵自定义操作指南

## 📋 目录
1. [概述](#概述)
2. [系统架构](#系统架构)
3. [快速开始](#快速开始)
4. [配置详解](#配置详解)
5. [高级自定义](#高级自定义)
6. [实验指南](#实验指南)
7. [故障排除](#故障排除)
8. [API参考](#api参考)

---

## 概述

GNN_Trans模型的邻接矩阵自定义系统允许您灵活控制图神经网络中不同交通元素之间的连接关系。通过修改连接规则，您可以：

- 禁止ego车辆与道路元素连接
- 排除特定干扰车辆
- 控制连接距离阈值
- 实现以ego为中心的连接模式
- 进行消融实验研究

## 系统架构

### 核心组件

```
GNN_Trans/
├── utils/
│   └── gnn_module.py           # 邻接矩阵构建器核心代码
├── adjacency_configs.py        # 预定义配置文件
├── utils/dataset.py            # 数据集加载（集成自定义邻接矩阵）
└── docs/                       # 技术文档
    └── adjacency_matrix_customization_guide.md
```

### 架构图

```
交通场景数据
    ↓
子图池化 (subgraph_pooling)
    ↓
邻接矩阵构建器 (AdjacencyMatrixBuilder)
    ↓
自定义连接规则应用
    ↓
GNN图结构
    ↓
GNN编码器
```

---

## 快速开始

### 步骤1: 选择预设配置

在 `utils/dataset.py` 文件第159行，修改配置：

```python
# 当前配置位置
custom_builder = get_adjacency_config("default")  # 🎯 修改这里
```

### 步骤2: 可用配置选项

| 配置名称 | 描述 | 适用场景 |
|---------|------|----------|
| `"default"` | 默认全连接模式 | 基准实验 |
| `"ego_only"` | 以ego为中心的星型连接 | 研究ego中心假设 |
| `"no_ego_road"` | 禁止ego连接道路元素 | 测试道路信息重要性 |
| `"nearby_only"` | 只连接15米内元素 | 减少远距离干扰 |
| `"selective"` | 选择性连接特定元素 | 排除干扰车辆 |

### 步骤3: 重新训练

```bash
# 删除旧的处理数据
rm -rf processed_data/*/processed/

# 重新训练模型
python train.py
```

---

## 配置详解

### 默认配置 (default)
```python
builder.set_connection_rule('ego_to_vehicle', True)      # ego → 车辆: ✓
builder.set_connection_rule('ego_to_lane', True)         # ego → 车道: ✓  
builder.set_connection_rule('ego_to_road_elements', True) # ego → 道路: ✓
builder.set_connection_rule('vehicle_to_vehicle', True)   # 车辆 ↔ 车辆: ✓
builder.set_connection_rule('distance_threshold', 30.0)   # 距离阈值: 30米
```

### ego为中心配置 (ego_only)
```python
# ego连接所有元素，但其他元素间不连接
builder.set_connection_rule('vehicle_to_vehicle', False)  # 车辆间: ✗
builder.set_connection_rule('vehicle_to_lane', False)     # 车辆-车道: ✗
builder.set_connection_rule('lane_to_lane', False)        # 车道间: ✗
```

### 禁止ego-道路连接 (no_ego_road)
```python
builder.set_connection_rule('ego_to_lane', False)          # ego → 车道: ✗
builder.set_connection_rule('ego_to_road_elements', False) # ego → 道路: ✗
builder.set_connection_rule('distance_threshold', 20.0)    # 距离阈值: 20米
```

### 近距离连接 (nearby_only)
```python
builder.set_connection_rule('distance_threshold', 15.0)    # 距离阈值: 15米
builder.set_connection_rule('vehicle_to_vehicle', False)   # 车辆间: ✗
builder.set_connection_rule('lane_to_lane', False)         # 车道间: ✗
```

---

## 高级自定义

### 创建自定义配置

在 `adjacency_configs.py` 中添加新函数：

```python
def get_my_custom_config():
    """我的自定义配置"""
    builder = AdjacencyMatrixBuilder()
    
    # 自定义规则
    builder.set_connection_rule('ego_to_vehicle', True)
    builder.set_connection_rule('ego_to_lane', False)
    builder.set_connection_rule('distance_threshold', 10.0)
    
    # 排除特定车辆
    builder.exclude_subgraph(5)  # 排除subgraph_id=5
    builder.exclude_subgraph(7)  # 排除subgraph_id=7
    
    return builder
```

### 动态排除干扰车辆

```python
def get_selective_config():
    builder = AdjacencyMatrixBuilder()
    
    # 方法1: 排除特定ID
    builder.exclude_subgraph(5)
    builder.exclude_subgraph(8)
    
    # 方法2: 只连接重要元素
    builder.set_only_connect_subgraphs([0, 1, 2, 3])
    
    return builder
```

### 基于距离和角度的连接

```python
def get_distance_angle_config():
    builder = AdjacencyMatrixBuilder()
    
    # 严格的距离和角度限制
    builder.set_connection_rule('distance_threshold', 12.0)  # 12米内
    builder.set_connection_rule('angle_threshold', 30.0)     # 30度内
    
    return builder
```

---

## 实验指南

### 实验1: 道路信息重要性

**目标**: 测试ego是否需要道路信息进行轨迹预测

**实验设置**:
```python
# 实验组: 禁止ego-道路连接
custom_builder = get_adjacency_config("no_ego_road")

# 对照组: 默认全连接
custom_builder = get_adjacency_config("default")
```

**评估指标**: 比较ADE和FDE的差异

### 实验2: 距离敏感性分析

**目标**: 确定最优的连接距离阈值

**实验设置**:
```python
# 测试不同距离阈值: 10m, 15m, 20m, 25m, 30m
for threshold in [10.0, 15.0, 20.0, 25.0, 30.0]:
    builder = AdjacencyMatrixBuilder()
    builder.set_connection_rule('distance_threshold', threshold)
```

### 实验3: 车辆交互重要性

**目标**: 测试车辆间直接交互 vs ego为中心的交互

**实验设置**:
```python
# 实验组: ego为中心
custom_builder = get_adjacency_config("ego_only")

# 对照组: 车辆间可交互
custom_builder = get_adjacency_config("default")
```

### 实验4: 噪声车辆过滤

**目标**: 识别并排除干扰车辆

**方法**:
1. 分析车辆对预测的影响
2. 排除影响较小的远距离车辆
3. 比较性能提升

---

## 故障排除

### 常见问题

#### 1. 导入错误
```
ImportError: cannot import name 'AdjacencyMatrixBuilder'
```

**解决方案**: 确保 `utils/gnn_module.py` 文件存在且完整

#### 2. 配置未生效
```
邻接矩阵未按预期连接
```

**解决方案**: 
1. 检查配置名称是否正确
2. 删除 `processed_data/*/processed/` 重新处理
3. 确认数据中包含 `SUBGRAPH_ID_TO_TYPE` 信息

#### 3. 内存错误
```
CUDA out of memory
```

**解决方案**: 使用距离阈值较小的配置减少连接数量

### 调试方法

#### 打印邻接矩阵信息
```python
# 在 dataset.py 中添加调试代码
print(f"图节点数: {len(gnn_graph['nodes'])}")
print(f"边数: {len(gnn_graph['edges'])}")
print(f"边类型: {gnn_graph['edge_types']}")
```

#### 可视化连接关系
```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(gnn_graph):
    G = nx.Graph()
    G.add_nodes_from(gnn_graph['nodes'])
    G.add_edges_from(gnn_graph['edges'])
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10)
    plt.title("GNN邻接矩阵可视化")
    plt.show()
```

---

## API参考

### AdjacencyMatrixBuilder类

#### 构造函数
```python
builder = AdjacencyMatrixBuilder()
```

#### 主要方法

##### set_connection_rule(rule_name, value)
设置连接规则

**参数**:
- `rule_name`: 规则名称 
  - `'ego_to_vehicle'`: ego连接车辆
  - `'ego_to_lane'`: ego连接车道
  - `'ego_to_road_elements'`: ego连接道路元素
  - `'vehicle_to_vehicle'`: 车辆间连接
  - `'vehicle_to_lane'`: 车辆连接车道
  - `'lane_to_lane'`: 车道间连接
  - `'distance_threshold'`: 距离阈值
  - `'angle_threshold'`: 角度阈值
- `value`: 规则值 (bool或float)

##### exclude_subgraph(subgraph_id)
排除特定子图

**参数**:
- `subgraph_id`: 要排除的子图ID

##### set_only_connect_subgraphs(subgraph_ids)
设置只连接特定子图

**参数**:
- `subgraph_ids`: 允许连接的子图ID列表

##### build_adjacency_matrix(subgraph_vectors, subgraph_types)
构建邻接矩阵

**参数**:
- `subgraph_vectors`: {subgraph_id: vector} 字典
- `subgraph_types`: {subgraph_id: type} 字典

**返回**:
- `dict`: {'nodes': [...], 'edges': [...], 'edge_types': [...]}`

### 元素类型定义

```python
ELEMENT_TYPES = {
    'EGO': 0,           # 自车
    'VEHICLE': 1,       # 其他车辆  
    'LANE': 2,          # 车道线
    'CROSSWALK': 3,     # 人行横道
    'TRAFFIC_LIGHT': 4, # 交通信号灯
    'STOP_SIGN': 5,     # 停止标志
    'ROAD_EDGE': 6      # 道路边界
}
```

### 边类型定义

```python
EDGE_TYPES = {
    0: 'ego-vehicle',    # ego与车辆连接
    1: 'ego-lane',       # ego与车道连接
    2: 'vehicle-vehicle', # 车辆间连接
    3: 'vehicle-lane',   # 车辆与车道连接
    4: 'lane-lane',      # 车道间连接
    5: 'other'           # 其他类型连接
}
```

---

## 总结

通过这个邻接矩阵自定义系统，您可以：

1. **控制连接关系**: 精确控制哪些元素可以连接
2. **排除干扰**: 移除对预测无益的元素
3. **距离控制**: 设置合适的连接范围
4. **消融实验**: 系统性研究不同组件的重要性
5. **性能优化**: 通过减少无效连接提升模型效率

这个系统为深入理解GNN在轨迹预测中的作用机制提供了强大的实验工具。

---

**文档版本**: v1.0  
**最后更新**: 2025年8月20日  
**作者**: GNN_Trans团队
