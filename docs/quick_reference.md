# GNN_Trans 邻接矩阵快速参考

## 🚀 5分钟快速上手

### 1️⃣ 修改连接配置
**文件**: `utils/dataset.py` 第159行
```python
custom_builder = get_adjacency_config("配置名")
```

### 2️⃣ 可用配置
| 配置 | 效果 | 用途 |
|-----|------|------|
| `default` | 全连接 | 基准对比 |
| `no_ego_road` | ego不连道路 | 测试道路信息重要性 |
| `ego_only` | ego中心星型 | 研究交互模式 |
| `nearby_only` | 15米内连接 | 减少噪声 |

### 3️⃣ 重新训练
```bash
rm -rf processed_data/*/processed/
python train.py
```

## 🔧 自定义示例

### 排除干扰车辆
```python
def my_config():
    builder = AdjacencyMatrixBuilder()
    builder.exclude_subgraph(5)  # 排除ID=5的车
    return builder
```

### 距离控制
```python
builder.set_connection_rule('distance_threshold', 10.0)  # 10米内
```

### 连接规则
```python
builder.set_connection_rule('ego_to_lane', False)      # 禁止ego连车道
builder.set_connection_rule('vehicle_to_vehicle', False) # 禁止车辆互连
```

## 📊 实验对比

**测试ego是否需要道路信息**:
- 对照组: `"default"`
- 实验组: `"no_ego_road"`
- 比较: ADE/FDE指标

**测试距离敏感性**:
- 测试: 10m, 15m, 20m, 25m, 30m
- 找到最优距离阈值

## 🐛 故障排除

**问题**: 配置未生效  
**解决**: 删除 `processed_data/*/processed/`

**问题**: 内存不足  
**解决**: 减小 `distance_threshold`

**问题**: 导入错误  
**解决**: 检查 `utils/gnn_module.py` 完整性
