# Diffusion Danger Model - 数据处理

## 目录结构

```
diffusion_danger_model/
├── config.py              # 配置文件
├── process_pittsburgh.py  # Pittsburgh数据处理脚本
├── processed_data/        # 输出数据目录
│   ├── features_*.pkl     # 处理后的特征文件
│   └── processing_report_*.md  # 自动生成的处理报告
└── utils/                 # 工具函数（如需要）
```

## 配置说明

| 参数 | 值 | 说明 |
|------|---|------|
| PAST_TIME_HORIZON | 4秒 | 历史时间窗口 |
| FUTURE_TIME_HORIZON | 6秒 | 未来时间窗口 |
| NUM_PAST_POSES | 40帧 | 历史轨迹帧数 |
| NUM_FUTURE_POSES | 60帧 | 未来轨迹帧数 |
| NUM_AGENTS | 30 | 最大周围车辆上限 |
| QUERY_RADIUS | 50m | 地图车道提取范围 |
| MIN_DISPLACEMENT | 10m | 最小位移（排除静止场景） |

## 使用方法

### 1. 处理少量数据（测试）

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/diffusion_danger_model
python process_pittsburgh.py --max 100
```

### 2. 处理全部数据

```bash
python process_pittsburgh.py
```

### 3. 自定义输出目录

```bash
python process_pittsburgh.py --output /path/to/output --max 10000
```

## 输出数据格式

每个 `features_*.pkl` 文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| POLYLINE_FEATURES | (N, 12) | 12维特征向量 |
| GROUND_TRUTH | (1, 60, 2) | 60帧未来轨迹增量 |
| TRAJ_ID_TO_INDICES | dict | 轨迹ID到索引映射 |
| EGO_LAST_POS | [x, y] | Ego绝对位置 |

## 自动生成的处理报告

处理完成后会自动生成 `processing_report_*.md` 报告文件，包含：

- **处理配置**：时间窗口、车辆参数、筛选配置
- **处理结果统计**：成功/失败数量、处理速度、用时
- **周围车辆数量分布**：车辆数统计和占比
- **数据格式验证**：字段形状和特征说明
- **说明**：ID含义、注意事项

### 查看报告

```bash
# 查看最新生成的报告
cat processed_data/processing_report_*.md | head -100
```

## 车辆数量说明

### ID含义

| ID | 类型 | 是否计入统计 |
|----|------|------------|
| ID=0 | Ego（本车） | ❌ 不计入 |
| ID>0 + type='agent' | 周围车辆 | ✓ 计入 |
| ID>0 + type='lane' | 车道 | ❌ 不计入 |

### 注意事项

1. **车辆数量取决于NuPlan数据本身**，不是危险筛选后的结果
2. **0辆车场景**：NuPlan数据中该时刻确实无周围车辆
3. **后续处理**：需要使用 `ttc_filter.py` 进行TTC危险场景筛选

## 后续处理

处理完成后，使用 `ttc_filter.py` 进行TTC筛选：
- 从 processed_data 提取危险场景
- 生成 danger_pair_scenarios 用于Diffusion训练