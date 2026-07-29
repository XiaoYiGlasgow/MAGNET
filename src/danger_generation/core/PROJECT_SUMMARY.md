# Diffusion危险场景生成器 - 项目总结报告

**生成时间**: 2026-04-14

---

## 项目概述

本项目旨在为Diffusion危险场景生成器准备训练数据，通过重新处理NuPlan Pittsburgh数据集，使用新的时间窗口配置（4秒历史 + 6秒未来），并进行TTC（Time To Collision）危险场景筛选。

---

## 1. 数据处理阶段

### 1.1 处理配置

| 参数 | 值 | 说明 |
|------|---|------|
| 数据源 | Pittsburgh | NuPlan训练数据集 |
| 历史时间窗口 | 4秒 (40帧) | 与原配置(2秒)不同 |
| 未来时间窗口 | 6秒 (60帧) | 与原配置(8秒)不同 |
| 帧率 | 10Hz | 采样频率 |
| 最大周围车辆 | 30辆 | NUM_AGENTS上限 |
| 地图提取半径 | 50米 | QUERY_RADIUS |
| 最小位移筛选 | 10米 | 排除静止场景 |

### 1.2 处理结果

| 指标 | 结果 |
|------|------|
| 成功处理 | **803个场景** |
| 处理失败 | 0个 |
| 处理速度 | 258.1场景/分钟 |
| 总用时 | 3.1分钟 |

### 1.3 周围车辆数量分布

| 车辆数 | 场景数 | 占比 |
|--------|--------|------|
| 0辆 | 163 | 20.3% |
| 1辆 | 147 | 18.3% |
| 2辆 | 128 | 15.9% |
| 3辆 | 105 | 13.1% |
| 4-5辆 | 118 | 14.7% |
| 6-10辆 | 115 | 14.3% |
| 11-16辆 | 31 | 3.9% |

**统计汇总**:
- 平均周围车辆数: 3.06辆
- 最少: 0辆，最多: 16辆
- 有>=3辆车的场景: 365个 (45.5%)
- 有>=5辆车的场景: 196个 (24.4%)

### 1.4 输出数据格式

每个 `features_*.pkl` 文件包含:

| 字段 | 形状 | 说明 |
|------|------|------|
| POLYLINE_FEATURES | (N, 12) | 12维特征向量 |
| GROUND_TRUTH | (1, 60, 2) | 60帧未来轨迹增量 |
| TRAJ_ID_TO_INDICES | dict | 轨迹ID到索引映射 |
| EGO_LAST_POS | [x, y] | Ego绝对位置 |

**12维特征向量说明**:
| 维度 | 字段 | 说明 |
|------|------|------|
| 0-1 | x_start, y_start | segment起点坐标 |
| 2-3 | x_end, y_end | segment终点坐标 |
| 4 | heading_start | 起点朝向 |
| 5 | heading_end | 终点朝向 |
| 6 | normalized_timestamp | 归一化时间戳 |
| 7-8 | v_start, v_end | 起点/终点速度 |
| 9 | subgraph_ID | 子图ID |
| 10-11 | start_time, end_time | 相对时间 |

---

## 2. TTC危险场景筛选阶段

### 2.1 TTC筛选配置

| 参数 | 值 | 说明 |
|------|---|------|
| 纵向TTC阈值 | 5.0秒 | 超过此值无纵向碰撞风险 |
| 横向TTC阈值 | 2.0秒 | 超过此值无横向碰撞风险 |
| 考虑车辆尺寸 | False | 使用点模型计算 |
| 尺寸缓冲区 | 0.5米 | 安全边距 |

### 2.2 TTC筛选结果

| 指标 | 结果 |
|------|------|
| 输入场景 | 803个 |
| 有危险车辆的场景 | **564个** |
| 筛选成功率 | 70.2% |
| 原始车辆总数 | 2460辆 |
| 危险车辆(kept) | **1813辆** |
| 车辆保留率 | 73.7% |
| danger_pair保存数 | **564个** |

### 2.3 选择原因分布

| 选择原因 | 场景数 | 占比 |
|---------|--------|------|
| Front_agent_min_lon_ttc | 229 | 40.6% |
| Dual_risk_front_agent | 194 | 34.4% |
| Rear_agent_min_ttc | 141 | 25.0% |

**说明**:
- `Front_agent_min_lon_ttc`: 前方车辆纵向TTC最小的场景
- `Dual_risk_front_agent`: 前方车辆同时具有纵向和横向风险
- `Rear_agent_min_ttc`: 后方车辆TTC最小的场景

### 2.4 危险车辆数量分布

| 危险车辆数 | 场景数 | 占比 |
|-----------|--------|------|
| 1辆 | 157 | 27.8% |
| 2辆 | 146 | 25.9% |
| 3辆 | 95 | 16.8% |
| 4-5辆 | 83 | 14.7% |
| 6-10辆 | 64 | 11.3% |
| 11-15辆 | 19 | 3.4% |

### 2.5 danger_pair数据格式

每个 `danger_pair_features_*.pkl` 文件包含:

| 字段 | 形状 | 说明 |
|------|------|------|
| ego_history | (40, 5) | 4秒历史轨迹 |
| key_agent_history | (40, 5) | 关键危险agent历史轨迹 |
| ego_future_gt | (60, 5) | 6秒未来轨迹真值 |
| key_agent_future | (60, 3)或None | agent未来轨迹 |
| key_agent_lon_ttc | array | 纵向TTC序列 |
| key_agent_lat_ttc | array | 横向TTC序列 |
| boundary_agents | list | 其他危险agent列表 |
| time_window_config | dict | 时间窗口配置 |

---

## 3. 文件结构

```
diffusion_danger_model/
├── config.py                          # 配置文件
├── process_pittsburgh.py              # 批量处理脚本
├── processed_data/                    # 原始处理数据
│   ├── features_*.pkl                 # 803个场景文件
│   └── processing_report_*.md         # 处理报告
├── danger_pair_scenarios/             # TTC筛选后的危险场景
│   ├── danger_pair_features_*.pkl     # 564个危险场景文件
│   └── ttc_filter_report_*.md         # TTC筛选报告
└── ttc_filter_results/                # TTC筛选日志
    └── ttc_filter_batch_*.txt
```

---

## 4. 关键发现

### 4.1 数据特点

1. **时间窗口变化**: 从原来的2秒历史+8秒未来改为4秒历史+6秒未来，更适合短期危险场景预测
2. **车辆分布**: 约20%场景无周围车辆，45%场景有3辆以上车辆
3. **危险场景比例**: 70%的场景包含至少一个危险车辆（TTC低于阈值）

### 4.2 风险类型分布

- **纵向风险为主**: 40.6%的场景由前方车辆纵向TTC最小触发
- **双重风险**: 34.4%的场景前方车辆同时存在纵向和横向风险
- **后方风险**: 25%的场景由后方车辆触发

---

## 5. 后续使用建议

### 5.1 Diffusion模型训练

筛选后的564个danger_pair数据可直接用于训练:

```bash
# 训练命令示例
python train_diffusion_generator.py \
    --data_dir diffusion_danger_model/danger_pair_scenarios \
    --epochs 100 \
    --batch_size 32
```

### 5.2 数据增强

可通过以下方式扩展数据集:
- 增加TTC阈值范围（当前仅保留高风险场景）
- 处理其他NuPlan数据集（如Boston）
- 调整MIN_DISPLACEMENT获取更多低速场景

---

## 6. 代码修复记录

在项目执行过程中修复了以下问题:

| 问题 | 文件 | 解决方案 |
|------|------|----------|
| 函数定义顺序错误 | ttc_filter.py | 将`generate_ttc_report`移到`main()`之前 |
| f-string语法错误 | 多处 | 修复`print(f'{'='*60}')`格式问题 |
| 场景类型名称错误 | config.py | 修正`near_multiple_vehicle`为`near_multiple_vehicles` |

---

**报告生成完毕**

---

## 附录: 相关文件路径

- 数据处理脚本: [process_pittsburgh.py](diffusion_danger_model/process_pittsburgh.py)
- 配置文件: [config.py](diffusion_danger_model/config.py)
- TTC筛选脚本: [ttc_filter.py](GNN_Trans/utils/ttc_filter.py)
- 处理报告: [processing_report_20260414_162359.md](diffusion_danger_model/processed_data/processing_report_20260414_162359.md)
- TTC报告: [ttc_filter_report_20260414_162458.md](diffusion_danger_model/danger_pair_scenarios/ttc_filter_report_20260414_162458.md)