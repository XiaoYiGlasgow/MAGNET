# Diffusion Trajectory Visualization Module

快速可视化和评估 Diffusion 模型生成的轨迹。

## 中文完整流程文档

如果您想查看当前项目已经验证过的完整中文运行说明，包括：

- 推荐环境
- quick / bokeh / full / stats 的运行命令
- 输出目录说明
- `gnn_trans_encoder` checkpoint 的使用方式
- 常见报错排查

请优先阅读：

- [../../../docs/reference/visualization_workflow_guide.md](../../../docs/reference/visualization_workflow_guide.md)

## 模块结构

```
visualization/
├── __init__.py              # Factory & Registry
├── trajectory_converter.py  # 轨迹格式转换 & 运动学解码
├── standard_metrics.py      # 文献对齐标准指标（CR、TTC、Jerk JSD 等）
├── matplotlib_visualizer.py # Matplotlib 快速验证（自动调用 standard_metrics）
├── nuboard_visualizer.py    # Bokeh/NuBoard 详细展示
├── run_evaluation.py        # 统一运行入口
└── README.md                # 本文档
```

## 使用方法

### 1. Quick Validation (Matplotlib)

快速检查轨迹质量，生成静态图片。

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

# 评估 10 个样本
python visualization/run_evaluation.py \
    --mode quick \
    --model_path diffusion_training_experiments/full_training/best_model.pt \
    --num_samples 10

# 显示交互式图表
python visualization/run_evaluation.py --mode quick --show
```

**输出**:
- `sample_*.png` - 单样本轨迹对比图
- `summary.png` - 综合统计图
- `evaluation_report.json` - 详细数据

### 2. Bokeh Visualization

交互式 HTML 可视化，支持时间轴播放。

```bash
# 生成 Bokeh 可视化
python visualization/run_evaluation.py \
    --mode bokeh \
    --model_path diffusion_training_experiments/full_training/best_model.pt \
    --num_samples 5
```

**输出**:
- `diffusion_visualization.html` - 可在浏览器打开
- `scenario_*.pkl` - 场景数据文件
- `summary.json` - 统计摘要

### 3. Full Evaluation

同时运行两种可视化。

```bash
python visualization/run_evaluation.py \
    --mode full \
    --model_path diffusion_training_experiments/full_training/best_model.pt \
    --num_samples 10
```

### 4. Statistics Analysis

大规模统计分析。

```bash
python visualization/run_evaluation.py \
    --mode stats \
    --model_path diffusion_training_experiments/full_training/best_model.pt \
    --num_samples 100
```

## 可视化内容

### Matplotlib Quick Validation

每个样本生成 4 个子图:

1. **轨迹对比图** - GT vs Generated 轨迹位置
2. **速度曲线** - 速度随时间变化
3. **Danger 分数** - GT vs Generated danger 对比
4. **样本信息** - SVO angle, validity checks, metrics

综合统计图:
- Danger 分数散点图
- Danger 提升分布直方图
- 速度分布箱线图
- 汇总统计

### Bokeh Visualization

- **Scenario Tabs** - 每个场景一个标签页
- **时间轴 Hover** - 鼠标悬停显示时间点
- **Danger 对比柱状图** - 底部汇总图

## 核心指标

### 主要指标（项目自定义）

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| Margin Met Rate | Gen > GT + 0.5 的比例 | 成功数 / 总数 |
| Mean Gen Danger | 生成轨迹绝对危险程度均值 | danger_score 均值 |
| Valid Rate | 速度 ≤ 80 m/s 且无 NaN 的比例 | `check_kinematic_validity()` |
| Danger Improvement | Gen - GT 平均值 | 均值差 |

### 文献对齐标准指标（`standard_metrics.py`，自动计算）

| 指标 | 说明 | 参考文献 |
|------|------|---------|
| **Collision Rate** | ego 与 agent 发生碰撞的比例（中心距 < 2.0m） | AdvDiffuser, SafeBench |
| **Mean Min Distance** | ego-agent 最小距离均值（m） | KING, SaFeR |
| **Mean TTC** | 平均碰撞时间（s），无碰撞时为 inf | SaFeR, AdvSim |
| **Near Miss Rate** | 出现近距离接触（TTC < inf）的比例 | - |
| **Kinematic Feasibility** | 满足 accel ≤ 8 m/s²、jerk ≤ 15 m/s³ 的比例 | KING |
| **Jerk JSD** | 生成 vs GT jerk 分布的 Jensen-Shannon 散度 | AdvDiffuser |

标准指标结果自动写入 `evaluation_report.json` 的 `standard_metrics` 字段。

## 物理约束阈值

| 参数 | 值 | 说明 |
|------|-----|------|
| Valid Speed | ≤ 80 m/s | 覆盖高速公路场景（~288 km/h） |
| Max Accel (feasibility) | 8 m/s² | KING / AdvDiffuser 约定 |
| Max Jerk (feasibility) | 15 m/s³ | KING / AdvDiffuser 约定 |
| Collision threshold | 2.0 m | 车辆宽度之和的一半 |

## 示例输出

```
[Evaluation Summary]
Samples evaluated: 10
Margin met rate: 30.0%
Valid trajectory rate: 80.0%
Mean GT danger: 31.5
Mean Gen danger: 36.2
Danger improvement: +4.7

Results saved to: visualization_results/quick_validation_20260420_163000/
```

## 注意事项

1. **GPU 内存**: 批量评估时注意 GPU 内存限制
2. **Bokeh 依赖**: Bokeh 模式需要安装 `bokeh` 包 (`pip install bokeh`)
3. **模型路径**: 确保 `model_path` 正确指向 checkpoint
4. **数据路径**: `data_dir` 应包含 `danger_pair_features_*.pkl` 文件

## 扩展用法

### Python API

```python
from visualization import MatplotlibVisualizer

viz = MatplotlibVisualizer(
    model_path="best_model.pt",
    data_dir="diffusion_danger_model/danger_pair_scenarios"
)

results = viz.evaluate_samples(num_samples=20)
```

### 自定义分析

```python
from visualization import recover_trajectory_from_increment, check_trajectory_validity

# 从增量恢复轨迹
traj = recover_trajectory_from_increment(increments, start_pos, start_heading, vx, vy)

# 检查有效性
validity = check_trajectory_validity(traj)
print(validity['metrics'])
```

## 与 nuBoard 集成

NuBoard 完整集成需要:
1. NuPlan scenario 数据格式
2. 地图数据库连接
3. 仿真结果路径

当前 Bokeh 模式提供轻量级替代方案，无需完整 NuPlan 依赖。
