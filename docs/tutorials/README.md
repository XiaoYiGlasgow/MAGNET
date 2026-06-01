# GNN_Trans 使用教程总览

> 文档约定：
> - `项目 A`：轨迹预测 / baseline / planning
> - `项目 B`：危险场景生成 / diffusion / danger visualization

本系列教程面向初次使用本项目的研究者，从零开始介绍如何运行轨迹预测和危险场景生成两条完整流水线。

---

## 项目是什么

GNN_Trans 是一个基于 NuPlan 数据集的自动驾驶研究框架，包含两条独立的研究线：

| 研究线 | 目标 | 核心方法 |
|--------|------|----------|
| **轨迹预测** | 预测周围车辆和自车的未来轨迹 | GNN + Transformer 编码器 |
| **危险场景生成** | 生成比真实数据更危险的对抗场景 | Diffusion 模型 + 运动学参数化 |

两条线可以独立运行，也可以组合使用（Diffusion 生成的危险轨迹可以作为 NuPlan 仿真的 challenge 输入）。

---

## 教程文件列表

| 文件 | 内容 | 建议阅读时间 |
|------|------|-------------|
| [01_environment_and_structure.md](01_environment_and_structure.md) | 环境配置、项目目录结构、运行前必读 | 10 分钟 |
| [02_trajectory_prediction.md](02_trajectory_prediction.md) | 轨迹预测全流程：数据 → 训练 → 评估 → 仿真 | 20 分钟 |
| [03_danger_scene_generation.md](03_danger_scene_generation.md) | 危险场景生成全流程：训练 → 评估 → 消融实验 → 批量生成 → 可视化 | 20 分钟 |
| [04_claude_prompt_templates.md](04_claude_prompt_templates.md) | Claude 常用提示词模板：调试、改代码、实验分析、写记录 | 10 分钟 |

---

## 5 分钟快速验证（什么都不用改，直接跑）

如果你只想先确认环境和代码都正常，执行以下命令：

```bash
# 激活环境
conda activate nuplan

# 1. 轨迹预测：NuPlan 规划仿真（约 10 秒）
cd /mnt/local/Xiao_Yi/nuplan-devkit
python GNN_Trans/run_nuboard_simulation.py --no-nuboard
# 预期输出：Number of successful simulations: 2, failed: 0

# 2. 危险场景生成：统计评估当前最佳模型（约 2 分钟）
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python visualization/run_evaluation.py \
    --mode stats \
    --model_path experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt \
    --data_dir diffusion_danger_model/danger_pair_scenarios \
    --processed_data_dir diffusion_danger_model/processed_data \
    --num_samples 40 --min_gt_danger 2.0 --device cuda
# 预期输出：Valid Rate 100%，Margin Met 87.5%，Danger Improvement +369.62
```

两条命令都跑通 → 环境完全正常，可以继续阅读详细教程。

---

## 快速查找入口

`docs/` 根下有两个导航文件，按研究线分类列出了所有脚本和数据路径：

- [`../TRAJECTORY_PREDICTION.md`](../TRAJECTORY_PREDICTION.md) — 项目 A：轨迹预测 / baseline / planning
- [`../DANGER_GENERATION.md`](../DANGER_GENERATION.md) — 项目 B：危险场景生成 / diffusion / 可视化

---

## 遇到问题先检查这几点

1. **ModuleNotFoundError** → 确认在 `conda activate nuplan` 环境下运行
2. **路径不存在** → 确认工作目录正确（见各教程开头）
3. **CUDA out of memory** → 降低 `batch_size`（改为 4 或 2）
4. **NuPlan 数据库找不到** → 检查 `/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/` 是否挂载
