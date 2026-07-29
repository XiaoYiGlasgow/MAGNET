# 危险场景生成系统 — 入口导航

> 适用项目：`项目 B（危险场景生成 / diffusion / danger visualization）`
>
> 本文件现位于 `docs/` 根下，作为项目 B 的主入口说明。

基于 DDPM + GNN+Trans Encoder 的对抗性驾驶场景生成。

## 技术报告

- `reports/danger_scene_generation_report.html` — 完整 HTML 技术报告（含实验结果）
- `reports/danger_scene_generation_talk.md` — 汇报演讲稿

## 主体脚本（根目录）

| 脚本 | 用途 |
|------|------|
| `train_diffusion_gnn_trans.py` | **主训练入口**（当前版本，v4.0） |
| `generate_danger_scenes.py` | **主推理入口**，支持 stats/quick/nuboard/full/bokeh 五种模式 |
| `run_full_training_DEPRECATED.py` | 旧版训练脚本（v3.x，已废弃） |
| `train_diffusion_generator.py` | 最早期训练脚本（v1/v2，使用 SimpleSceneEncoder） |
| `process_nuplan_to_12d.py` | 数据预处理：NuPlan pkl → 12维 polyline 特征 |
| `process_by_scenario_type.py` | 数据预处理：按场景类型分组 |

## 消融实验

- `danger_gen_ablation/` — 全部消融代码（A1 w/o DangerLoss / A2 w/o SVO / A3 w/o GNN / A4 N=1）

## 模型与数据

| 路径 | 内容 |
|------|------|
| `experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt` | **当前最优模型** |
| `experiments/danger_generation/training_runs/` | 所有版本训练记录（v3.1 ~ v3.11） |
| `diffusion_danger_model/processed_data/` | 训练用预处理数据 |
| `diffusion_danger_model/danger_pair_scenarios/` | 危险对场景数据 |

## 核心库代码

| 路径 | 内容 |
|------|------|
| `src/danger_generation/models/danger_diffusion.py` | DDPM 模型 |
| `src/danger_generation/models/scene_condition_encoder.py` | 条件编码器 |
| `src/danger_generation/visualization/` | 可视化模块 |
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` | GNN+Trans 场景编码器（共享目录） |
| `utils/danger_score.py` | 危险打分 |
| `utils/danger_score_torch.py` | 可微分危险打分（训练用） |

## 当前最优结果（v3.10）

| 指标 | 数值 |
|------|------|
| Margin Met Rate | 87.5% |
| Valid Rate (≤80m/s) | 100% |
| Danger Improvement | +369.62 |
| Collision Rate | 47.5% |
