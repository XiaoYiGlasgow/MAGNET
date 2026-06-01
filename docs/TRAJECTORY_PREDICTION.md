# 轨迹预测系统 — 入口导航

> 适用项目：`项目 A（轨迹预测 / baseline / planning）`
>
> 本文件现位于 `docs/` 根下，作为项目 A 的主入口说明。

基于 GNN+Transformer 的自动驾驶轨迹预测，含 Transformer-only 和 LSTM baseline。

## 主体脚本（根目录）

| 脚本 | 用途 |
|------|------|
| `train.py` | **GNN+Trans 训练入口** |
| `train_transformer_only.py` | Transformer-only baseline 训练 |
| `train_lstm.py` | LSTM baseline 训练 |
| `run_gnn_manifest_eval.py` | GNN+Trans 评估 |
| `run_transformer_only_manifest_eval.py` | Transformer-only 评估 |
| `run_idm_pkl_eval.py` | IDM baseline 评估 |
| `run_lstm_pkl_eval.py` | LSTM baseline 评估 |
| `run_nuboard_simulation.py` | NuBoard 闭环仿真 |
| `idm_predictor.py` | IDM 规划器模型 |
| `lstm_predictor.py` | LSTM 预测器模型 |

## Shell 脚本

| 脚本 | 用途 |
|------|------|
| `run_lstm_train_and_eval.sh` | LSTM 一键训练+评估 |
| `run_medium_speed_3000_comparison.sh` | medium_speed 场景 3000 样本对比实验 |
| `run_medium_speed_3000_from_manifest.sh` | 从 manifest 运行对比实验 |

## 分析脚本（根目录）

| 脚本 | 用途 |
|------|------|
| `analyze_training_results.py` | 训练结果分析 |
| `analyze_distribution.py` | 数据分布分析 |
| `aggregate_result_type_logs.py` | 多类型结果汇总 |
| `advanced_analysis.py` | 高级统计分析 |
| `visualize_evaluation.py` | 评估结果可视化 |
| `visualize_training_logs.py` | 训练曲线可视化 |
| `heatmap_only.py` | 热力图生成 |
| `generate_medium_speed_manifest.py` | 生成 medium_speed 场景清单 |
| `generate_stratified_manifest.py` | 生成分层采样清单 |

## 模型与数据

| 路径 | 内容 |
|------|------|
| `experiments/analysis/gnn_trans_ablation/` | GNN+Trans 消融实验结果 |
| `experiments/baselines/` | Transformer-only / LSTM baseline 训练记录 |
| `data/processed_data/` | 轨迹预测训练数据 |
| `data/processed_data_by_type/` | 按场景类型分组的数据 |
| `data/model_weights/` | GNN+Trans 模型权重 |
| `configs/` | Hydra 仿真配置 |

## 核心库代码

| 路径 | 内容 |
|------|------|
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` | GNN+Trans 模型 |
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_only_predictor.py` | Transformer-only 模型 |
| `src/ego_planning/planner/` | NuPlan planner 接口 |
| `utils/dataset.py` | 数据集加载 |
| `config.py` | 全局配置 |
