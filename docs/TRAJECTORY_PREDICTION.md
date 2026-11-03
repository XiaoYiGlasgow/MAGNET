# Trajectory Prediction System — Entry Guide

> Applies to: `Project A (trajectory prediction / baseline / planning)`
>
> This file now lives at the root of `docs/` and serves as the main entry documentation for Project A.

GNN+Transformer-based trajectory prediction for autonomous driving, including Transformer-only and LSTM baselines.

## Main Scripts (Root Directory)

| Script | Purpose |
|------|------|
| `train.py` | **GNN+Trans training entry point** |
| `train_transformer_only.py` | Transformer-only baseline training |
| `train_lstm.py` | LSTM baseline training |
| `run_gnn_manifest_eval.py` | GNN+Trans evaluation |
| `run_transformer_only_manifest_eval.py` | Transformer-only evaluation |
| `run_idm_pkl_eval.py` | IDM baseline evaluation |
| `run_lstm_pkl_eval.py` | LSTM baseline evaluation |
| `run_nuboard_simulation.py` | NuBoard closed-loop simulation |
| `idm_predictor.py` | IDM planner model |
| `lstm_predictor.py` | LSTM predictor model |

## Shell Scripts

| Script | Purpose |
|------|------|
| `run_lstm_train_and_eval.sh` | One-step LSTM training + evaluation |
| `run_medium_speed_3000_comparison.sh` | Comparison experiment on 3000 medium_speed scenario samples |
| `run_medium_speed_3000_from_manifest.sh` | Run the comparison experiment from a manifest |

## Analysis Scripts (Root Directory)

| Script | Purpose |
|------|------|
| `analyze_training_results.py` | Training result analysis |
| `analyze_distribution.py` | Data distribution analysis |
| `aggregate_result_type_logs.py` | Aggregation of results across multiple types |
| `advanced_analysis.py` | Advanced statistical analysis |
| `visualize_evaluation.py` | Evaluation result visualisation |
| `visualize_training_logs.py` | Training curve visualisation |
| `heatmap_only.py` | Heatmap generation |
| `generate_medium_speed_manifest.py` | Generate the medium_speed scenario manifest |
| `generate_stratified_manifest.py` | Generate a stratified sampling manifest |

## Models and Data

| Path | Contents |
|------|------|
| `experiments/analysis/gnn_trans_ablation/` | GNN+Trans ablation results |
| `experiments/baselines/` | Transformer-only / LSTM baseline training records |
| `data/processed_data/` | Trajectory prediction training data |
| `data/processed_data_by_type/` | Data grouped by scenario type |
| `data/model_weights/` | GNN+Trans model weights |
| `configs/` | Hydra simulation configuration |

## Core Library Code

| Path | Contents |
|------|------|
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` | GNN+Trans model |
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_only_predictor.py` | Transformer-only model |
| `src/ego_planning/planner/` | NuPlan planner interface |
| `utils/dataset.py` | Dataset loading |
| `config.py` | Global configuration |
