# Danger Scenario Generation System — Entry Navigation

> Project: `Project B (danger scenario generation / diffusion / danger visualization)`
>
> This file now sits at the root of `docs/` and serves as the main entry description for Project B.

Adversarial driving scenario generation based on DDPM + a GNN+Trans Encoder.

## Technical reports

- `reports/danger_scene_generation_report.html` — Full HTML technical report (including experimental results)
- `reports/danger_scene_generation_talk.md` — Presentation script

## Main scripts (root directory)

| Script | Purpose |
|------|------|
| `train_diffusion_gnn_trans.py` | **Main training entry point** (current version, v4.0) |
| `generate_danger_scenes.py` | **Main inference entry point**, supports five modes: stats/quick/nuboard/full/bokeh |
| `run_full_training_DEPRECATED.py` | Old training script (v3.x, deprecated) |
| `train_diffusion_generator.py` | Earliest training script (v1/v2, uses SimpleSceneEncoder) |
| `process_nuplan_to_12d.py` | Data preprocessing: NuPlan pkl → 12-dimensional polyline features |
| `process_by_scenario_type.py` | Data preprocessing: grouping by scenario type |

## Ablation experiments

- `danger_gen_ablation/` — All ablation code (A1 w/o DangerLoss / A2 w/o SVO / A3 w/o GNN / A4 N=1)

## Models and data

| Path | Content |
|------|------|
| `experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt` | **Current best model** |
| `experiments/danger_generation/training_runs/` | Training records for all versions (v3.1 ~ v3.11) |
| `diffusion_danger_model/processed_data/` | Preprocessed data for training |
| `diffusion_danger_model/danger_pair_scenarios/` | Danger-pair scenario data |

## Core library code

| Path | Content |
|------|------|
| `src/danger_generation/models/danger_diffusion.py` | DDPM model |
| `src/danger_generation/models/scene_condition_encoder.py` | Condition encoder |
| `src/danger_generation/visualization/` | Visualisation module |
| `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` | GNN+Trans scene encoder (shared directory) |
| `utils/danger_score.py` | Danger scoring |
| `utils/danger_score_torch.py` | Differentiable danger scoring (for training) |

## Current best results (v3.10)

| Metric | Value |
|------|------|
| Margin Met Rate | 87.5% |
| Valid Rate (≤80m/s) | 100% |
| Danger Improvement | +369.62 |
| Collision Rate | 47.5% |
