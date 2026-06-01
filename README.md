# MAGNET

MAGNET is a trajectory prediction and planning project built on top of the nuPlan dataset and simulation stack. This repository contains the current public trajectory-prediction code path, including the main GNN + Transformer model, Transformer-only and LSTM baselines, preprocessing scripts, evaluation entrypoints, and nuPlan simulation integration.

## Highlights

- GNN + Transformer trajectory prediction pipeline
- Transformer-only and LSTM baselines
- Preprocessing scripts for nuPlan `.pkl` scene data
- Batch evaluation scripts for ADE / FDE comparison
- nuPlan closed-loop simulation entrypoints
- Project docs, tutorials, and result summaries

## Repository Layout

```text
MAGNET/
├── configs/                       # Hydra and simulation configs
├── docs/                          # Documentation, tutorials, reports
├── scripts/
│   ├── analysis/                  # Analysis and visualization scripts
│   └── preprocessing/             # Dataset preprocessing scripts
├── src/
│   ├── ego_planning/              # Planner and nuBoard integration
│   └── shared/utils/              # Shared data and model utilities
├── config.py                      # Main training configuration
├── train.py                       # GNN + Transformer training
├── train_transformer_only.py      # Transformer-only baseline
├── train_lstm.py                  # LSTM baseline
├── run_gnn_manifest_eval.py       # GNN + Transformer evaluation
├── run_transformer_only_manifest_eval.py
├── run_lstm_pkl_eval.py
├── run_idm_pkl_eval.py
└── run_nuboard_simulation.py      # nuPlan simulation entry
```

## Environment Setup

### 1. Install `nuplan-devkit`

This project is designed to live inside a `nuplan-devkit` workspace.

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

conda create -n nuplan python=3.9
conda activate nuplan

pip install -e .
pip install torch torchvision torchaudio
pip install torch-geometric pandas matplotlib seaborn scipy
```

### 2. Clone MAGNET into the workspace

```bash
cd /path/to/nuplan-devkit
git clone git@github.com:XiaoYiGlasgow/MAGNET.git
cd MAGNET
```

### 3. Configure dataset paths

Set the required nuPlan paths in your shell environment and project config.

```bash
export NUPLAN_DATA_ROOT="/path/to/nuplan/data"
export NUPLAN_MAPS_ROOT="/path/to/nuplan/maps"
```

Then adjust dataset paths in `config.py` if needed.

## Data Preparation

The training and evaluation pipeline expects processed scene files in `.pkl` format.

Typical layout:

```text
data/
├── processed_data/
├── processed_data_by_type/
│   ├── processed_medium_magnitude_speed/
│   ├── processed_high_magnitude_speed/
│   ├── processed_near_multiple_vehicles/
│   └── ...
└── model_weights/
```

Useful preprocessing scripts:

```bash
python scripts/preprocessing/process_nuplan_to_12d.py
python scripts/preprocessing/process_by_scenario_type.py
python scripts/preprocessing/generate_medium_speed_manifest.py
python scripts/preprocessing/generate_stratified_manifest.py
```

## Quick Start

### Train the main GNN + Transformer model

```bash
cd /path/to/nuplan-devkit/MAGNET
conda activate nuplan
python train.py
```

`train.py` launches an adjacency-configuration prompt. Press Enter to use the default configuration if you do not need a custom graph setting.

### Train the Transformer-only baseline

```bash
python train_transformer_only.py \
    --train_path data/processed_data_by_type/processed_medium_magnitude_speed \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --device cuda:0
```

### Train the LSTM baseline

```bash
python train_lstm.py
```

## Evaluation

### Evaluate the GNN + Transformer model

```bash
python run_gnn_manifest_eval.py \
    --model_path data/model_weights/best_model.pth \
    --data_dir data/processed_data_by_type/processed_medium_magnitude_speed \
    --manifest data/test_manifest.json \
    --adjacency_config nearby_only \
    --device cuda:0 \
    --output_json experiments/analysis/gnn_eval_result.json
```

### Evaluate the baselines

```bash
python run_transformer_only_manifest_eval.py
python run_lstm_pkl_eval.py
python run_idm_pkl_eval.py
```

### Compare multiple methods on a fixed subset

```bash
bash run_medium_speed_3000_comparison.sh
```

or

```bash
bash run_medium_speed_3000_from_manifest.sh
```

## Closed-Loop Simulation

To run the planner inside the nuPlan simulation stack:

```bash
cd /path/to/nuplan-devkit
conda activate nuplan
python MAGNET/run_nuboard_simulation.py --no-nuboard
```

To launch the visualization UI:

```bash
python MAGNET/run_nuboard_simulation.py
```

If the run succeeds, nuBoard can be opened in the browser, typically at `http://localhost:5006`.

## Documentation

- `docs/README.md`: documentation index
- `docs/TRAJECTORY_PREDICTION.md`: project-level navigation
- `docs/tutorials/02_trajectory_prediction.md`: step-by-step tutorial
- `docs/reports/summary.md`: result summary
- `docs/reports/results_analysis_report.md`: detailed analysis report

## Notes

- This public repository tracks the trajectory-prediction line only. The separate danger-generation work is intentionally excluded.
- Large datasets, model checkpoints, experiment outputs, and private work records are not included.
- Some internal documents may still mention the historical local folder name `GNN_Trans`; in this public repo, use `MAGNET` instead.
