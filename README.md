# MAGNET

Trajectory prediction and planning code built on the nuPlan dataset.

This public repository tracks the current trajectory-prediction project line from the private development repo. It keeps the active training, evaluation, simulation, and documentation structure, and does not include the separate danger-generation work.

## Main Entry Points

- `train.py`: GNN + Transformer training
- `train_transformer_only.py`: Transformer-only baseline training
- `train_lstm.py`: LSTM baseline training
- `run_gnn_manifest_eval.py`: GNN + Transformer evaluation
- `run_transformer_only_manifest_eval.py`: Transformer-only evaluation
- `run_idm_pkl_eval.py`: IDM baseline evaluation
- `run_lstm_pkl_eval.py`: LSTM baseline evaluation
- `run_nuboard_simulation.py`: nuPlan simulation entry

## Current Structure

```text
MAGNET/
├── configs/                  # Hydra and simulation configs
├── docs/                     # Project docs, tutorials, and reports
├── scripts/
│   ├── analysis/             # Analysis and visualization scripts
│   └── preprocessing/        # Dataset preprocessing utilities
├── src/
│   ├── ego_planning/         # Planner and nuBoard integration
│   └── shared/utils/         # Shared data and model utilities
├── config.py                 # Training configuration
├── idm_predictor.py          # IDM baseline model
├── lstm_predictor.py         # LSTM baseline model
├── train.py
├── train_lstm.py
├── train_transformer_only.py
└── run_*                     # Evaluation and simulation entrypoints
```

## Documentation

- `docs/README.md`: documentation index
- `docs/TRAJECTORY_PREDICTION.md`: project-level trajectory prediction guide
- `docs/tutorials/02_trajectory_prediction.md`: step-by-step training and evaluation tutorial
- `docs/reports/summary.md`: summary of trajectory-prediction results

## Notes

- The examples in some internal docs may still show a local folder name like `GNN_Trans` from the private development environment. If you clone this public repo as `MAGNET`, replace that path accordingly when running commands.
- Large data, checkpoints, training outputs, and private work records are intentionally excluded from this public repo.
