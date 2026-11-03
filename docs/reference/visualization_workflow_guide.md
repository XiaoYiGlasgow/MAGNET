# Visualisation Troubleshooting and Code Location Quick Reference

> For the full usage commands, see [tutorials/03_danger_scene_generation.md](tutorials/03_danger_scene_generation.md);
> this document records only troubleshooting points and key code locations.

---

## Troubleshooting Common Issues

### `No module named torch` / `No module named transformer_predictor`

The correct environment is not activated, or the command is run from the wrong directory:

```bash
conda activate nuplan
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
```

### `Processed graph file not found`

When using a `gnn_trans_encoder` checkpoint, the corresponding `processed_data/features_*.pkl` files are required:

```bash
ls diffusion_danger_model/processed_data/ | wc -l   # should be 804
```

If it is empty, first run `diffusion_danger_model/process_pittsburgh.py` to redo the preprocessing.

### Valid Rate is 0%

The model version does not match the code. Models earlier than v3.9 do not support kinematically parameterised output; switch to the current best model:

```bash
# Always use this path
experiments/danger_generation/training_runs/full_training_v3_9/best_model.pt
```

### `generate() got an unexpected keyword argument` / JSON serialisation error

These bugs have been fixed in the current code. If the error still occurs, an old cached module has been imported; restarting the Python process is sufficient.

### `Warning: nuPlan imports not available`

Non-blocking warning: none of the four modes `quick` / `bokeh` / `stats` / `full` depends on NuPlan;
only `--mode nuboard` requires the NuPlan environment, so this warning can be ignored.

### CUDA out of memory

Lower the inference batch size by reducing `--num_inference_samples` in `run_evaluation.py` (default 8; change to 1 or 2):

```bash
python visualization/run_evaluation.py --mode stats \
    --num_inference_samples 2 ...
```

### Recommendations for the First Run

First run a single-sample smoke test with `--num_samples 1`, and scale up after confirming that the output is correct:

```bash
python visualization/run_evaluation.py --mode quick --num_samples 1 ...
```

---

## Key Code Locations

| Function | File |
|------|------|
| Unified evaluation entry point | `src/danger_generation/visualization/run_evaluation.py` |
| Quick / Stats visualisation | `src/danger_generation/visualization/matplotlib_visualizer.py` |
| Bokeh / NuBoard visualisation | `src/danger_generation/visualization/nuboard_visualizer.py` |
| Trajectory format conversion (kinematic decoding) | `src/danger_generation/visualization/trajectory_converter.py` |
| Condition encoder | `src/danger_generation/models/scene_condition_encoder.py` |
| Main Diffusion model | `src/danger_generation/models/danger_diffusion.py` |
| Danger score computation (numpy) | `src/shared/utils/danger_score.py` |
| Danger score computation (torch, for training) | `src/shared/utils/danger_score_torch.py` |
| GNN+Trans encoder | `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` |
