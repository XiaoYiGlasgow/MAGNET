# Danger Scenario Generation — Ablation Experiments

This directory contains all the ablation experiment code for the danger scenario generation system (Diffusion + GNN+Trans).
It is fully isolated from the main scripts in the root directory and does not affect how the existing code runs.

## Ablation groups

| ID | Name | Comparison | Requires retraining |
|------|------|--------|-----------|
| A1 | `no_danger_loss` | Remove DangerExceedLoss to verify the necessity of danger supervision | ✅ |
| A2 | `no_svo` | Set the SVO input to zero to verify the effectiveness of the behavioural-intent condition | ✅ |
| A3 | `no_gnn` | Set the GNN features to zero to verify the contribution of the graph neural network branch | ✅ |
| A4 | `n1_inference` | N=1 at inference (no retraining) to verify the value of the Best-of-8 strategy | ❌ |

## Reference model

Comparison reference: `experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt`
- danger_weight=5.0, epoch 12, Margin Met=87.5%, Valid Rate=100%

## File description

```
danger_gen_ablation/
├── train_ablation.py     Training script (the --ablation argument drives A1/A2/A3)
├── eval_ablation.py      Evaluation script (supports all ablation groups, outputs a summary comparison table)
├── results/              Evaluation output directory for each ablation group
│   ├── no_danger_loss/
│   ├── no_svo/
│   ├── no_gnn/
│   ├── n1_inference/
│   └── ablation_summary.json   Final comparison table
└── README.md
```

## Path conventions

All scripts use `os.path.dirname(os.path.dirname(__file__))` to locate the GNN_Trans root directory;
the dependencies `models/`, `utils/` and `config.py` are all resolved from the root directory, so no existing files need to be modified.

## How to run

> Working directory: `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans`, with the `nuplan` environment activated.

```bash
# Train the ablation groups (about 50 epochs each, about 30~60 minutes per group)
python danger_gen_ablation/train_ablation.py --ablation no_danger_loss
python danger_gen_ablation/train_ablation.py --ablation no_svo
python danger_gen_ablation/train_ablation.py --ablation no_gnn

# Quick debugging (only 20 samples, 5 epochs)
python danger_gen_ablation/train_ablation.py --ablation no_danger_loss --max_samples 20 --epochs 5

# Evaluate all groups (including A4 N=1 and the Full reference value)
python danger_gen_ablation/eval_ablation.py --num_samples 40 --device cuda

# Evaluate specific groups only
python danger_gen_ablation/eval_ablation.py --groups A1 A4 Full
```

Ablation models are saved to `danger_gen_ablation/results/<name>/best_model.pt`,
and the evaluation summary is written to `danger_gen_ablation/results/ablation_summary.json`.

## Implementation details

| Ablation group | Implementation |
|--------|---------|
| A1 no_danger_loss | `DangerExceedLoss(danger_weight=0.0)` |
| A2 no_svo | `svo_angle = torch.zeros_like(svo_angle)` in the training loop |
| A3 no_gnn | After `get_encoder_features()`, the last 128 dimensions are zeroed (indices 256:384) |
| A4 N=1   | `eval_ablation.py` uses the v3.10 model, `num_inference_samples=1` |
