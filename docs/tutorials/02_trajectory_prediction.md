# Tutorial 02: Trajectory Prediction

This tutorial explains how to train and evaluate the GNN+Transformer trajectory prediction model, and how to run it in the NuPlan simulator.

---

## Working Directory

```bash
# Most training/evaluation commands are run from the GNN_Trans root directory
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
conda activate nuplan

# NuPlan simulation commands are run from the nuplan-devkit root directory (noted separately where relevant)
cd /mnt/local/Xiao_Yi/nuplan-devkit
```

---

## Overview: Three Models

| Model | Entry Script | Characteristics | Recommended Use |
|------|----------|------|----------|
| **GNN + Transformer** | `train.py` | Full graph neural network, highest accuracy | Main model |
| **Transformer Only** | `train_transformer_only.py` | GNN removed, Transformer only | Ablation study |
| **LSTM** | `train_lstm.py` | Classic sequence model | Baseline comparison |

Trained GNN+Transformer weights are already available: `data/model_weights/best_model.pth`, **so you can skip training and go straight to evaluation**.

---

## Step 1: Check the Training Data

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
ls data/processed_data_by_type/
```

Output:
```
processed_high_lateral_acceleration     ← high lateral acceleration scenarios
processed_high_magnitude_speed          ← high-speed scenarios
processed_medium_magnitude_speed        ← medium-speed scenarios (default training set)
processed_near_multiple_vehicles        ← multi-vehicle interaction scenarios
processed_traversing_intersection       ← intersection scenarios
```

Check the number of samples in the default training set:
```bash
ls data/processed_data_by_type/processed_medium_magnitude_speed/ | wc -l
# Output: 20002 (about 20,000 scenarios)
```

Each `.pkl` file is an independent driving scenario containing the historical trajectories (past 4 seconds) of the ego vehicle and up to 15 surrounding vehicles, together with map features.

---

## Step 2: (Optional) Modify the Training Configuration

Open `config.py` and modify the following parameters as needed:

```python
# Change the scenario type (default: medium_magnitude_speed)
TRAIN_PATH = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")
TEST_PATH  = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")

# Training hyperparameters
EPOCHS        = 100     # Number of epochs (50-100 recommended when training from scratch)
BATCH_SIZE    = 8       # Change to 4 if GPU memory is insufficient
LEARNING_RATE = 1e-3    # Initial learning rate

# Model architecture (usually no need to change)
EMBED_DIM  = 128
NUM_LAYERS = 3
NUM_HEADS  = 8
```

---

## Step 3: Train the Main GNN+Transformer Model

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python train.py
```

After launch, an interactive menu appears for selecting the adjacency matrix configuration:

```
============================================================
SELECT ADJACENCY MATRIX CONFIGURATION
============================================================
1. DEFAULT: Full connectivity between all entities
2. EGO_ONLY: Only ego vehicle connects to other entities
3. NO_EGO_ROAD: Ego doesn't connect to road elements
4. NEARBY_ONLY: Only connect nearby entities
5. SELECTIVE: Custom connection patterns
============================================================
Select configuration (1-5) [default=1]:
```

**Press Enter** to use the default configuration 1 (Full connectivity), which suits most cases.

Example output during training:
```
Epoch [1/100] | Train Loss: 2.3451 | Val Loss: 2.1203 | ADE: 1.8432 | FDE: 3.2105
Epoch [2/100] | Train Loss: 1.9823 | Val Loss: 1.8341 | ADE: 1.6201 | FDE: 2.8934
...
```

Lower **ADE** (Average Displacement Error) and **FDE** (Final Displacement Error) are better; both are in metres.

After training, the weights are saved automatically to:
```
data/model_weights/
└── best_model_<scenario_type>_<adjacency_config>.pth    ← best on validation set
└── final_model_<scenario_type>_<adjacency_config>.pth   ← final epoch
```

Training records are saved to:
```
experiments/analysis/training_records/<experiment_name>/
```

---

## Step 4: Train the Transformer-Only Baseline (for Ablation)

```bash
python train_transformer_only.py \
    --train_path data/processed_data_by_type/processed_medium_magnitude_speed \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --device cuda:0
```

Weights are saved to:
```
experiments/baselines/transformer_only/
└── best_model_transformer_only.pth
```

---

## Step 5: Train the LSTM Baseline

```bash
python train_lstm.py
```

This also starts interactively; just press Enter to use the default configuration. Weights are saved to:
```
experiments/baselines/lstm/
```

---

## Step 6: NuPlan Planning Simulation (the Most Direct End-to-End Test)

This test runs the trained GNN+Transformer as a NuPlan planner, verifying that the model works correctly in a realistic simulation environment.

> **Note**: This command must be run from the `nuplan-devkit` root directory, not the GNN_Trans directory.

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
python GNN_Trans/run_nuboard_simulation.py --no-nuboard
```

Successful output:
```
...
Number of successful simulations: 2
Number of failed simulations: 0
Finished running simulation!
```

### Visualising Simulation Results (NuBoard)

To view the simulated trajectories in a browser, remove `--no-nuboard`:

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
python GNN_Trans/run_nuboard_simulation.py
```

Then open `http://localhost:5006` in a browser to see the ego vehicle's trajectory and the surrounding vehicles.

---

## Step 7: Evaluate Model Performance (ADE/FDE Metrics)

`run_gnn_manifest_eval.py` computes ADE/FDE in batch on a specified test set.

First, prepare a manifest JSON file listing the test samples:

```bash
# Check whether a manifest file already exists
find experiments/ -name "*.json" | head -10
find data/ -name "manifest*.json" | head -5
```

If none exists, you can quickly generate one (using a Python script that lists the pkl files in the test directory):

```python
# Run this code in the GNN_Trans directory to generate the manifest
import os, json
data_dir = "data/processed_data_by_type/processed_medium_magnitude_speed"
files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
# Take the last 10% as the test set
test_files = files[int(len(files)*0.9):]
manifest = {"sample_files": test_files}
with open("data/test_manifest.json", "w") as f:
    json.dump(manifest, f)
print(f"Generated manifest with {len(test_files)} test samples")
```

Then run the evaluation:

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python run_gnn_manifest_eval.py \
    --model_path data/model_weights/best_model.pth \
    --data_dir data/processed_data_by_type/processed_medium_magnitude_speed \
    --manifest data/test_manifest.json \
    --adjacency_config nearby_only \
    --device cuda:0 \
    --output_json experiments/analysis/gnn_eval_result.json
```

Parameters:

| Parameter | Description |
|------|------|
| `--model_path` | Path to the model weights |
| `--data_dir` | Test data directory (the directory containing the pkl files) |
| `--manifest` | JSON list of test samples |
| `--adjacency_config` | Adjacency matrix configuration (must match the one used in training) |
| `--device` | `cpu` or `cuda:0` |
| `--output_json` | Path for saving results (optional) |

---

## Step 8: Evaluate the LSTM and IDM Baselines

```bash
# Evaluate LSTM
python run_lstm_pkl_eval.py

# Evaluate IDM (Intelligent Driver Model, rule-based)
python run_idm_pkl_eval.py
```

---

## FAQ

### Q: train.py fails to import `GraphDataset`

```
ModuleNotFoundError: No module named 'dataset'
```

Cause: not running in the correct conda environment, or `torch_geometric` is not installed.

Solution:
```bash
conda activate nuplan
python -c "from torch_geometric.data import Data; print('OK')"
```

### Q: Training is slow

- Check whether the GPU is being used: `python -c "import torch; print(torch.cuda.is_available())"`
- Too small a `BATCH_SIZE` in `config.py` leads to low GPU utilisation; increase it appropriately

### Q: Training loss does not converge

- Lower `LEARNING_RATE` (from `1e-3` to `5e-4` or `1e-4`)
- Check that the data path is correct: `ls data/processed_data_by_type/processed_medium_magnitude_speed/ | head -5`

### Q: NuPlan simulation reports a Hydra configuration error

Make sure you run from the `nuplan-devkit` root directory (not the `GNN_Trans` directory):
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit  # ← must be this directory
python GNN_Trans/run_nuboard_simulation.py --no-nuboard
```

---

## Reference: Output Directory for Each Script

| Script | Where Weights/Results Are Saved |
|------|-----------------|
| `train.py` | `data/model_weights/` + `experiments/analysis/training_records/` |
| `train_transformer_only.py` | `experiments/baselines/transformer_only/` |
| `train_lstm.py` | `experiments/baselines/lstm/` |
| `run_gnn_manifest_eval.py` | Path specified by `--output_json` |
| `run_nuboard_simulation.py` | `/mnt/local/Xiao_Yi/gnn_trans_simulation_results/` |
