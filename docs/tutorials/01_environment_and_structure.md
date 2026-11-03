# Tutorial 01: Environment Setup and Project Structure

## 1. Runtime environment

### Conda environment

All commands in this project must be run in the `nuplan` environment:

```bash
conda activate nuplan
```

Verify that the environment is correct:

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Example of normal output:
```
PyTorch: 2.x.x
CUDA: True
```

### Working directory conventions

Commands related to **trajectory prediction** are run from the **nuplan-devkit root directory** (required by NuPlan Hydra simulation):
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
```

Commands related to **danger scenario generation** are run from the **GNN_Trans root directory**:
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
```

> **Remember this difference**: most reported path errors are caused by the wrong working directory.

---

## 2. Project directory structure

```
GNN_Trans/
│
├── config.py                        ← Global configuration (paths, hyperparameters) ★ most frequently modified
│
│   [Trajectory prediction entry points]
├── train.py                         ← GNN+Transformer trajectory prediction training
├── train_transformer_only.py        ← Transformer-only baseline training
├── train_lstm.py                    ← LSTM baseline training
├── run_gnn_manifest_eval.py         ← GNN+Trans evaluation
├── run_nuboard_simulation.py        ← NuPlan planning simulation
├── idm_predictor.py / lstm_predictor.py  ← Model inference modules
│
│   [Danger scenario generation entry points]
├── train_diffusion_gnn_trans.py     ← Diffusion training (current main entry point) ★
├── generate_danger_scenes.py        ← Batch generation of danger scenarios (inference) ★
├── run_full_training_DEPRECATED.py  ← Old training script (v3.x, deprecated)
│
├── danger_gen_ablation/             ← Ablation experiments (danger scenario generation)
│   ├── README.md                    ← Ablation experiment description
│   ├── train_ablation.py            ← Ablation training (to be written)
│   ├── eval_ablation.py             ← Ablation evaluation (to be written)
│   └── results/                     ← Evaluation output for each ablation group
│
├── configs/                         ← NuPlan Hydra configuration files
│   ├── simple_simulation.yaml       ← Simulation entry configuration
│   └── planner/                     ← planner configuration
│
├── data/                            ← All data (not in Git)
│   ├── model_weights/               ← GNN+Trans trajectory prediction weights
│   │   └── best_model.pth           ★
│   ├── processed_data_by_type/      ← Training data classified by scenario type
│   │   └── processed_medium_magnitude_speed/  ← Default training set (20002 scenarios)
│   └── ...
│
├── diffusion_danger_model/          ← Data dedicated to danger scenario generation
│   ├── processed_data/              ← Diffusion training data (803 pkl) ★
│   ├── danger_pair_scenarios/       ← Danger pair scenarios (labelled, 390) ★
│   └── ttc_filter_results/
│
├── experiments/                     ← All experiment results (not in Git)
│   ├── danger_generation/
│   │   ├── training_runs/
│   │   │   ├── full_training_v3_10/ ← Current best model ★ (Margin Met 87.5%)
│   │   │   │   └── best_model.pt
│   │   │   └── full_training_v3_1 ~ v3_11/  ← Historical versions
│   │   └── ablation/               ← Ablation experiment results (generated after running)
│   ├── analysis/gnn_trans_ablation/ ← GNN+Trans trajectory prediction ablation
│   └── baselines/                  ← Transformer-only / LSTM training records
│
├── src/                             ← Library code (shared by both research lines)
│   ├── danger_generation/
│   │   ├── models/                  ← danger_diffusion.py, scene_condition_encoder.py
│   │   └── visualization/           ← run_evaluation.py ★
│   ├── ego_planning/
│   │   ├── planner/                 ← GNNTransPlanner
│   │   └── nuboard/
│   └── shared/utils/                ← danger_score.py, dataset.py, etc.
│
└── docs/
    ├── DANGER_GENERATION.md                 ← Navigation for all entry points of the danger scenario generation research line ★
    ├── TRAJECTORY_PREDICTION.md             ← Navigation for all entry points of the trajectory prediction research line ★
    ├── reference/                           ← Technical notes and troubleshooting documents
    ├── reports/                             ← Result summaries, presentation drafts, technical reports
    └── tutorials/                           ← You are here
```

> Directories such as `utils/`, `models/`, `visualization/` under the root directory are all **symlinks**,
> pointing to the real locations under `src/`; no manual handling is needed.
> The entry-point descriptions of both research lines are now gathered under the `docs/` root; consult `docs/DANGER_GENERATION.md` or `docs/TRAJECTORY_PREDICTION.md` by research line.

---

## 3. Key configuration files in detail

### `config.py` — Global parameters (most frequently need modifying)

```python
# Data paths (usually no need to change, unless the mount point has changed)
DATA_PATH   = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_PATH    = '/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps'

# Training data paths ← change here when switching scenario type
TRAIN_PATH  = ".../data/processed_data_by_type/processed_medium_magnitude_speed"
TEST_PATH   = ".../data/processed_data_by_type/processed_medium_magnitude_speed"

# Training hyperparameters
EPOCHS      = 100
BATCH_SIZE  = 8
LEARNING_RATE = 1e-3

# Model architecture
EMBED_DIM   = 128      # Transformer embedding dimension
NUM_LAYERS  = 3        # Number of Transformer layers
NUM_HEADS   = 8        # Number of multi-head attention heads
```

After modifying `config.py`, simply run the corresponding training script for the change to take effect; no command-line arguments are needed.

### `train_diffusion_gnn_trans.py` — Diffusion training configuration (CLI recommended)

The current training entry point no longer requires editing `CUSTOM_CONFIG` inside the script; instead, pass command-line arguments directly:

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

python train_diffusion_gnn_trans.py \
    --output_dir experiments/danger_generation/training_runs/v4_0_debug \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda
```

If you only want to check the data and the forward pass without actual training, you can add:

```bash
python train_diffusion_gnn_trans.py --test_only
```

---

## 4. Data flow of the two research lines

### Trajectory prediction

```
NuPlan raw database (.db files)
    │
    ↓ scripts/preprocessing/process_nuplan_to_12d.py
    │  (extract scene graph features, output .pkl files)
    │
data/processed_data_by_type/<scenario type>/   ← 20002 .pkl
    │
    ↓ train.py
    │  (GNN encoding + Transformer prediction)
    │
data/model_weights/best_model.pth          ← saved weights
    │
    ↓ run_nuboard_simulation.py / run_gnn_manifest_eval.py
    │
Simulation results / ADE·FDE evaluation metrics
```

> **Good news**: `processed_data_by_type/` and `model_weights/best_model.pth` already exist,
> so you can skip straight to the training or evaluation step.

### Danger scenario generation

```
NuPlan raw database
    │
    ↓ diffusion_danger_model/process_pittsburgh.py
    │  (extract 12D scene features)
    │
diffusion_danger_model/processed_data/      ← 804 .pkl (already present)
    │
    ↓ utils/ttc_filter.py
    │  (TTC filtering, find pairs with collision risk)
    │
diffusion_danger_model/danger_pair_scenarios/ ← 81 .pkl (already present)
    │
    ↓ train_diffusion_gnn_trans.py
    │  (DDPM Diffusion training, v4.0: TrajectoryPredictionHead + dynamic danger_target)
    │
experiments/.../full_training_v3_10/best_model.pt ← current best (already present, Margin Met 87.5%)
    │
    ├─→ visualization/run_evaluation.py   ← statistical evaluation
    │
    └─→ generate_danger_scenes.py         ← batch generation of new scenarios
```

> **Good news**: all data and models are ready, so you can run evaluation and generation directly.
