# MAGNET

MAGNET (Multi-agent Attributed Graph Network Transformer) is research code for joint multi-agent,
multimodal trajectory prediction on nuPlan. It accompanies the manuscript *Counterfactual
Supervision of Adaptive Interaction Topology for Multi-Vehicle Trajectory Prediction*.

The model keeps a separate token for every vehicle and lane element from encoding to decoding, so
that individual source and target entities remain addressable. On top of this representation, the
repository implements a source-to-target counterfactual interaction utility: the change in a target
vehicle's prediction error when every vehicle-to-vehicle connection incident to a source vehicle is
masked, while the source's observed state and lane relations are kept. The utility is used to
supervise scene-dependent weights on vehicle-to-vehicle edges and is tested against oracle,
shuffled and random weighting controls.

This repository publishes source code and small provenance manifests only. nuPlan data, processed
features, checkpoints, experiment outputs and internal work records are not included.

This is an active research snapshot rather than a packaged Python library. It supports source and
protocol inspection; it does not provide a one-command reproduction of the manuscript tables.

## What is in this release

- **Joint predictor** (`scripts/training/magnet_joint_*.py`): entity encoder with attention pooling
  over polyline segments, ego-history encoding and fusion, a complete directed candidate graph
  without self-loops, two layers of relation-aware message passing over four directed relation types
  (vehicle→vehicle, vehicle→lane, lane→vehicle, lane→lane) with 10-dimensional edge attributes, and a
  shared decoder that predicts `K` trajectories and mode probabilities for the ego vehicle and every
  represented surrounding vehicle. No scene-level pooling is used on the prediction path.
- **Edge gating**: an edge scorer that assigns run-time weights to vehicle-to-vehicle edges from
  observed inputs only, renormalised per target; lane-related edges keep unit weight.
- **Counterfactual utility labels** (`scripts/analysis/s3_edge_utility_labels.py`) and the
  within-target ranking supervision used to train the scorer.
- **Weighting controls**: oracle, shuffled and random utility weights injected as edge weights
  (`scripts/training/build_oracle_gate_cache.py`). The oracle uses ground-truth futures and is an
  offline reference, not a deployable configuration.
- **Data splits**: log-disjoint train/validation splits and a frozen test split built from logs
  that appear in neither (`scripts/training/build_fullret_test_split.py`).
- **Evaluation** on the frozen test split (`scripts/analysis/eval_joint_test.py`): top-1 ADE/FDE,
  minADE@K and mode-selection hit rate for the ego vehicle and surrounding vehicles.
- Unit tests for the joint model, edge scorer, utility labels and data contracts under `tests/`.
- Vendored snapshots of the modified nuPlan integration files (`devkit_patches/`), and a
  reconstructed closed-loop evaluation harness (`scripts/eval_closed_loop/`).

The repository also retains an earlier single-output imitation pipeline (route features, ego
perturbation, 2×2 runner, planning layer), danger-generation utilities and pre-v8 code for
continuity. These are not part of the joint-prediction method; see
[Earlier pipeline](#earlier-pipeline). Files under `archive/` are historical.

## Repository layout

```text
MAGNET/
├── configs/                       # nuPlan and Hydra configuration
├── devkit_patches/                # vendored planner/model integration files
├── docs/                          # schema, tutorials and technical notes
├── scripts/
│   ├── analysis/                  # utility labels, evaluation and analysis tools
│   ├── eval_closed_loop/          # reconstructed closed-loop harness
│   ├── preprocessing/             # feature generation
│   └── training/                  # joint model, caches, splits and trainer
├── src/                           # shared utilities, planning layer and danger generation
├── tests/                         # unit tests for the joint model and data contracts
├── archive/                       # legacy pre-v8 source code
└── config.py
```

## Requirements

MAGNET is designed to be cloned directly inside a `nuplan-devkit` checkout:

```text
nuplan-devkit/
├── MAGNET/
├── models/
└── nuplan/
```

The development environment uses Python 3.9, PyTorch 2.0.1 and PyTorch Geometric 2.6.1 with a
locally prepared nuPlan devkit checkout. The training path also imports NumPy, pandas,
scikit-learn and PyYAML; analysis, visualisation and tests additionally use SciPy, Matplotlib,
Shapely, tqdm and pytest.

This repository does not include a lockfile and does not pin the upstream nuPlan commit. Follow the
[official nuPlan setup](https://github.com/motional/nuplan-devkit) first and record the exact
nuPlan, Python, CUDA, PyTorch and PyG versions used for an experiment.

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .

git clone https://github.com/XiaoYiGlasgow/MAGNET.git
cd MAGNET
```

## nuPlan integration files

Integration snapshots used by the development workspace are versioned under
`devkit_patches/files/`, including `models/transformer_predictor.py`, which several training and
calibration modules import. Inspect them before writing them into a nuPlan checkout.

```bash
# From nuplan-devkit/MAGNET
python devkit_patches/sync_devkit_patches.py --check

# After reviewing and backing up the target files, install the vendored snapshots.
python devkit_patches/sync_devkit_patches.py --push
python devkit_patches/sync_devkit_patches.py --check
```

On a fresh upstream checkout, the first `--check` may report missing files; this is expected and
shows what `--push` would create. `--push` writes into the parent `nuplan-devkit` tree and may
overwrite existing files.

The vendored `plantf_planner.yaml` is only a nuPlan registration file. The external
[PlanTF repository](https://github.com/jchengai/planTF), its environment and weights are not
vendored here.

## Data contract

The pipeline consumes precomputed feature files (`features_*.pkl`). The feature schema is documented
in [`docs/feature_schema_v2.md`](docs/feature_schema_v2.md).

- Segment features have 12 fields; the entity encoder reads the first nine.
- History is 20 frames (2 s at 10 Hz); the prediction horizon is 60 frames (6 s at 10 Hz).
- Joint targets are absolute positions in the ego-centred frame at prediction time, for the ego
  vehicle and for every represented surrounding vehicle with a valid future. Surrounding-vehicle
  targets are attached by the agent cache below; truncated futures are masked frame by frame.

Data and checkpoints are intentionally excluded from GitHub. Several preprocessing entry points
under `scripts/preprocessing/` still require local nuPlan dataset, map and output paths.

## Training and evaluation

All commands are run from `nuplan-devkit/MAGNET`; replace `/path/to/...` with local paths.

**1. Splits.** Create log-disjoint train/validation splits, then a frozen test split from logs used
by neither:

```bash
python scripts/training/split_log_disjoint3.py \
  --data_dir /path/to/features --out_dir /path/to/features_split --seed 42
python scripts/training/build_fullret_test_split.py \
  --split_root /path/to/features_split --n_logs 60 --seed 42
```

`build_fullret_test_split.py` also reads a splitter YAML and a log manifest whose default locations
are workstation paths near the top of the script; set them before use.

**2. Agent cache.** Attach surrounding-vehicle targets to each split (required by the joint model):

```bash
python scripts/training/build_agent_aux_cache.py \
  --root /path/to/features_split/train --adjacency_config default
# repeat for val and test
```

**3. Fixed complete topology** (all vehicle-to-vehicle weights equal to one). Train one model per
seed:

```bash
python scripts/training/train_magnet_imitation.py \
  --joint_adaptive --no_learn_topology --num_modes 6 \
  --train_root /path/to/features_split/train \
  --val_root /path/to/features_split/val \
  --output_dir /path/to/output/fixed_seed1 \
  --seed 1 --device cuda:0
```

**4. Counterfactual utility labels.** Compute source-to-target utilities with a trained fixed-topology
model as the teacher, then merge the shards:

```bash
python scripts/analysis/s3_edge_utility_labels.py \
  --teacher /path/to/output/fixed_seed1/models/checkpoint_best.pt \
  --root /path/to/features_split/train --out /path/to/edge_labels/train --device cuda:0
python scripts/analysis/s3_merge_parts.py --root /path/to/edge_labels/train
# repeat for val
```

**5. Learned gate.** Build the per-edge label cache and train with the ranking loss on the edge
scorer:

```bash
python scripts/training/magnet_edge_label_dataset.py \
  --root /path/to/features_split/train --labels /path/to/edge_labels/train
python scripts/training/train_magnet_imitation.py \
  --joint_adaptive --edge_rank_loss --num_modes 6 \
  --train_root /path/to/features_split/train \
  --val_root /path/to/features_split/val \
  --output_dir /path/to/output/learned_seed1 \
  --seed 1 --device cuda:0
```

`scripts/training/train_edge_scorer_s4.py` trains the scorer alone against a frozen teacher; its
checkpoint can initialise the scorer through `--scorer_init`.

**6. Weighting controls.** Build oracle, shuffled or random edge-weight caches from the utility
labels and train with them injected:

```bash
python scripts/training/build_oracle_gate_cache.py \
  --root /path/to/features_split/train --labels /path/to/edge_labels/train --variant oracle
python scripts/training/train_magnet_imitation.py \
  --joint_adaptive --oracle_gate oracle --num_modes 6 \
  --train_root /path/to/features_split/train \
  --val_root /path/to/features_split/val \
  --output_dir /path/to/output/oracle_seed1 \
  --seed 1 --device cuda:0
```

**7. Frozen-test evaluation.**

```bash
python scripts/analysis/eval_joint_test.py \
  --checkpoint /path/to/output/fixed_seed1/models/checkpoint_best.pt \
  --test_root /path/to/features_split/test --out /path/to/eval/fixed_seed1.json
```

Checkpoints, hyperparameters and graph configurations should be selected on training and validation
data only; the frozen test split is for final reporting. Keep `models/train_meta.json` with every
checkpoint so evaluation rebuilds the same model class and options.

## Earlier pipeline

The trainer still exposes the earlier single-output imitation path (`--route_mode`,
`--use_perturbation`, `--gnn_pooling_fix`, `--context_pooling`, `--ego_query_fusion`,
`--ego_shortcut_suppress`), the matched 2×2 runner `scripts/training/run_2x2_ablation.py`, and an
experimental candidate-based planning layer under `src/planning_layer/`. These are retained for
continuity and are not part of the joint-prediction method. The `--multimodal` and `--gnn_operator`
variants of the trainer depend on modules that are not included in this release.

## Verification

Unit tests for the joint model and its data contracts:

```bash
python tests/test_joint_encoder.py
python tests/test_joint_layers.py
python tests/test_joint_loss.py
python tests/test_joint_adaptive.py
python tests/test_edge_scorer_s4.py
python tests/test_oracle_gate.py
python tests/test_agent_slot_join_fullret.py
python tests/test_trajectory_contract_joint.py
python tests/test_s3_edge_utility_labels.py
python devkit_patches/sync_devkit_patches.py --check
```

Several tests load a few validation samples from `data/processed_features_fullret_L1_logsplit/val`
and import `models/transformer_predictor.py` from the parent devkit, so they require a prepared
nuPlan workspace with processed features.

## Closed-loop evaluation

See [`scripts/eval_closed_loop/README.md`](scripts/eval_closed_loop/README.md) for the reconstructed
nuPlan harness, held-out manifests, score extraction and provenance checks. The harness contains
workstation-specific paths that must be configured before use. The joint-prediction manuscript
reports open-loop prediction only.

## Reproducibility boundaries

- No dataset, model weight, generated result or internal experiment record is distributed here.
- The exact upstream nuPlan revision and a complete dependency lock are not published.
- PlanTF source changes, environment patches and trained weights are not included.
- Several configuration, closed-loop and analysis files still contain local absolute paths.
- The repository supports code and protocol inspection; it does not by itself reproduce the paper
  tables without the external nuPlan data, prepared features, checkpoints and environment.
- No software license has been declared in this repository. Permission to reuse the code is not
  granted until a license is added by the authors.
