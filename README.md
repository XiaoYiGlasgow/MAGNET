# MAGNET

MAGNET is research code for trajectory prediction and nuPlan planning experiments. The current
public mainline supports graph-based scene encoding, Transformer trajectory prediction, controlled
route-feature injection, ego-state perturbation, and a matched 2×2 ablation workflow for the
VT-2026-04191 resubmission.

This repository publishes source code and small provenance manifests only. nuPlan data, processed
features, checkpoints, experiment outputs, and internal work records are not included.

## What is in this release

- MAGNET imitation training with weighted ADE/FDE, smoothness, and direction losses.
- Two route modes: `none` and `three_channel`.
- Train-only ego perturbation with deterministic seeds.
- A controlled B/R/P/RP factorial runner for route and perturbation effects.
- Log-disjoint train/validation/test split and independent-test utilities.
- A per-sample GNN pooling correction that is opt-in through `--gnn_pooling_fix`.
- Vendored snapshots of the three modified nuPlan integration files.
- A reconstructed closed-loop evaluation harness with provenance checks.

The repository also retains danger-generation utilities and pre-v8 code for continuity. Files under
`archive/` are historical and are not the active MAGNET training path.

## Repository layout

```text
MAGNET/
├── configs/                       # nuPlan and Hydra configuration
├── devkit_patches/                # vendored planner/model integration files
├── docs/                          # schema, tutorials, and technical notes
├── scripts/
│   ├── analysis/                  # result manifests and ablation analysis
│   ├── eval_closed_loop/          # reconstructed closed-loop harness
│   ├── preprocessing/             # feature generation and route preprocessing
│   └── training/                  # MAGNET training, splitting, tests, and 2×2 runner
├── src/                           # shared, planning, and danger-generation modules
├── archive/                       # legacy pre-v8 source code
├── config.py
└── train_diffusion_gnn_trans.py   # retained danger-generation training entrypoint
```

## Requirements

MAGNET is designed to be cloned directly inside a `nuplan-devkit` checkout:

```text
nuplan-devkit/
├── MAGNET/
├── models/
└── nuplan/
```

The development environment uses Python 3.9, PyTorch, PyTorch Geometric, NumPy, pandas,
scikit-learn, and the nuPlan devkit. This release does not yet include a lockfile, so reproduce the
exact package versions in your own environment record.

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .

git clone https://github.com/XiaoYiGlasgow/MAGNET.git
cd MAGNET
```

## nuPlan integration files

Three modified files required by training and online planning are versioned under
`devkit_patches/files/`. Inspect them before writing them into a nuPlan checkout.

```bash
# From nuplan-devkit/MAGNET
python devkit_patches/sync_devkit_patches.py --push
python devkit_patches/sync_devkit_patches.py --check
```

`--push` writes into the parent `nuplan-devkit` tree. It refuses to overwrite the frozen
`models/transformer_predictor.py` when an existing copy differs from the vendored snapshot.

## Data contract

The main imitation pipeline consumes precomputed `GraphDataset` feature files. The frozen v2 schema
is documented in [`docs/feature_schema_v2.md`](docs/feature_schema_v2.md); its executable definition
is `FEATURE_SCHEMA_V2` in `scripts/analysis/result_manifest.py`.

Important boundaries:

- Features have shape `(N, 12)`.
- Ego history is 20 frames (2 s at 10 Hz), not 4 s.
- The target is a 60-frame local incremental trajectory (6 s at 10 Hz).
- Route mode `three_channel` reuses lane columns 6–8; it is not a separate route branch.
- `route_mode` used for preprocessing and training must match.

Data and checkpoints are intentionally excluded from GitHub.

## Training

Create log-disjoint splits before training:

```bash
python scripts/training/split_log_disjoint3.py \
  --data_dir /path/to/features \
  --out_dir /path/to/features_split3 \
  --seed 42
```

Train one MAGNET configuration:

```bash
python scripts/training/train_magnet_imitation.py \
  --train_root /path/to/features_split3/train \
  --val_root /path/to/features_split3/val \
  --output_dir /path/to/output \
  --route_mode none \
  --seed 42 \
  --device cuda:0
```

Optional flags include `--use_perturbation` and `--gnn_pooling_fix`. The pooling correction is off
by default to preserve the legacy execution path.

Preview the matched 2×2 experiment without starting training:

```bash
python scripts/training/run_2x2_ablation.py \
  --stage smoke \
  --none_root_base /path/to/no_route_split3 \
  --route_root_base /path/to/route_v2_split3 \
  --out_base /path/to/ablation_outputs \
  --device cuda:0 \
  --dry_run
```

The four cells are B (`none`, no perturbation), R (`three_channel`, no perturbation), P (`none`,
perturbation), and RP (`three_channel`, perturbation). The runner checks that both data variants use
the same log split before executing a non-dry run.

## Verification

The lightweight checks included in the public tree are:

```bash
python scripts/training/tests/test_perturbation.py
python scripts/training/tests/test_gnn_pooling.py
python scripts/preprocessing/tests/test_route_lane_alignment.py
python scripts/analysis/result_manifest.py --self_test
python devkit_patches/sync_devkit_patches.py --check
```

The GNN pooling test and devkit sync check require the repository to be located inside the prepared
nuPlan workspace described above.

## Closed-loop evaluation

See [`scripts/eval_closed_loop/README.md`](scripts/eval_closed_loop/README.md) for the reconstructed
nuPlan harness, held-out manifests, score extraction, and provenance checks.

The harness currently contains workstation-specific paths that must be configured before use.
Historical Boston scores mentioned in its documentation do not have preserved simulation artifacts
and must not be described as reproduced by this release.

## Reproducibility boundaries

- No dataset, model weight, generated result, or internal experiment record is distributed here.
- Several legacy configuration and closed-loop files still contain local absolute paths.
- The public repository supports code and protocol inspection; it does not by itself reproduce the
  paper tables without the external nuPlan data, prepared features, checkpoints, and environment.
- No software license has been declared in this repository. Permission to reuse the code is not
  granted until a license is added by the authors.
