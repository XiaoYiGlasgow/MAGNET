# MAGNET

MAGNET is research code for trajectory prediction and nuPlan planning experiments. The current
public mainline supports graph-based scene encoding, Transformer trajectory prediction, controlled
route-feature injection, ego-state perturbation, candidate-based planning refinement, and matched
analysis workflows for the VT-2026-04191 resubmission.

This repository publishes source code and small provenance manifests only. nuPlan data, processed
features, checkpoints, experiment outputs, and internal work records are not included.

This is an active research snapshot rather than a packaged Python library. The public tree is meant
for source and protocol inspection; it does not provide a one-command reproduction of manuscript
tables.

## What is in this release

- MAGNET imitation training with weighted ADE/FDE, smoothness, and direction losses.
- Two route modes: `none` and `three_channel`.
- Train-only ego perturbation with deterministic seeds.
- A controlled B/R/P/RP factorial runner for route and perturbation effects.
- Log-disjoint train/validation/test split and independent-test utilities, including an option to
  consume a shared splitter YAML for paired comparisons.
- A per-sample GNN pooling correction that is opt-in through `--gnn_pooling_fix`.
- Flag-gated experimental variants for context polyline pooling, ego-conditioned attention pooling,
  and ego-history shortcut suppression. Their implementations and unit tests are included, but their
  presence is not a performance or novelty claim.
- An experimental candidate-based planning layer with kinematic, collision, rule, comfort, and
  progress terms; its identity candidate and component switches support controlled ablations.
- Analysis utilities for paired bootstrap inference, interaction-reliance audits, scaling studies,
  checkpoint calibration, and closed-loop readout. These are protocols and tools, not published
  performance claims.
- Vendored snapshots of four modified nuPlan integration files, including MAGNET and PlanTF planner
  configuration.
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
├── src/                           # shared utilities, planning layer, and danger generation
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

The development environment uses Python 3.9 and a locally prepared nuPlan devkit checkout. The core
training path imports PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, and PyYAML. Analysis,
visualization, and tests additionally use packages including SciPy, Matplotlib, Shapely, tqdm, and
pytest.

This repository does not include a lockfile and does not yet pin the upstream nuPlan commit. Follow
the [official nuPlan setup](https://github.com/motional/nuplan-devkit) first, then record the exact
nuPlan, Python, CUDA, PyTorch, and PyG versions used for an experiment. Cloning the current upstream
default branch is not a guarantee of bitwise compatibility with the development environment.

```bash
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .

git clone https://github.com/XiaoYiGlasgow/MAGNET.git
cd MAGNET
```

## nuPlan integration files

Four integration snapshots used by the development workspace are versioned under
`devkit_patches/files/`. Inspect them before writing them into a nuPlan checkout.

```bash
# From nuplan-devkit/MAGNET
python devkit_patches/sync_devkit_patches.py --check

# After reviewing and backing up the target files, install the vendored snapshots.
python devkit_patches/sync_devkit_patches.py --push
python devkit_patches/sync_devkit_patches.py --check
```

On a fresh upstream checkout, the first `--check` may report missing planner files; this is expected
and shows what `--push` would create. `--push` writes into the parent `nuplan-devkit` tree and may
overwrite existing files. It refuses to overwrite the frozen `models/transformer_predictor.py` when
an existing copy differs from the vendored snapshot.

The vendored `plantf_planner.yaml` is only a nuPlan registration file. The external
[PlanTF repository](https://github.com/jchengai/planTF), its environment, weights, and the local
MAGNET-to-PlanTF hook are not vendored here; the PlanTF path is therefore an optional integration
boundary, not a standalone reproduction supplied by this repository.

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

The split commands below expect `features_*.pkl` files and a sibling `log_manifest.json` generated
by the preprocessing pipeline. The relevant entry points are under `scripts/preprocessing/`; several
still require local nuPlan dataset, map, and output paths to be configured.

## Training

Create log-disjoint splits before training:

```bash
python scripts/training/split_log_disjoint3.py \
  --data_dir /path/to/features \
  --out_dir /path/to/features_split3 \
  --seed 42
```

For a paired comparison that must reuse an externally defined log partition, use the shared YAML
mode instead of drawing another split:

```bash
python scripts/training/split_log_disjoint.py \
  --data_dir /path/to/features \
  --manifest /path/to/log_manifest.json \
  --out_dir /path/to/features_logsplit \
  --splitter_yaml /path/to/shared_splitter.yaml
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

### Experimental architecture flags

The main trainer also exposes three default-off research variants:

| Flag | Implementation | Contract |
|---|---|---|
| `--context_pooling` | `scripts/training/magnet_context_pooled.py` | Pools segment tokens by polyline before context attention; produces a checkpoint incompatible with the baseline architecture. |
| `--ego_query_fusion` | `scripts/training/magnet_ego_query_fusion.py` | Replaces fixed mean pooling with ego-conditioned attention; use with `--dataset_config nearby_only` as required by the trainer. |
| `--ego_shortcut_suppress` | `scripts/training/magnet_ego_shortcut.py` | Replaces the ego-history path with a current-state encoder and train-time state dropout; treated as an adapted comparison, not a MAGNET novelty claim. |

These flags may be combined only according to the trainer's logged architecture contract. Always
retain `models/train_meta.json` with a checkpoint so inference and calibration rebuild the same
model class and options.

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

## Experimental planning layer

`src/planning_layer/` provides a default-off candidate generator and scorer. The available presets
are `off`, `full`, `no_dynamics`, `no_collision`, `no_rules`, and `no_progress`. MAGNET's vendored
nuPlan planner accepts the preset through its `planning_layer` configuration field or the
`MAGNET_PLANNING_LAYER` environment variable.

The layer is an experimental evaluation component. Its presence in the public tree establishes an
inspectable implementation and test contract, not evidence of improved closed-loop safety or planner
competitiveness.

## Analysis entry points

The latest protocol-oriented utilities include:

| Purpose | Entry point |
|---|---|
| Paired factorial and bootstrap inference | `scripts/analysis/agg_2x2.py`, `scripts/analysis/c4_paired_bootstrap.py` |
| Interaction-reliance intervention audit | `scripts/analysis/run_interaction_audit.py`, `scripts/analysis/agg_interaction_audit.py` |
| Scaling and power analysis | `scripts/analysis/s3_scaling_power_analysis.py`, `scripts/analysis/s6_alpha_estimate.py` |
| Checkpoint calibration and architecture reconstruction | `scripts/analysis/bn_calibrate_checkpoint.py`, `src/shared/utils/bn_calibration.py` |

These scripts require user-supplied checkpoints, features, manifests, or result directories. None of
those artifacts are distributed in this repository.

## Verification

The lightweight checks included in the public tree are:

```bash
python scripts/training/tests/test_perturbation.py
python scripts/training/tests/test_gnn_pooling.py
python scripts/preprocessing/tests/test_route_lane_alignment.py
python scripts/training/tests/test_splitter_yaml_split.py
python -m pytest -q scripts/training/tests/test_context_pooling.py
python -m pytest -q scripts/training/tests/test_ego_query_fusion.py
python -m pytest -q scripts/training/tests/test_ego_state_dropout.py
python -m pytest -q src/planning_layer/tests
python -m pytest -q scripts/eval_closed_loop/tests
python scripts/analysis/result_manifest.py --self_test
python devkit_patches/sync_devkit_patches.py --check
```

Some checks skip integration cases when the corresponding external nuPlan data, modified PlanTF
checkout, or prepared experiment artifacts are unavailable. The GNN pooling test, route-alignment
test, result-manifest self-test, and devkit sync check require the repository to be located inside a
prepared nuPlan workspace. The route-alignment test also reads nuPlan scenarios and is not a purely
synthetic unit test.

## Closed-loop evaluation

See [`scripts/eval_closed_loop/README.md`](scripts/eval_closed_loop/README.md) for the reconstructed
nuPlan harness, held-out manifests, score extraction, and provenance checks.

The harness currently contains workstation-specific paths that must be configured before use.
Historical Boston scores mentioned in its documentation do not have preserved simulation artifacts
and must not be described as reproduced by this release.

## Reproducibility boundaries

- No dataset, model weight, generated result, or internal experiment record is distributed here.
- The exact upstream nuPlan revision and a complete dependency lock are not yet published.
- PlanTF source changes, environment patches, and trained weights are not included.
- Several legacy configuration and closed-loop files still contain local absolute paths.
- The public repository supports code and protocol inspection; it does not by itself reproduce the
  paper tables without the external nuPlan data, prepared features, checkpoints, and environment.
- No software license has been declared in this repository. Permission to reuse the code is not
  granted until a license is added by the authors.
