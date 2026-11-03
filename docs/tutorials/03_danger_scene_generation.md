# Tutorial 03: Danger Scene Generation

> **Last updated**: 2026-06-08 (current main line: `v7 map-conditioned DDPM prior + full guidance + E1-E6 evidence chain`)

---

## Working Directory

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
conda activate nuplan
```

---

## 1. Current Main Line

This research line has moved from "training a model that directly outputs dangerous trajectories" to a clearer two-layer framework:

1. **First train a natural joint future prior**: use a DDPM to learn the real joint future trajectory distribution of `ego + key agent`.
2. **Then perform counterfactual editing at inference time**: use SDEdit and differentiable guidance to push trajectories towards near-misses that are more dangerous but still physically plausible.

As of 2026-06-08, the current main line is not the `v6.0 ego-only guidance diagnosis` of 4 June, but:

- `v7_map_speed`
- `v7_map_intersection`
- `v7_nomap_speed`
- `v7_nomap_intersection`
- `sdedit_with_full_guidance_phase2`
- `compute_ego_svo_torch_timevarying`
- `eval_synergy_report.py`

The current paper narrative in one sentence:

> A dangerous near-miss is not a collision forced into the scene, but an interaction outcome jointly induced by aggressive intent, social-force time-varying SVO, road-level physical constraint and kinematic realism.

---

## 2. Task Definition

Given a real nuPlan danger pair:

- `ego_history`
- `key_agent_history`
- scene graph features
- optional `lane_polylines`
- GT joint future

generate a new `ego + key agent` joint future trajectory such that the result satisfies:

1. It is more dangerous than the baseline future.
2. It preserves road boundaries and kinematic plausibility as far as possible.
3. The change in danger can be explained by the SVO / social-force mechanism.
4. It does not treat "driving straight into a collision" as the only objective, but covers the danger spectrum spanning collision and near-miss.

The currently recommended reporting convention is:

- **full split**: describes the overall data distribution.
- **GT-anchored close subset**: uses `GT min_dist < 8m` to report the samples in which interaction actually occurs.
- **Reason**: the danger pairs selected by TTC include long-distance or weak-interaction samples, and the full split dilutes the guidance effect.

---

## 3. Current Architecture

```text
NuPlan scenes / danger pairs
    │
    ├── ego + key agent histories
    ├── graph / scene features
    └── lane_polylines
            │
            ▼
GNN / Transformer scene encoder
            │
            ├── scene_feat
            └── lane_encoder(lane_polylines)   # map prior optional
            │
            ▼
v7 DDPM prior
            │
            ▼
SDEdit initialization
            │
            ▼
full guidance
    ├── symmetric time-varying SVO
    ├── soft-min proximity
    ├── road-level L_lane
    └── kinematic + lowpass control
            │
            ▼
generated ego-agent joint future
```

Four key current changes:

1. `train_diffusion_gnn_trans.py` supports `--use_map_conditioning`.
2. `LaneEncoder` fuses `lane_polylines` into the diffusion condition in the map models.
3. `sdedit_with_full_guidance_phase2` non-destructively keeps the old v6.0 while adding full guidance.
4. `compute_ego_svo_torch_timevarying` replaces the old whole-horizon averaged pressure with a per-timestep social force.

---

## 4. Data and Representation

### 4.1 Generation Space

The current joint future is 240-dimensional:

```text
x0 = ego_delta_xy[60 * 2] || agent_delta_xy[60 * 2]
```

That is:

- ego future, 60 steps of `(Δx, Δy)`
- key agent future, 60 steps of `(Δx, Δy)`
- the two concatenated into one joint diffusion sample

### 4.2 Danger Pair Data

Two kinds of data are currently in stable use:

| Dataset | Samples | Purpose |
|---|---:|---|
| `danger_pairs_speed` | 533 | High speed / chasing / close-range interaction |
| `danger_pairs_intersection` | 719 | Intersection / complex-geometry interaction |

Note: the danger pair directory must be passed explicitly for training or evaluation.

```text
diffusion_danger_model/dataset/danger_pairs_speed
diffusion_danger_model/dataset/danger_pairs_intersection
```

Do not use the default danger pair path by mistake. The 5 June log confirmed that the default path contains only 2 intersection samples, which easily leads to spurious experiments.

### 4.3 Lane Type Fix

A key data bug was fixed on 5 June:

- The old logic mistakenly treated `pf[:, 9]` as the lane type.
- In fact, `pf[:, 9]` is `subgraph_id`.
- The road / lane semantics should now be recovered via the `SUBGRAPH_ID_TO_TYPE` mapping.

This fix directly affects the credibility of the map-conditioned prior and the road guidance.

---

## 5. v7 Training Status

v7 currently has four complete checkpoints:

| checkpoint | Dataset | map conditioning | epoch100 loss | Status |
|---|---|---:|---:|---|
| `v7_map_speed` | speed | yes | 0.226 | Done |
| `v7_map_intersection` | intersection | yes | 0.208 | Done |
| `v7_nomap_speed` | speed | no | 0.213 | Done |
| `v7_nomap_intersection` | intersection | no | 0.233 | Done |

Training time has been confirmed to be much shorter than originally estimated:

- GPU training takes about 23 seconds / epoch.
- 100 epochs take about 38-40 minutes.
- One OOM occurred due to host RAM / DataLoader concurrency; `v7_map_intersection` was later rerun on its own and has completed.

The conclusion on the map prior cannot simply be stated as "always better":

- Intersection geometry is complex, and the map prior clearly reduces baseline offroad.
- Speed scenes have simpler geometry, and the map prior is redundant or even slightly negative.

The paper's position should therefore be:

> The map prior is an enabling component under complex geometry, not a standalone source of danger.

---

## 6. SVO and the Social Force Mechanism

The core current contribution is **time-varying SVO**.

The problem with the old SVO version was that the social force was collapsed into a whole-horizon average, which could not explain dynamic behaviour such as "no need to yield at long distance, sudden yielding at close range".

The new version is implemented in `src/shared/utils/svo_torch.py`:

```text
force[t] = relu(8 - dist[t]) / 8
```

Intuitive explanation:

- When the two vehicles are far apart, the social force is approximately 0.
- When the two vehicles come within 8m, the social force rises.
- The ego SVO becomes more prosocial under close-range pressure.
- This forms a near-miss mechanism of "aggressive approach first, near-zone avoidance afterwards".

The 7 June S2 mechanism validation gave concrete samples:

| Minimum distance | far SVO | near SVO | Explanation |
|---:|---:|---:|---|
| 0.16m | approx. 1-2.6° | 4.10° | Starts yielding at close range |
| 0.19m | approx. 1-2.6° | 5.43° | Social force rises |
| 0.42m | approx. 1-2.6° | 7.58° | Near-miss mechanism clearly visible |

The aggressive target currently uses a fixed initial commitment:

```text
target = baseline_svo - relative_delta_deg
relative_delta_deg = 15
```

This corresponds to aggressive intent in the paper narrative.

---

## 7. Inference: SDEdit + Full Guidance

The current main entry point is:

```text
scripts/eval_v5_svo.py
mode = full
```

The corresponding internal implementation is:

```text
sdedit_with_full_guidance_phase2
```

Full guidance contains four kinds of constraint:

| Component | Role |
|---|---|
| symmetric time-varying SVO | Both ego and agent enter the interaction constraint, instead of pushing only ego |
| soft-min proximity | Encourages controllable close-range interaction |
| road-level `L_lane` | Constrains road/on-road, no longer misclassifying normal lane changes as failures |
| kinematic + lowpass | Controls acceleration and trajectory smoothness |

The physical operating point selected on 7 June:

```text
lambda_kinematic = 10.0
lowpass_cutoff_ratio = 0.15
```

Effect of this operating point:

- Reduces mean acceleration from about 12 to about 9.3, below 1g.
- Danger barely decreases.
- Collision / near-miss intensity remains close to the GT danger spectrum.

Note: jerk is currently unsuitable as the main realism metric. GT trajectories already have high jerk because of finite-difference noise at `dt=0.1`. The main reported metric should currently prioritise accel.

---

## 8. Evaluation Protocol

### 8.1 Current Scripts

| Script | Purpose |
|---|---|
| `scripts/eval_v5_svo.py` | Evaluation entry point for baseline / full guidance |
| `scripts/analysis/eval_synergy_report.py` | Outputs full split and close subset metrics for GT, baseline and full in a unified way |
| `scripts/analysis/viz_near_miss.py` | Generates near-miss mechanism visualisations |

### 8.2 Current Main Metrics

| Metric | Purpose |
|---|---|
| `danger` | Main metric of danger intensity |
| `collision_rate` | The more severe end of the danger spectrum |
| `min_distance` | Whether close-range interaction forms |
| `offroad` | Road keeping |
| `accel` | Main metric of physical plausibility |
| `ego_svo` | Behavioural semantic explanation |

### 8.3 Offroad Convention

Both conventions need to be stated:

- `offroad @1.75m`: strict lane-level deviation, which easily counts normal lane changes as failures.
- `offroad @3m`: closer to the current road-level soft boundary design.

The current method targets road keeping, not prohibiting lane changes.

---

## 9. Current Key Results

### 9.1 S2 Speed Split: Is Full Guidance Effective

`v7_map_speed`, B=63, road 3m + time-varying SVO + proximity + kinematic.

| Method | danger | offroad | egoSVO | minDist | collide |
|---|---:|---:|---:|---:|---:|
| GT | 2.96 | 0.0% | 11.8° | 21.5m | 27.0% |
| baseline | 0.80 | 53.6% | 8.2° | 23.9m | 3.2% |
| full(timevar) | 2.19 | 8.0% | 13.5° | 21.2m | 19.0% |

Conclusions:

- Danger rises clearly.
- Offroad drops substantially.
- Collision rate moves towards the GT danger spectrum.
- Time-varying SVO can explain the change in near-miss behaviour.

### 9.2 Intersection Close Subset: Does It Hold Under Complex Geometry

| Method | danger | offroad | egoSVO | minDist | collide |
|---|---:|---:|---:|---:|---:|
| GT | 3.097 | 1.8% | 5.7° | 3.68m | 40.5% |
| baseline | 1.763 | 35.2% | 5.2° | 6.18m | 25.7% |
| full(timevar) | 2.407 | 8.7% | 10.7° | 5.33m | 35.1% |

Conclusions:

- In the close subset, full guidance is effective for both speed and intersection.
- In the full split the effect is diluted by long-distance or weak-interaction samples, so the paper needs to explain why the close subset is necessary.

### 9.3 E2 Map Prior Conclusions

Baseline offroad comparison:

| Dataset | map | nomap | Conclusion |
|---|---:|---:|---|
| speed all | 53.4% | 49.1% | map slightly worse |
| speed close | 58.0% | 48.3% | map slightly worse |
| intersection all | 40.8% | 56.1% | map clearly better |
| intersection close | 37.8% | 55.5% | map clearly better |

Conclusions:

- The map prior has significant value for intersection.
- The map prior is redundant or slightly negative for speed.
- Once road guidance is added, offroad decreases for both map and nomap, so the marginal value of the map is masked by guidance.

### 9.4 E5 Statistical Significance

Both core claims now have statistical support:

| claim | Result | Statistics |
|---|---|---|
| full guidance increases danger | baseline 1.648 -> full 5.136 | Wilcoxon p=1.74e-09, bootstrap diff 95% CI [2.556, 4.323] |
| map prior reduces intersection baseline offroad | map median 33.3% vs nomap median 65.0% | Wilcoxon p=2.53e-09, median -31.7pt |

---

## 10. Current Paper Claims

The most robust current paper structure is:

1. **Method claim**: use a map-conditioned DDPM prior to learn the real joint trajectory distribution.
2. **Mechanism claim**: use social-force time-varying SVO to express "aggressive approach + near-zone avoidance".
3. **Inference claim**: use full guidance to generate dangerous near-misses under physical constraints.
4. **Experimental evidence**: full guidance significantly increases danger, and the map prior significantly improves intersection offroad.

More specifically:

- Danger does not come from the map.
- Danger comes mainly from SVO / social-force guidance.
- The map is an enabling component that reduces offroad under complex road geometry.
- Kinematic / lowpass is the constraint layer that makes trajectories physically plausible.

---

## 11. Known Issues and TODO List

TODO List:

- Quantify the proportion of degenerate danger pairs, especially samples where "the min gap occurs only at t=0" or "ego is almost stationary".
- Check and fix the issue in `ttc_filter` where ego speed uses the instantaneous speed instead of the historical mean.
- Design stronger danger pair selection criteria: a genuine approach process, mid-trajectory min distance, both parties moving.
- Decide whether the danger pairs need to be reselected and the v7 prior retrained.
- Report both `@1.75m` and `@3m` when reporting offroad, to avoid misinterpreting normal lane changes as road failures.
- Continue addressing the long-tail physical plausibility issue of accel p95 at about 21.
- Organise the E1-E6 evidence chain into the experiments section of the paper.

---

## 12. Quick Entry Points

### 12.1 Training Entry Point

```text
train_diffusion_gnn_trans.py
--use_map_conditioning
--danger_pair_dir diffusion_danger_model/dataset/danger_pairs_speed
--danger_pair_dir diffusion_danger_model/dataset/danger_pairs_intersection
```

The current training main line is the four v7 checkpoints. Before continuing experiments, first confirm that the checkpoint matches the danger pair directory.

### 12.2 Evaluation Entry Points

```text
scripts/eval_v5_svo.py
scripts/analysis/eval_synergy_report.py
```

It is recommended to use `eval_synergy_report.py` to generate, in a unified way:

- GT
- baseline
- full guidance
- full split
- close subset
- JSON results

### 12.3 Visualisation Entry Points

```text
scripts/analysis/viz_near_miss.py
docs/figures/near_miss/
```

The current visualisation already supports representative speed / intersection near-misses, used to show:

- aggressive approach
- distance V-shape
- near-zone SVO rise
- late avoidance

---

## 13. One-Sentence Summary

> The current question is no longer v6.0's "can ego-only guidance still be tuned to work", but that v7 prior + full guidance already forms an evidence chain that can support a paper; the next priorities are cleaning up danger pair quality, unifying the offroad/accel conventions, and writing E1-E6 into the method and experiments sections.
