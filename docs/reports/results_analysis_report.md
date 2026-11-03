# GNN_Trans results Directory Experimental Results Analysis Report

> Applicable project: `Project A (trajectory prediction / baseline / planning)`
>
> Role: interim full report. It retains the complete statistical process and is not intended as the main entry point for first-time readers.

Analysis scope: `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/results*`
Generated on: 2026-04-28
Output location: `GNN_Trans/docs/reports/results_analysis_report.md`

---

## 1. Data Sources and Statistical Basis

This report scans all experimental result directories whose names begin with `results` under the `GNN_Trans` root directory. A total of **20 top-level result directories** were found, covering the following experimental matrix:

- Scenario types: `high_lateral_acceleration`, `high_magnitude_speed`, `medium_magnitude_speed`, `near_multiple_vehicles`, `traversing_intersection`
- Adjacency configurations: `default`, `ego_only`, `nearby_only`, `no_ego_road`

Each training directory mainly contains:

- `training_progress.json`: summary of the last epoch
- `results/loss_record_*.json`: full training curves, final ADE/FDE, best ADE/FDE, best validation loss, etc.
- `results/loss_components_*.json`: loss components
- `models/best_model_*.pth` and `models/final_model_*.pth`
- `logs/training_log.txt`

This report preferentially uses the fields in `loss_record_*.json`, because it contains the full curves and final statistics. The main metrics in the report are defined as follows:

| Metric | Meaning |
|------|------|
| `final_ade` | ADE of the last epoch |
| `final_fde` | FDE of the last epoch |
| `best_ade` | Lowest ADE recorded during training |
| `best_fde` | Lowest FDE recorded during training |
| `best_val_loss` | Lowest validation loss during training |
| `best_val_epoch` | Epoch corresponding to `best_val_loss` |

Note: `best_val_loss` and the lowest ADE/FDE do not necessarily occur at the same epoch, because the training loss is a weighted combination of ADE, FDE, smoothness and direction, with the following weights:

| Loss term | Weight |
|---------|------|
| ADE | 0.5 |
| FDE | 3.0 |
| Smoothness | 0.3 |
| Direction | 0.5 |

---

## 2. Overview of Experiment Directories

A total of **22 training records** were parsed. The 2 extra records come from `high_magnitude_speed/default`, for which three historical training versions, `ds900`, `ds450` and `ds4500`, were retained.

| Scenario type | Adjacency configuration | Training samples | Epoch | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Loss | Training directory |
|----------|----------|------------|-------|----------|----------|----------|----------|---------------|----------|
| high_lateral_acceleration | default | 13500 | 100 | 2.180 | 4.317 | 2.134 | 4.200 | 0.3462 | `training_20250930_143101_ds13500_bs8_lr5e-04` |
| high_lateral_acceleration | ego_only | 13500 | 100 | 2.063 | 3.870 | 2.059 | 3.870 | 0.3009 | `training_20250930_143124_ds13500_bs8_lr5e-04` |
| high_lateral_acceleration | nearby_only | 13500 | 100 | 1.647 | 3.440 | 1.581 | 3.306 | 0.3176 | `training_20251001_143320_ds13500_bs8_lr5e-04` |
| high_lateral_acceleration | no_ego_road | 13500 | 93 | 2.369 | 4.641 | 2.337 | 4.566 | 0.3673 | `training_20250930_155757_ds13500_bs8_lr5e-04` |
| high_magnitude_speed | default | 900 | 42 | 3.386 | 7.234 | 3.093 | 6.888 | 0.5793 | `training_20250922_173700_ds900_bs8_lr5e-04` |
| high_magnitude_speed | default | 450 | 64 | 3.373 | 6.901 | 2.529 | 5.600 | 0.5343 | `training_20250923_165322_ds450_bs8_lr5e-04` |
| high_magnitude_speed | default | 4500 | 30 | 3.717 | 7.583 | 2.588 | 5.394 | 0.4664 | `training_20250923_223705_ds4500_bs8_lr5e-04` |
| high_magnitude_speed | ego_only | 4500 | 41 | 3.671 | 7.700 | 3.006 | 6.379 | 0.5939 | `training_20250923_224659_ds4500_bs8_lr5e-04` |
| high_magnitude_speed | nearby_only | 4500 | 40 | 2.831 | 5.917 | 2.526 | 5.479 | 0.4494 | `training_20250924_020904_ds4500_bs8_lr5e-04` |
| high_magnitude_speed | no_ego_road | 4500 | 60 | 3.318 | 6.733 | 2.922 | 5.995 | 0.4468 | `training_20250924_020806_ds4500_bs8_lr5e-04` |
| medium_magnitude_speed | default | 18000 | 68 | 2.739 | 5.396 | 2.712 | 5.326 | 0.4881 | `training_20251001_154145_ds18000_bs8_lr5e-04` |
| medium_magnitude_speed | ego_only | 18000 | 81 | 2.734 | 5.413 | 2.684 | 5.308 | 0.4840 | `training_20251001_154203_ds18000_bs8_lr5e-04` |
| medium_magnitude_speed | nearby_only | 18000 | 69 | 1.447 | 3.694 | 1.345 | 3.506 | 0.4541 | `training_20251002_124458_ds18000_bs8_lr5e-04` |
| medium_magnitude_speed | no_ego_road | 18000 | 91 | 1.925 | 4.350 | 1.725 | 4.095 | 0.4415 | `training_20251002_124442_ds18000_bs8_lr5e-04` |
| near_multiple_vehicles | default | 27000 | 100 | 1.552 | 2.871 | 1.514 | 2.767 | 0.2395 | `training_20250925_170041_ds27000_bs8_lr5e-04` |
| near_multiple_vehicles | ego_only | 27000 | 100 | 1.398 | 2.445 | 1.380 | 2.427 | 0.2197 | `training_20250925_170040_ds27000_bs8_lr5e-04` |
| near_multiple_vehicles | nearby_only | 27000 | 98 | 1.838 | 3.443 | 1.795 | 3.349 | 0.3137 | `training_20250927_141105_ds27000_bs8_lr5e-04` |
| near_multiple_vehicles | no_ego_road | 27000 | 100 | 1.623 | 2.945 | 1.623 | 2.945 | 0.2403 | `training_20250925_170041_ds27000_bs8_lr5e-04` |
| traversing_intersection | default | 4500 | 55 | 2.508 | 5.648 | 2.453 | 5.473 | 0.5064 | `training_20250924_170750_ds4500_bs8_lr5e-04` |
| traversing_intersection | ego_only | 4500 | 80 | 2.603 | 5.969 | 2.504 | 5.712 | 0.4967 | `training_20250924_170810_ds4500_bs8_lr5e-04` |
| traversing_intersection | nearby_only | 4500 | 63 | 3.353 | 7.190 | 3.178 | 6.678 | 0.5460 | `training_20250925_022650_ds4500_bs8_lr5e-04` |
| traversing_intersection | no_ego_road | 4500 | 57 | 3.169 | 6.778 | 3.057 | 6.401 | 0.5249 | `training_20250925_022551_ds4500_bs8_lr5e-04` |

---

## 3. Main Comparison Table: One Record per Configuration, Selected by Largest Sample Size

For a fair side-by-side comparison, this section keeps only one main record for each "scenario type + adjacency configuration" pair. The selection rule is: prefer the record with the largest number of training samples; if the sample counts are equal, choose the more recent record.

This fixes the main record of `high_magnitude_speed/default` as the `ds4500` version, rather than `ds900` or `ds450`.

| Scenario type | default ADE/FDE | ego_only ADE/FDE | nearby_only ADE/FDE | no_ego_road ADE/FDE | Best final ADE/FDE |
|----------|-----------------|------------------|---------------------|---------------------|------------------|
| high_lateral_acceleration | 2.180 / 4.317 | 2.063 / 3.870 | **1.647 / 3.440** | 2.369 / 4.641 | nearby_only |
| high_magnitude_speed | 3.717 / 7.583 | 3.671 / 7.700 | **2.831 / 5.917** | 3.318 / 6.733 | nearby_only |
| medium_magnitude_speed | 2.739 / 5.396 | 2.734 / 5.413 | **1.447 / 3.694** | 1.925 / 4.350 | nearby_only |
| near_multiple_vehicles | 1.552 / 2.871 | **1.398 / 2.445** | 1.838 / 3.443 | 1.623 / 2.945 | ego_only |
| traversing_intersection | **2.508 / 5.648** | 2.603 / 5.969 | 3.353 / 7.190 | 3.169 / 6.778 | default |

**Direct observations**

1. `nearby_only` wins 3 of the 5 scenarios: `high_lateral_acceleration`, `high_magnitude_speed` and `medium_magnitude_speed`.
2. `ego_only` is strongest on `near_multiple_vehicles`, indicating that in this scenario the ego vehicle's own historical motion already provides a very strong prediction signal.
3. `default` is strongest on `traversing_intersection`, indicating that intersection scenarios may rely more on the full graph structure rather than only on nearby agents or ego information.
4. `no_ego_road` does not rank first on final ADE/FDE in any scenario, but it has the lowest validation loss on `medium_magnitude_speed`, indicating that the loss and the ADE/FDE metrics are not fully consistent.

---

## 4. Overall Comparison of Adjacency Configurations

Averaged over the 5 main scenarios:

| Adjacency configuration | Mean final ADE | Mean final FDE | Rank |
|----------|--------------|--------------|------|
| nearby_only | **2.223** | **4.737** | 1 |
| no_ego_road | 2.481 | 5.089 | 2 |
| ego_only | 2.494 | 5.079 | 3 |
| default | 2.539 | 5.163 | 4 |

**Interpretation**

`nearby_only` is currently the most stable adjacency configuration overall, with the lowest mean ADE and FDE. This result is somewhat interesting: the full `default` graph does not win consistently, which suggests that the current model may not yet make full use of the complex graph structure, or that the full graph introduces additional noise.

The mean results of `ego_only` and `no_ego_road` are very close. `ego_only` performs best in the multi-vehicle proximity scenario, but has no advantage in the high-speed and intersection scenarios; `no_ego_road` is relatively strong in the medium-speed scenario, but does not rank first on the final metrics in any scenario.

`default` ranks low on average, but it is best on `traversing_intersection`, which is important. It cannot simply be said that default is useless; a more reasonable judgement is that the full graph structure helps in certain scenarios, but the current training or feature-fusion approach does not allow it to benefit consistently across all scenarios.

---

## 5. Scenario Difficulty Comparison

Averaging the final metrics of the 4 adjacency configurations within each scenario:

| Scenario type | Mean final ADE | Mean final FDE | Difficulty assessment |
|----------|--------------|--------------|----------|
| near_multiple_vehicles | **1.603** | **2.926** | Easiest |
| high_lateral_acceleration | 2.065 | 4.067 | Relatively easy |
| medium_magnitude_speed | 2.211 | 4.713 | Medium |
| traversing_intersection | 2.908 | 6.396 | Relatively hard |
| high_magnitude_speed | **3.384** | **6.983** | Hardest |

**Interpretation**

Although `near_multiple_vehicles` sounds complex, it is the easiest scenario in these results; in particular, `ego_only` reaches `ADE 1.398 / FDE 2.445`. This may indicate that the near multiple vehicles scenarios in this dataset do not necessarily contain strongly interactive future changes, or that the ego trajectory inertia is very stable.

`high_magnitude_speed` is the hardest scenario. Even the best configuration, `nearby_only`, only reaches `ADE 2.831 / FDE 5.917`, and the other configurations are generally at `FDE 6.7-7.7m`. In high-speed scenarios, speed estimation, long-horizon integration error and terminal deviation are all amplified.

`traversing_intersection` is the second-hardest scenario, and `nearby_only` and `no_ego_road` degrade markedly. Intersection scenarios are more sensitive to road topology and full context, so `default` instead becomes the best configuration.

---

## 6. Detailed Analysis by Scenario

### 6.1 high_lateral_acceleration

| Adjacency configuration | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 2.180 | 4.317 | 2.134 | 4.200 | 89 |
| ego_only | 2.063 | 3.870 | 2.059 | 3.870 | 100 |
| nearby_only | **1.647** | **3.440** | **1.581** | **3.306** | 95 |
| no_ego_road | 2.369 | 4.641 | 2.337 | 4.566 | 93 |

`nearby_only` is clearly the strongest, indicating that in high-lateral-acceleration scenarios, nearby dynamic objects or local interaction information are more valuable for prediction. `no_ego_road` is the weakest, possibly because removing the ego-road relation loses key road constraints.

It is worth noting that `ego_only` has `best_val_epoch=100`, and its final and best metrics almost coincide, indicating that this configuration is still converging stably without obvious overfitting.

### 6.2 high_magnitude_speed

| Adjacency configuration | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 3.717 | 7.583 | 2.588 | **5.394** | 15 |
| ego_only | 3.671 | 7.700 | 3.006 | 6.379 | 26 |
| nearby_only | **2.831** | **5.917** | **2.526** | 5.479 | 25 |
| no_ego_road | 3.318 | 6.733 | 2.922 | 5.995 | 45 |

This is the hardest scenario overall. `nearby_only` has the best final metrics, but `default` reached a very good best FDE (5.394) during training, only to regress markedly to 7.583 at the final epoch.

This indicates that the `ds4500` version of `high_magnitude_speed/default` shows clear late-stage degradation: the best validation point occurs at epoch 15, while the metrics have worsened by the time training ends at epoch 30. Therefore, when using this experiment, `best_model_default.pth` should be preferred, and the final model should not be used directly.

In addition, `high_magnitude_speed/default` retains three historical scales:

| Samples | Epoch | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Loss |
|--------|-------|----------|----------|----------|----------|---------------|
| 450 | 64 | 3.373 | 6.901 | 2.529 | 5.600 | 0.5343 |
| 900 | 42 | 3.386 | 7.234 | 3.093 | 6.888 | 0.5793 |
| 4500 | 30 | 3.717 | 7.583 | 2.588 | 5.394 | 0.4664 |

`ds4500` has a better best validation loss and best FDE, but worse final metrics. This suggests that the second half of training may need stronger early stopping, or a lower learning rate / more validation monitoring.

### 6.3 medium_magnitude_speed

| Adjacency configuration | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 2.739 | 5.396 | 2.712 | 5.326 | 57 |
| ego_only | 2.734 | 5.413 | 2.684 | 5.308 | 66 |
| nearby_only | **1.447** | **3.694** | **1.345** | **3.506** | 56 |
| no_ego_road | 1.925 | 4.350 | 1.725 | 4.095 | 76 |

`nearby_only` has an overwhelming advantage in the medium-speed scenario. Compared with `default`:

- ADE drops from 2.739 to 1.447, a decrease of about 47.2%
- FDE drops from 5.396 to 3.694, a decrease of about 31.5%

This indicates that in medium-speed scenarios, the full graph structure is not necessarily the source of gains; cleaner nearby-object relations may be better suited to what the current model can learn.

The `best_val_loss=0.4415` of `no_ego_road` is the lowest in this scenario, but its final ADE/FDE are worse than those of `nearby_only`. This again shows that there is a discrepancy between the weighted loss and the ADE/FDE evaluation objectives.

### 6.4 near_multiple_vehicles

| Adjacency configuration | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 1.552 | 2.871 | 1.514 | 2.767 | 97 |
| ego_only | **1.398** | **2.445** | **1.380** | **2.427** | 99 |
| nearby_only | 1.838 | 3.443 | 1.795 | 3.349 | 83 |
| no_ego_road | 1.623 | 2.945 | 1.623 | 2.945 | 100 |

This is the scenario with the best overall performance. `ego_only` achieves the best result, indicating that in this scenario the ego historical trajectory is already strong enough, and additionally adding nearby or full graph structure brings no benefit.

`nearby_only` performs worst here, in contrast to its strength in the other scenarios. This may mean that "multiple nearby vehicles" in the near multiple vehicles dataset does not necessarily imply strong interaction, or that the nearby-only graph, lacking road/ego information, easily misreads the relations among nearby objects.

### 6.5 traversing_intersection

| Adjacency configuration | Final ADE | Final FDE | Best ADE | Best FDE | Best Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | **2.508** | **5.648** | **2.453** | **5.473** | 40 |
| ego_only | 2.603 | 5.969 | 2.504 | 5.712 | 65 |
| nearby_only | 3.353 | 7.190 | 3.178 | 6.678 | 48 |
| no_ego_road | 3.169 | 6.778 | 3.057 | 6.401 | 42 |

The intersection-traversal scenario depends most on full context. `default` is the configuration with the best final ADE/FDE, `ego_only` is close but slightly worse, and `nearby_only` and `no_ego_road` lag clearly behind.

This indicates that intersection scenarios cannot rely only on nearby dynamic objects, nor can the ego-road relation simply be removed. Road topology, lane connectivity and intersection geometry are important for the 6-second future prediction.

---

## 7. Convergence and Early-Stopping Observations

Judging from `epochs_trained` and `best_val_epoch`, many experiments do not reach the best validation loss at the last epoch.

Typical examples:

| Scenario | Adjacency configuration | Total epochs | Best Val Epoch | Final ADE/FDE | Best ADE/FDE | Observation |
|------|----------|------------|----------------|--------------|--------------|------|
| high_magnitude_speed | default | 30 | 15 | 3.717 / 7.583 | 2.588 / 5.394 | Clear degradation in the second half |
| high_magnitude_speed | ego_only | 41 | 26 | 3.671 / 7.700 | 3.006 / 6.379 | Late-stage degradation |
| high_magnitude_speed | nearby_only | 40 | 25 | 2.831 / 5.917 | 2.526 / 5.479 | Slight degradation |
| medium_magnitude_speed | no_ego_road | 91 | 76 | 1.925 / 4.350 | 1.725 / 4.095 | Slight degradation |
| traversing_intersection | default | 55 | 40 | 2.508 / 5.648 | 2.453 / 5.473 | Slight degradation |

**Conclusion**

These results favour using `best_model_*.pth` rather than `final_model_*.pth`. Especially in the high-speed scenario, the gap between the final model and the best model is large, and subsequent experiment reports should state explicitly which checkpoint is cited.

---

## 8. Key Conclusions

### 8.1 The Best Adjacency Configuration Is Not Fixed

The current results do not support the conclusion that "one adjacency configuration is best across all scenarios".

- `nearby_only`: best overall average, wins 3 scenarios
- `ego_only`: best on `near_multiple_vehicles`
- `default`: best on `traversing_intersection`
- `no_ego_road`: never first on the final metrics, but has very low validation loss in some scenarios

A more reasonable judgement is that the adjacency configuration should be chosen by scenario type rather than fixed globally.

### 8.2 nearby_only Is Currently the Strongest Configuration on Average

Averaged over the 5 main scenarios, `nearby_only` achieves:

- Mean ADE: 2.223
- Mean FDE: 4.737

It is clearly better than the 2.539 / 5.163 of `default`. This indicates that the current model learns effective patterns more easily from local neighbour relations, whereas the full graph structure may introduce noise or training difficulty.

### 8.3 The High-Speed Scenario Most Needs Further Optimisation

The mean metrics of `high_magnitude_speed` are:

- Mean ADE: 3.384
- Mean FDE: 6.983

These are the worst among all scenarios. At high speed, terminal error is more easily amplified; future work should focus on:

- Whether the initial speed estimate is accurate
- Whether predictions systematically underestimate/overestimate speed
- Whether long-horizon FDE mainly comes from heading error or from speed-integration error
- Whether the best checkpoint is significantly better than the final checkpoint

### 8.4 Intersection Scenarios Need Full Context

In `traversing_intersection`, `default` is best, while `nearby_only` is markedly worse. This indicates that intersection scenarios cannot rely only on nearby vehicles; road structure, the ego-road relation and a more complete graph context must be retained.

### 8.5 The near_multiple_vehicles Results Are Counter-Intuitive but Stable

In `near_multiple_vehicles`, `ego_only` is best, and the overall metrics are the lowest. There are two possible explanations:

1. Although this dataset contains multiple nearby vehicles, the ego vehicle's future is mainly determined by its own historical motion, and the influence of interaction is limited
2. The current nearby/default graph construction does not make effective use of multi-vehicle information and instead introduces interference

This scenario merits a follow-up case study, because it can help determine whether "multiple nearby vehicles" really corresponds to strongly interactive prediction difficulty.

---

## 9. Suggested Follow-Up Experiments

1. **Use the best checkpoint consistently for formal comparisons**

   The final-epoch metrics in the report are suitable for viewing the end-of-training state, but model selection should prefer `best_model_*.pth`. The final metrics degrade markedly, especially in the high-speed scenario.

2. **Choose the adjacency configuration by scenario**

   Current recommendation:

   | Scenario type | Recommended adjacency configuration |
   |----------|--------------|
   | high_lateral_acceleration | nearby_only |
   | high_magnitude_speed | nearby_only |
   | medium_magnitude_speed | nearby_only |
   | near_multiple_vehicles | ego_only |
   | traversing_intersection | default |

3. **Perform a separate error decomposition for the high-speed scenario**

   It is recommended to decompose speed error, heading error and terminal lateral/longitudinal error, to confirm the main source of the high FDE.

4. **Check whether the default graph contains noise**

   default does not win in most scenarios, suggesting that the full graph may not be used effectively by the model. Possible checks:

   - Whether there are too many distant lanes / agents
   - Whether the edge types are too coarse
   - Whether the road-agent connections are too dense
   - Whether the features after pooling lose key directional information

5. **Add an LSTM / IDM / GNN-Trans comparison on the same batch of samples**

   This report currently only analyses the `results*` training directories. LSTM and IDM already have separate records, but to reach formal conclusions it is best to re-aggregate them on the same batch of samples with the same metric-recovery logic.

---

## 10. Summary

This batch of `results*` directories reflects a fairly complete set of ablation experiments over scenario types and adjacency configurations. Overall, `nearby_only` is currently the configuration with the best average performance, indicating that local dynamic adjacency is most helpful for most scenarios; but it is not a universal solution, and it is overtaken by `ego_only` and `default` on `near_multiple_vehicles` and `traversing_intersection`, respectively.

From a scenario perspective, `near_multiple_vehicles` is, counter-intuitively, currently the easiest scenario, while `high_magnitude_speed` and `traversing_intersection` are the main difficulties. The high-speed scenario has the largest terminal error, and several experiments show the final model degrading markedly relative to the best model; the intersection scenario relies more on full graph context, demonstrating that road topology information remains irreplaceable.

The most valuable next step is not to keep blindly scaling up training, but to make model selection, scenario grouping and the adjacency-configuration strategy more precise: use the best checkpoint in formal evaluation; perform error decomposition for the high-speed scenario; retain the default graph for the intersection scenario; and conduct a case study on the multi-vehicle proximity scenario to confirm whether it really contains strong interaction. Only then can this batch of ablation experiments be advanced from a "list of results" to "evidence that can guide the next round of model design".

---

## 11. Detailed Comparison of Our Model with the Two Baselines

Beyond the ablation results of the main model, this section draws on the existing `lstm_edit_record.md` and `idm_edit_record.md` to provide a more direct side-by-side analysis of the **GNN-Trans main model**, the **LSTM baseline** and the **IDM baseline**.

### 11.1 Caveat First: The Current Results Do Not Follow Exactly the Same Standard

This is the most important premise for understanding the three sets of results. **The numbers currently at hand are sufficient to support directional judgements, but they are not yet a final comparison table under a "fully unified evaluation protocol".**

There are mainly 4 kinds of differences in basis:

| Dimension | GNN-Trans main model | LSTM baseline | IDM baseline | Impact |
|------|------------------|---------------|--------------|------|
| Data source | Training statistics from the `results*` training directories | Output of a separate evaluation script | Output of a separate evaluation script | The main-model results come directly from training records, while the baselines come from additional inference evaluation |
| Checkpoint selection | The report mostly uses `final_ade/fde`, while also retaining `best_ade/fde` | Inference evaluation with the retrained `trained_lstm_model.pth` | Inference evaluation with the fixed current version of `idm_predictor.py` | The current main-model table leans towards "final-epoch results", while the baselines lean towards "actual deployed inference results" |
| Sample size | Each scenario uses statistics from its own training/validation process, not a unified fixed sample list | Explicit evaluation on 100 / 500 samples of `medium_magnitude_speed` | Explicit evaluation on 3000 samples of `medium_magnitude_speed` | The number of samples and the specific sample sets of the three are not fully consistent |
| Metric-recovery pipeline | Ready-made ADE/FDE statistics from the training directories | Computed after trajectory recovery by the fixed `lstm_predictor.py` | Computed after trajectory recovery by the fixed `idm_predictor.py` | Although all are ADE/FDE, they are not recomputed uniformly with the same script on the same batch of samples |

Therefore, **the current results have "differences in standard"**, but these differences mainly affect "the precision of the strict numerical ranking", and **do not overturn the overall judgement of the model capability hierarchy**.

More specifically:

1. **The current main-model report uses a training-result aggregation basis**
   - The main table in `results_analysis_report.md` mostly takes `final_ade / final_fde` from `loss_record_*.json`
   - This is more like a "summary of the end-of-training state"
   - But in many experiments `best_model` is in fact clearly better than `final_model`, especially in the high-speed scenario

2. **LSTM / IDM use an offline inference evaluation basis**
   - They actually run the checkpoint or predictor on processed `.pkl` samples, then recover the trajectories and compute ADE/FDE
   - This is more like "model performance in actual use"

3. **The sample sets are not fully unified**
   - The new LSTM model results come from 100 / 500 samples of `processed_medium_magnitude_speed`
   - The main final IDM results come from 3000 samples of the same scenario
   - The main-model results come from summaries of the corresponding training directories, not from offline recomputation on the same fixed batch of samples

4. **The scenario coverage also differs**
   - The main-model report covers 5 scenario types
   - The reliable LSTM / IDM results compiled so far are mainly concentrated on `medium_magnitude_speed`

**Therefore, the most reliable way to use the current comparison is:**
- Read it as "the approximate capability hierarchy of the three model types on `medium_magnitude_speed`"
- Rather than directly treating it as "a final leaderboard across all scenarios under a fully unified protocol"

### 11.2 Direct Comparison on `medium_magnitude_speed`

To minimise differences in basis, this section compares only the scenario for which all three lines currently have reliable results: `medium_magnitude_speed`.

| Model | Result source | Samples | ADE | FDE | Notes |
|------|----------|--------|-----|-----|------|
| GNN-Trans `default` | `results_analysis_report` main record | Training statistics | 2.739 | 5.396 | Full graph |
| GNN-Trans `ego_only` | `results_analysis_report` main record | Training statistics | 2.734 | 5.413 | Ego-related graph only |
| GNN-Trans `nearby_only` | `results_analysis_report` main record | Training statistics | **1.447** | **3.694** | Current best configuration of the main model |
| GNN-Trans `no_ego_road` | `results_analysis_report` main record | Training statistics | 1.925 | 4.350 | ego-road connections removed |
| LSTM baseline | `lstm_edit_record` | 500 | **2.19** | **6.39** | Fixed and retrained version |
| IDM baseline | `idm_edit_record` | 3000 | **9.38** | **18.54** | Fixed complete version |

This table directly shows a three-tier difference in capability:

1. **The best configuration of the main model outperforms LSTM**
   - `nearby_only` compared with LSTM:
   - ADE drops from `2.19` to `1.447`, a decrease of about **33.9%**
   - FDE drops from `6.39` to `3.694`, a decrease of about **42.2%**

2. **LSTM clearly outperforms IDM**
   - LSTM compared with IDM:
   - ADE drops from `9.38` to `2.19`, a decrease of about **76.7%**
   - FDE drops from `18.54` to `6.39`, a decrease of about **65.5%**

3. **There is an order-of-magnitude gap between the main model and the rule-based baseline**
   - `nearby_only` compared with IDM:
   - ADE decreases by about **84.6%**
   - FDE decreases by about **80.1%**

This indicates that:
- **IDM** provides a rule-based lower bound
- **LSTM** provides a credible lightweight learning baseline
- **GNN-Trans** is the main model that currently makes truly effective use of interaction-structure information

### 11.3 Where Exactly Our Model Differs from the LSTM Baseline

After the fixes, LSTM is already a credible baseline, but in essence it is still a **single-agent sequential model**.

Its main characteristics are:
- It mainly uses the ego historical trajectory
- It is good at learning smooth motion trends
- It can form a stable short-to-medium-term prediction baseline
- It makes limited use of complex interaction, road topology and graph-structure information

This is why LSTM can already reach `ADE ≈ 2.2m / FDE ≈ 6.4m` on `medium_magnitude_speed`, showing that it is not weak; but compared with the best configuration of the main model, **its terminal error FDE is still clearly larger**. This usually means that:

- LSTM fits the "overall motion inertia" fairly well
- But it lacks control over the long-term drift of the terminal position at 6 seconds in the future
- It accumulates error more easily on samples that require understanding nearby interactions

The advantage of the GNN-Trans main model lies precisely here. The report has already shown that:

- `nearby_only` is significantly better than `default` on `medium_magnitude_speed`
- ADE drops from `2.739` to `1.447`
- FDE drops from `5.396` to `3.694`

This indicates that what the current main model really benefits from is not "the more complex the graph, the better", but **an appropriate local interaction graph**. Compared with LSTM, the essential gains of the main model come from:

1. **Looking not only at how the ego vehicle itself moves, but also at how nearby key objects move**
2. **Modelling future trajectory prediction as an interaction problem, rather than only as single-sequence extrapolation**
3. **A larger improvement in FDE, which is more sensitive to terminal deviation, indicating more stable long-horizon prediction**

In other words, **LSTM is more like "learning ego inertia", while GNN-Trans is more like "learning the interaction between the ego vehicle and its neighbours"**.

### 11.4 Where Exactly Our Model Differs from the IDM Baseline

After the fixes, the IDM baseline has changed from "logically invalid" to a rule-based model that is "limited in capability but credible", but it still belongs to the **traditional rule-based paradigm**.

Its strengths are:
- Interpretable
- Stable behaviour
- Explicit lane-following constraints
- Suitable as an engineering lower-bound reference

Its weaknesses are:
- It relies mainly on hand-crafted rules and longitudinal car-following logic
- It has almost no learning capacity for complex interactions
- It adapts poorly to future diversity and scenario variation
- Its long-horizon terminal error is large

So the value of IDM is not in "catching up with the learning models", but in answering one question:

> If only rule-based kinematics + basic lane constraints are used, roughly what level can this task reach?

Judging from `ADE 9.38 / FDE 18.54`, it can already serve as a **reasonable lower bound**, but its gap to the learning models is structural and cannot be closed by simple hyperparameter tuning.

Therefore, the advantage of the main model over IDM should not be understood merely as "lower scores", but as:

- The main model has learnt spatio-temporal relations that a rule-based model can hardly encode explicitly
- Interaction information, local context and data-driven modelling are the main sources of the current performance leap

### 11.5 Why Our Model Is Not "a Single Structure That Is Best Across All Scenarios"

A very important conclusion in the report is: **although the main model is the strongest overall, internally no single configuration is best across all scenarios.**

The best configurations for the different scenarios are:

| Scenario type | Current best adjacency configuration |
|----------|------------------|
| high_lateral_acceleration | nearby_only |
| high_magnitude_speed | nearby_only |
| medium_magnitude_speed | nearby_only |
| near_multiple_vehicles | ego_only |
| traversing_intersection | default |

This indicates that the real advantage of the main model is not "a larger graph structure", but:

- Using the right graph structure in the right scenario
- Avoiding feeding irrelevant nodes, distant lanes and overly dense connections to the model as noise

This is also precisely the key difference between it and the two baselines:

- **IDM** has essentially no capacity for learning graph structure
- **LSTM** hardly models graph structure explicitly
- **GNN-Trans** derives its performance ceiling from graph structure, provided that the graph is constructed sensibly

So the correct understanding of the main model is not "we always win", but:

> We have shown that interaction-graph modelling is an effective direction, but the graph structure still needs to be chosen per scenario; all configurations cannot be lumped together.

### 11.6 The Most Reasonable Model Positioning at This Stage

Taking the current results together, the positioning of these three model types in the project is now fairly clear:

| Model | Current positioning | Core value |
|------|----------|----------|
| IDM | Rule-based lower-bound baseline | Interpretable, reproducible, provides the lower bound of traditional methods |
| LSTM | Lightweight learning baseline | Verifies the level achievable "using ego history alone" |
| GNN-Trans | Main model | Verifies the gain of interaction-graph modelling for trajectory prediction |

Therefore, the most reasonable conclusion at this stage is not "the baselines are meaningless", but:

- **IDM** establishes the lower-bound position of rule-based methods
- **LSTM** shows that a learning-based single-sequence model can already push the error down to a relatively low level
- **GNN-Trans** further shows that when the task requires stronger interaction modelling, a graph-structured model can markedly improve ADE/FDE, especially FDE

### 11.7 Conclusion: How the Current Results Should Be Interpreted

If the only question is "do the current results follow different standards", the answer is:

**Yes.**

The main differences are:
- The main model currently uses mostly a training-result aggregation basis
- LSTM / IDM use an offline inference evaluation basis
- The sample sizes and sample lists have not yet been fully unified
- Checkpoint selection has not been fully unified as "all best model" or "all final model" either

But if the further question is "will these differences overturn the overall judgement", the answer is:

**No.**

The current data already support the following judgements robustly enough:

1. On `medium_magnitude_speed`, **the best configuration of the GNN-Trans main model is clearly better than LSTM and IDM**
2. **LSTM is a credible and competitive learning baseline**, but is still weaker than the main model
3. **IDM can only serve as a rule-based lower bound**, and is not competitive at the same level as the learning models
4. The advantage of the main model comes mainly from **appropriate local interaction modelling**, rather than simply stacking more graph structure

### 11.8 Next-Step Recommendations: How to Make a Strictly Fair Comparison of the Three

If a formal paper table or final experimental conclusion is to be produced later, it is recommended to add a round under a **unified evaluation protocol**:

1. Fix the same list of `medium_magnitude_speed` sample files
2. Use the same ADE/FDE recovery and statistics code for GNN-Trans, LSTM and IDM
3. Specify explicitly whether `best checkpoint` or `final checkpoint` is used for all
4. Additionally output, on the same batch of samples:
   - mean / median ADE
   - mean / median FDE
   - Proportion of long-tail samples
   - Error broken down by scenario subtype

Only then can the "directional advantage" be upgraded to a "strictly publishable comparative conclusion".

---

## 12. Execution Plan for the Unified Fair Comparison on `medium_magnitude_speed`

To place GNN-Trans, LSTM and IDM under **as unified an evaluation standard as possible**, this project will next adopt the following protocol, conducting the formal comparison only on the `medium_magnitude_speed` scenario.

### 12.1 Comparison Subjects

- **GNN-Trans main model**: `nearby_only` configuration, using `best_model_nearby_only.pth`
- **LSTM baseline**: using `trained_lstm_model.pth` obtained by retraining after the fixes
- **IDM baseline**: using the fixed current version of `idm_predictor.py`

### 12.2 Sample Pool

Uniformly, from:

`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_data_by_type/processed_medium_magnitude_speed/*.pkl`

**3000 samples** are drawn.

### 12.3 Sample Selection Method

Instead of directly using "the first 3000 files", **stratified sampling** with the fixed random seed `42` is used.

Stratification criteria:

1. **Total future displacement length**
2. **Future heading change**

Specific procedure:

- First sort all `.pkl` files by file name
- Read the `GROUND_TRUTH` of each sample
- Compute:
  - `total_distance`: total displacement length of the 60 future step increments
  - `direction_change`: cumulative change in future heading angle
- Split each by tertiles into `low / medium / high`
- Form `3 × 3 = 9` bins
- Sample according to the proportion of samples in each bin, finally producing one fixed manifest

The purpose of this is to:

- Avoid the ordering bias introduced by "the first 3000 files"
- Avoid an excess of straight-driving or easy samples caused by pure random sampling
- Make the 3000 samples cover short/long displacement and low/high turning scenarios at the same time

### 12.4 Unified Evaluation Protocol

All three models must:

- Use the same manifest
- Predict the 60-step future trajectory
- Use the same ADE / FDE statistics
- Output results in a unified json format

It should be noted that:

- **LSTM / IDM** will directly read the `.pkl` files listed in the manifest for evaluation
- **GNN-Trans**, in order to reuse the existing training-time graph-construction logic, will:
  - Use `GraphDataset`
  - Load / generate `processed/config_<adj>/dataset_<adj>.pt` under the corresponding adjacency configuration
  - Then map manifest file names to specific sample indices

### 12.5 Resource Allocation

To reduce interference with existing tasks and complete the unified comparison in parallel, the following resource allocation is used:

| Model | Device |
|------|------|
| GNN-Trans | `GPU 2` |
| LSTM | `GPU 0` |
| IDM | `CPU` |
| `GPU 1` | Reserved, not used |

Notes:

- GNN-Trans is the heaviest model, so it is given priority on the less busy `GPU 2`
- LSTM can be placed on `GPU 0` on its own
- IDM is rule-based inference with limited benefit from a GPU, so it is placed on the CPU

It should additionally be noted that:

- In the runtime environment that Codex can actually access, `torch.cuda` is unavailable, and a quick probe reports directly:
  - `RuntimeError: No CUDA GPUs are available`
- Therefore, this round of the formal unified comparison has temporarily switched to a **CPU execution version**, to prioritise ensuring that the results can run through completely.
- If the host GPUs need to be used strictly later, the same manifest must be rerun in a terminal environment that can access the CUDA devices directly.

### 12.6 Background Execution

The formal tasks will be launched with `nohup`, so that they continue running after the SSH connection is closed.

After actual debugging, the background execution strategy was changed from "a single master script launching all three tasks" to "stepwise execution":

1. First generate the formal `3000`-sample manifest on its own
2. Then launch each independently:
   - `GNN-Trans`
   - `LSTM`
   - `IDM`
3. Each of the three tasks writes its own log and result files

The reasons for this are:

- It makes it easier to determine whether the hang is in the manifest, GNN, LSTM or IDM
- It avoids the whole task exiting silently because any single step of the master script fails
- It is better suited to long-running remote background execution and to checking after reconnection

After further debugging, an additional practical execution constraint was added:

- In the Codex automation environment, when child processes are launched with bare `nohup`, there is a risk that they are reaped quickly.
- The persistence approach finally adopted is therefore:
  - **`tmux` detached session + `nohup` parent script**
- This retains `nohup` background execution while improving survival stability after disconnection.

The core files planned to be generated include:

- manifest:
  - `experiments/analysis/comparisons/<run>/medium_speed_manifest_3000.json`
- GNN-Trans results:
  - `experiments/analysis/comparisons/<run>/gnn_nearby_only_3000.json`
- LSTM results:
  - `experiments/analysis/comparisons/<run>/lstm_3000.json`
- IDM results:
  - `experiments/analysis/comparisons/<run>/idm_3000.json`
- Logs:
  - `experiments/analysis/comparisons/<run>/master.log`
  - `experiments/analysis/comparisons/<run>/gnn_eval.log`
  - `experiments/analysis/comparisons/<run>/lstm_eval.log`
  - `experiments/analysis/comparisons/<run>/idm_eval.log`

### 12.7 Current Status

- [x] Confirmed the path of the main-model best checkpoint for `medium_magnitude_speed nearby_only`
- [x] Added LSTM / IDM manifest evaluation scripts
- [x] Added the GNN-Trans manifest evaluation script
- [x] Added the 3000-sample manifest generation script
- [x] Completed syntax checks
- [x] Completed standalone verification of manifest generation
- [x] Completed small-sample verification of the LSTM / IDM manifest evaluation
- [x] Completed GNN-Trans `nearby_only` graph-cache generation and small-sample verification
- [x] Identified the cause of the GPU-version failure as `CUDA` being unavailable in the current execution environment
- [x] Switched to the CPU-version formal evaluation plan
- [x] Relaunched the formal tasks via `tmux + nohup`
- [ ] Fill the results back into this report once the tasks are complete

Failed intermediate results that have been cleaned up:

- The following failed run directories and old logs have been deleted, to avoid mixing old and new results:
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_142805`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143615`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143641`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143752`
  - `experiments/analysis/comparisons/medium_speed_manifest_3000_test.json`
  - `experiments/analysis/comparisons/nohup_launch_20260429.log`

Status of the first round of background tasks:

- First `nohup` launch time: `2026-04-29 14:28:05`
- First run directory:
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_142805`
- First master log:
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_142805/master.log`

Debugging conclusions:

- The first-round `nohup` task exited at a very early stage and never entered the formal evaluation.
- Subsequent split verification found that:
  - `generate_medium_speed_manifest.py` can generate the 3000-sample manifest normally;
  - `run_lstm_pkl_eval.py --manifest` runs normally;
  - `run_idm_pkl_eval.py --manifest` runs normally;
  - On its first run, `run_gnn_manifest_eval.py` first generates the complete `processed/config_nearby_only/dataset_nearby_only.pt`; this process takes a long time and was one of the main risk factors behind the earlier misjudgement that "the task was making no progress".

Key verification results completed:

- A test version of the 3000-sample manifest has been generated successfully:
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_manifest_3000_test.json`
- LSTM test results on the first 50 samples of the unified manifest:
  - `ADE = 2.27 m`
  - `FDE = 6.79 m`
- IDM test results on the first 50 samples of the unified manifest:
  - `ADE = 8.83 m`
  - `FDE = 15.78 m`
- GNN-Trans test results on the first 12 samples of the unified manifest:
  - `ADE = 1.36 m`
  - `FDE = 3.43 m`
- The `nearby_only` graph cache has been generated successfully:
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_data_by_type/processed_medium_magnitude_speed/processed/config_nearby_only/dataset_nearby_only.pt`

Notes:

- Since the `nearby_only` cache has already been generated, the subsequent formal 3000-sample evaluation no longer needs to rebuild the graphs, and GNN-Trans will start noticeably faster than in the first test.
- Current time: `2026-04-29 15:23:34`
- The current execution plan is:
  1. Generate the formal manifest
  2. Launch the formal CPU evaluation via `tmux + nohup`
  3. Collect the three json files and produce the unified comparison conclusions

Current formal run information:

- Run directory:
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise`
- `tmux` session:
  - `medium3000_stepwise_150505`
- Master log:
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/master.log`
- GNN log:
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_eval.log`
- LSTM log:
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_eval.log`
- IDM log:
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_eval.log`

Currently confirmed live status:

- `GNN-Trans` has started progressing on the `3000` samples, and a progress bar has appeared in the log
- The `LSTM` and `IDM` subtasks have also been launched by the parent script; continue to monitor whether their logs are written to disk

- This section will later be supplemented with the final unified comparison results of the three models.
