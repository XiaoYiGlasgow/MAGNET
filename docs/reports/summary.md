# GNN_Trans Overall Results Summary (Paper-Writing Draft)

Last updated: 2026-05-08
Purpose: a centrally organised results summary for the paper, appendix and defence slides
Workspace: `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans`

## 1. Usage Recommendations

This summary prioritises results that "can currently be confirmed", and stratifies them by **how directly they can be cited**:

1. **Most reliable results, best suited for direct use in the paper**
   - The unified-manifest comparison on `medium_magnitude_speed`
   - The `transformer_only` and 4 GNN structure results already compiled in `analysis.md`
2. **Results that support directional conclusions, but are not yet a final unified-protocol main table**
   - Interim comparisons of `transformer_only` and the 4 GNN structures in the other scenarios
3. **Objects for which no directly citable benchmark has been found so far**
   - `VectorNet`

If the main text of the paper later keeps only one set of the most robust numbers, it is recommended to prioritise:

- The unified `3000`-sample manifest comparison on `medium_magnitude_speed`
- The `GNN_Trans / transformer_only` summary table for the 4 currently completed scenarios

## 2. Confirmed Data Sources

### 2.1 Main Model and Cross-Scenario Analysis

- [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md)
- [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md)

### 2.2 `medium_magnitude_speed` Unified-Manifest Comparison

- [gnn_nearby_only_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_nearby_only_3000.json)
- [lstm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_3000.json)
- [idm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_3000.json)
- [medium_speed_manifest_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/medium_speed_manifest_3000.json)

### 2.3 Unified Result Sources for `transformer_only`

- [transformer_only_medium_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/transformer_only_medium_run_20260506_160648/transformer_only_medium_3000.json)
- [gnn_inference_timing_summary.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/medium_magnitude_speed_infer_timing_with_transformer_only/gnn_inference_timing_summary.json)
- [strict_forward_summary.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/strict_forward_transformer_only_20260507/strict_forward_summary.json)

## 3. Current Most Robust Main Conclusions

### 3.1 Main Conclusion on GNN Adjacency Structure

Based on [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md), across the current 5 main scenarios:

- `nearby_only` is best on `high_lateral_acceleration`, `high_magnitude_speed` and `medium_magnitude_speed`
- `ego_only` is best on `near_multiple_vehicles`
- `default` is best on `traversing_intersection`

This indicates that:

1. `nearby_only` is currently the GNN adjacency configuration with the **most stable average performance**
2. It is not a universal solution
3. Different scenarios depend on the interaction structure to different degrees

### 3.2 Main Conclusion on `transformer_only`

Based on [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md), across the 4 currently completed scenarios:

1. `transformer_only` shows very strong inference efficiency in `medium_magnitude_speed`, `high_magnitude_speed`, `high_lateral_acceleration` and `traversing_intersection`
2. Its accuracy is clearly scenario-dependent
3. `high_lateral_acceleration` is currently the scenario that most strongly supports `transformer_only`

The most cautious formulation is neither "keeping the GNN is always better" nor "removing the GNN is always better", but rather:

> The model's gains come mainly from scenario-dependent structural design, rather than from the label of "whether a GNN module is present" itself.

## 4. Overview of Main GNN Records for the 5 Main Scenarios

The table below is compiled from [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md:83), using the final ADE/FDE from each scenario's main record.

| Scenario | default ADE/FDE | ego_only ADE/FDE | nearby_only ADE/FDE | no_ego_road ADE/FDE | Current best |
|---|---|---|---|---|---|
| high_lateral_acceleration | 2.180 / 4.317 | 2.063 / 3.870 | **1.647 / 3.440** | 2.369 / 4.641 | nearby_only |
| high_magnitude_speed | 3.717 / 7.583 | 3.671 / 7.700 | **2.831 / 5.917** | 3.318 / 6.733 | nearby_only |
| medium_magnitude_speed | 2.739 / 5.396 | 2.734 / 5.413 | **1.447 / 3.694** | 1.925 / 4.350 | nearby_only |
| near_multiple_vehicles | 1.552 / 2.871 | **1.398 / 2.445** | 1.838 / 3.443 | 1.623 / 2.945 | ego_only |
| traversing_intersection | **2.508 / 5.648** | 2.603 / 5.969 | 3.353 / 7.190 | 3.169 / 6.778 | default |

## 5. Unified Baseline Comparison on `medium_magnitude_speed`

This set is currently the baseline comparison best suited for direct use in the main text or appendix of the paper, because the corresponding `GNN / LSTM / IDM` have all been run on the same unified `3000`-sample manifest.

### 5.1 3-Model Comparison on the Unified Manifest

Data sources:

- [gnn_nearby_only_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_nearby_only_3000.json)
- [lstm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_3000.json)
- [idm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_3000.json)

| Model | Sample basis | mean_ADE | mean_FDE | Notes |
|---|---|---:|---:|---|
| GNN-Trans `nearby_only` | Unified manifest 3000 | **1.4969** | **3.6743** | Strongest baseline of the current main model |
| LSTM baseline | Unified manifest 3000 | 2.2109 | 6.5029 | Fixed and retrained |
| IDM baseline | Unified manifest 3000 | 9.1349 | 18.0458 | Rule-based lower bound |

From this set of results, the following can be stated directly:

1. `nearby_only` is clearly better than LSTM
2. LSTM is clearly better than IDM
3. There is a large gap between the main model and the rule-based baseline

### 5.2 Supplement: `transformer_only` Results in the Same Scenario

The `medium_magnitude_speed` results for `transformer_only` come from:

- [transformer_only_medium_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/transformer_only_medium_run_20260506_160648/transformer_only_medium_3000.json)

| Model | Sample basis | mean_ADE | mean_FDE | Notes |
|---|---|---:|---:|---|
| GNN-Trans `nearby_only` | Unified manifest 3000 | **1.4969** | 3.6743 | Best accuracy |
| Transformer-only | Same-scenario 3000-sample manifest | 1.6526 | **3.6243** | Strong efficiency baseline |
| LSTM baseline | Unified manifest 3000 | 2.2109 | 6.5029 | Lightweight learning baseline |
| IDM baseline | Unified manifest 3000 | 9.1349 | 18.0458 | Rule-based lower bound |

This set is best suited to supporting the following sentence:

> On `medium_magnitude_speed`, `nearby_only` gives the best overall accuracy, `transformer_only` gives the strongest efficiency-accuracy trade-off, while LSTM and IDM constitute a credible lightweight learning baseline and a rule-based lower bound, respectively.

## 6. `GNN + transformer_only` Overall Table for the 4 Currently Completed Scenarios

This table is compiled from [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md:939) and has been unified into:

1. `Average training time per 1000 samples (h)`
2. `mean_ADE`
3. `mean_FDE`
4. `strict sample_time_ms`
5. `end-to-end throughput(samples/s)`

| Scenario | Configuration | Average training time per 1000 samples (h) | mean_ADE | mean_FDE | strict sample_time_ms | end-to-end throughput(samples/s) |
|---|---|---:|---:|---:|---:|---:|
| medium_magnitude_speed | default | 1.1172 | 2.6904 | 5.1699 | 7.2072 | 76.9231 |
| medium_magnitude_speed | ego_only | 1.3339 | 2.4279 | 4.6692 | 7.3156 | 78.7402 |
| medium_magnitude_speed | nearby_only | 1.1289 | 1.4969 | 3.6743 | 7.2519 | 78.1250 |
| medium_magnitude_speed | no_ego_road | 1.4844 | 1.9259 | 4.1364 | 7.0416 | 81.9672 |
| medium_magnitude_speed | transformer_only | 1.3028 | 1.6526 | 3.6243 | 6.4457 | 99.0099 |
| high_magnitude_speed | default | 0.4822 | 2.7965 | 5.7615 | 7.3651 | 70.9220 |
| high_magnitude_speed | ego_only | 0.7356 | 3.3608 | 7.1566 | 7.5349 | 70.9220 |
| high_magnitude_speed | nearby_only | 0.6378 | 2.8843 | 5.8583 | 7.4476 | 70.9220 |
| high_magnitude_speed | no_ego_road | 0.9667 | 2.9215 | 5.8619 | 7.1208 | 73.5294 |
| high_magnitude_speed | transformer_only | 0.9822 | 2.9380 | 5.7821 | 6.5890 | 81.9672 |
| high_lateral_acceleration | default | 1.6489 | 1.9334 | 3.6450 | 7.3237 | 65.3595 |
| high_lateral_acceleration | ego_only | 1.6252 | 1.8791 | 3.3366 | 7.4597 | 67.1141 |
| high_lateral_acceleration | nearby_only | 1.6259 | 1.6177 | 3.2038 | 7.4486 | 68.4932 |
| high_lateral_acceleration | no_ego_road | 1.5230 | 2.1423 | 4.0396 | 7.2260 | 69.9301 |
| high_lateral_acceleration | transformer_only | 1.5148 | 1.5276 | 3.0625 | 6.6613 | 81.9672 |
| traversing_intersection | default | 0.9022 | 2.3971 | 5.2328 | 7.2586 | 70.4225 |
| traversing_intersection | ego_only | 1.3222 | 2.4004 | 4.9199 | 7.3798 | 69.9301 |
| traversing_intersection | nearby_only | 1.0333 | 2.9656 | 5.9692 | 7.3144 | 70.9220 |
| traversing_intersection | no_ego_road | 0.9422 | 2.8550 | 5.8315 | 7.1747 | 71.9424 |
| traversing_intersection | transformer_only | 0.9689 | 2.8211 | 5.6075 | 6.4681 | 84.7458 |

### 6.1 Conclusions That Can Be Written Directly from This Overall Table

1. `transformer_only` shows the strongest or near-strongest inference efficiency in all 4 completed scenarios
2. The best-accuracy configuration is clearly scenario-dependent
3. `high_lateral_acceleration` is currently the scenario that most strongly supports `transformer_only`
4. `medium_magnitude_speed` remains the most suitable scenario for emphasising the best-accuracy advantage of `nearby_only`

## 7. Currently Most Recommended Paper Wording

If only one overall sentence is written, the currently most cautious version is:

> The main benefit of GNN-Trans comes from scenario-dependent interaction structure design rather than from the mere presence of a GNN module. Across scenarios, `nearby_only` remains the most stable high-accuracy graph configuration, while the Transformer-only baseline consistently provides the strongest inference efficiency and becomes highly competitive in selected scenarios such as `high_lateral_acceleration`.

## 8. Current Gaps and Things Not to Misstate

### 8.1 `VectorNet`

At present, only `VectorNet`-related feature encoding and data processing code has been confirmed in the repository, for example:

- [data_utils.py](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/utils/data_utils.py:555)
- [data_processor.py](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/utils/data_processor.py:243)

**No directly citable `VectorNet` benchmark result table or result JSON has been located.**
Therefore, in the current paper draft:

1. It is not recommended to include `VectorNet` in the formal results table
2. Unless its evaluation results are found later, or a new round of unified evaluation is run

### 8.2 This Is Not Yet a Fully Unified-Protocol Overall Table

Although this summary brings the results together as far as possible, the following should still be noted:

1. The `GNN / LSTM / IDM` comparison on `medium_magnitude_speed` is the most unified and the most suitable for direct citation
2. The cross-scenario results for `transformer_only` are complete, but are currently mainly a comparison against the 4 GNN configurations
3. `transformer_only` on `near_multiple_vehicles` has not yet been run
4. `VectorNet` results are currently missing

## 9. Final Recommendations

If you are going to write the paper now, I recommend prioritising the following results structure:

1. **Main table**
   - Use, on `medium_magnitude_speed`:
     - `GNN-Trans nearby_only`
     - `Transformer-only`
     - `LSTM`
     - `IDM`
2. **Supplementary table**
   - Use the `GNN + transformer_only` overall table for the 4 currently completed scenarios
3. **Do not put in the main text yet**
   - `VectorNet`
   - `transformer_only` on `near_multiple_vehicles`
   - Old small-sample LSTM numbers not under the unified protocol

If needed later, the most suitable next steps for me are:

1. Directly rewrite this `summary.md` into a draft `Results` subsection for the paper
2. Or convert the tables here into LaTeX
