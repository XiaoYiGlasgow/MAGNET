# GNN_Trans 结果总汇总（论文写作草稿）

更新时间：2026-05-08  
用途：为论文、附录、答辩稿提供一份集中整理的结果摘要  
工作区：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans`

## 1. 使用建议

这份汇总优先整理“当前已经能确认”的结果，并按**可直接引用程度**分层：

1. **最可靠、最适合直接写论文的结果**
   - `medium_magnitude_speed` 上的统一 manifest 对比
   - `analysis.md` 中已经整理好的 `transformer_only` 与 4 个 GNN 结构结果
2. **可支持方向性结论，但还不是最终统一协议主表的结果**
   - 其他场景下 `transformer_only` 与 4 个 GNN 结构的阶段性对照
3. **当前未找到可直接引用 benchmark 的对象**
   - `VectorNet`

如果后续论文正文只保留一组最稳的数字，建议优先使用：

- `medium_magnitude_speed` 上的统一 `3000` 样本 manifest 对比
- 当前 4 个已完成场景的 `GNN_Trans / transformer_only` 汇总表

## 2. 已确认的数据来源

### 2.1 主模型与跨场景分析

- [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md)
- [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md)

### 2.2 `medium_magnitude_speed` 统一 manifest 对比

- [gnn_nearby_only_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_nearby_only_3000.json)
- [lstm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_3000.json)
- [idm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_3000.json)
- [medium_speed_manifest_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/medium_speed_manifest_3000.json)

### 2.3 `transformer_only` 的统一结果来源

- [transformer_only_medium_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/transformer_only_medium_run_20260506_160648/transformer_only_medium_3000.json)
- [gnn_inference_timing_summary.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/medium_magnitude_speed_infer_timing_with_transformer_only/gnn_inference_timing_summary.json)
- [strict_forward_summary.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/strict_forward_transformer_only_20260507/strict_forward_summary.json)

## 3. 当前最稳的主结论

### 3.1 GNN 邻接结构主结论

基于 [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md)，当前 5 个主场景里：

- `nearby_only` 在 `high_lateral_acceleration`、`high_magnitude_speed`、`medium_magnitude_speed` 上最优
- `ego_only` 在 `near_multiple_vehicles` 上最优
- `default` 在 `traversing_intersection` 上最优

这说明：

1. `nearby_only` 是当前**平均表现最稳定**的 GNN 邻接配置
2. 它不是万能解
3. 不同场景对交互结构的依赖不同

### 3.2 `transformer_only` 主结论

基于 [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md)，当前 4 个已完成场景里：

1. `transformer_only` 在 `medium_magnitude_speed`、`high_magnitude_speed`、`high_lateral_acceleration`、`traversing_intersection` 中都表现出很强的推理效率
2. 其精度表现具有明显场景依赖
3. `high_lateral_acceleration` 是当前最强的 `transformer_only` 支撑场景

最稳妥的表述不是“保留 GNN 一定更好”或“去掉 GNN 一定更好”，而是：

> 模型收益主要来自场景相关的结构设计，而不是“是否存在 GNN 模块”这个标签本身。

## 4. 5 个主场景的 GNN 主记录总览

下表整理自 [results_analysis_report.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/docs/reports/results_analysis_report.md:83)，采用各场景主记录中的最终 ADE/FDE。

| 场景 | default ADE/FDE | ego_only ADE/FDE | nearby_only ADE/FDE | no_ego_road ADE/FDE | 当前最优 |
|---|---|---|---|---|---|
| high_lateral_acceleration | 2.180 / 4.317 | 2.063 / 3.870 | **1.647 / 3.440** | 2.369 / 4.641 | nearby_only |
| high_magnitude_speed | 3.717 / 7.583 | 3.671 / 7.700 | **2.831 / 5.917** | 3.318 / 6.733 | nearby_only |
| medium_magnitude_speed | 2.739 / 5.396 | 2.734 / 5.413 | **1.447 / 3.694** | 1.925 / 4.350 | nearby_only |
| near_multiple_vehicles | 1.552 / 2.871 | **1.398 / 2.445** | 1.838 / 3.443 | 1.623 / 2.945 | ego_only |
| traversing_intersection | **2.508 / 5.648** | 2.603 / 5.969 | 3.353 / 7.190 | 3.169 / 6.778 | default |

## 5. `medium_magnitude_speed` 上的统一 baseline 对比

这一组是当前最适合直接写论文正文或附录的 baseline 对比，因为对应的 `GNN / LSTM / IDM` 都已经在统一的 `3000` 样本 manifest 上跑过。

### 5.1 统一 manifest 的 3 模型对比

数据来源：

- [gnn_nearby_only_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_nearby_only_3000.json)
- [lstm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_3000.json)
- [idm_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_3000.json)

| 模型 | 样本口径 | mean_ADE | mean_FDE | 备注 |
|---|---|---:|---:|---|
| GNN-Trans `nearby_only` | 统一 manifest 3000 | **1.4969** | **3.6743** | 当前主模型最强基线 |
| LSTM baseline | 统一 manifest 3000 | 2.2109 | 6.5029 | 已修复并重训练 |
| IDM baseline | 统一 manifest 3000 | 9.1349 | 18.0458 | 规则下界 |

从这组结果可以直接得到：

1. `nearby_only` 明显优于 LSTM
2. LSTM 明显优于 IDM
3. 主模型与规则 baseline 之间存在很大差距

### 5.2 补充：`transformer_only` 在同场景下的结果

`transformer_only` 的 `medium_magnitude_speed` 结果来自：

- [transformer_only_medium_3000.json](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/transformer_only_medium_run_20260506_160648/transformer_only_medium_3000.json)

| 模型 | 样本口径 | mean_ADE | mean_FDE | 备注 |
|---|---|---:|---:|---|
| GNN-Trans `nearby_only` | 统一 manifest 3000 | **1.4969** | 3.6743 | 精度最优 |
| Transformer-only | 同场景 3000 样本 manifest | 1.6526 | **3.6243** | 强效率基线 |
| LSTM baseline | 统一 manifest 3000 | 2.2109 | 6.5029 | 轻量学习基线 |
| IDM baseline | 统一 manifest 3000 | 9.1349 | 18.0458 | 规则下界 |

这一组最适合支持下面这句：

> 在 `medium_magnitude_speed` 上，`nearby_only` 给出最佳整体精度，`transformer_only` 给出最强效率-精度折中，而 LSTM 与 IDM 分别构成可信的轻量学习基线和规则下界。

## 6. 当前 4 个已完成场景的 `GNN + transformer_only` 总表

这张表整理自 [analysis.md](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/gnn_trans_ablation/analysis/analysis.md:939)，已经统一成：

1. `每1000样本平均训练时长(h)`
2. `mean_ADE`
3. `mean_FDE`
4. `strict sample_time_ms`
5. `end-to-end throughput(samples/s)`

| 场景 | 配置 | 每1000样本平均训练时长(h) | mean_ADE | mean_FDE | strict sample_time_ms | end-to-end throughput(samples/s) |
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

### 6.1 从这张总表能直接写出的结论

1. `transformer_only` 在 4 个已完成场景里都表现出最强或接近最强的推理效率
2. 最佳精度配置具有明显场景依赖
3. `high_lateral_acceleration` 是当前最强的 `transformer_only` 支撑场景
4. `medium_magnitude_speed` 仍然最适合强调 `nearby_only` 的最佳精度优势

## 7. 当前最推荐的论文表述

如果只写一句总述，当前最稳妥的版本是：

> The main benefit of GNN-Trans comes from scenario-dependent interaction structure design rather than from the mere presence of a GNN module. Across scenarios, `nearby_only` remains the most stable high-accuracy graph configuration, while the Transformer-only baseline consistently provides the strongest inference efficiency and becomes highly competitive in selected scenarios such as `high_lateral_acceleration`.

对应中文可以写成：

> GNN-Trans 的优势并不来自“是否加入 GNN 模块”本身，而主要来自场景相关的交互结构设计。跨场景看，`nearby_only` 仍然是最稳定的高精度图结构配置，而 `transformer_only` 则稳定体现出最强的推理效率，并在 `high_lateral_acceleration` 等场景中展现出很强的精度竞争力。

## 8. 当前缺口与不要误写的地方

### 8.1 `VectorNet`

当前在仓库里只确认到了 `VectorNet` 相关的特征编码与数据处理代码，例如：

- [data_utils.py](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/utils/data_utils.py:555)
- [data_processor.py](/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/utils/data_processor.py:243)

**没有定位到可直接引用的 `VectorNet` benchmark 结果表或结果 JSON。**  
因此，在当前论文草稿中：

1. 不建议把 `VectorNet` 写进正式结果表
2. 除非后续重新找到其评测结果，或重新运行一轮统一评测

### 8.2 当前还不是完全统一协议的大总表

虽然这份汇总已经尽量把结果放到一起，但仍需注意：

1. `medium_magnitude_speed` 上的 `GNN / LSTM / IDM` 对比最统一、最适合直接引用
2. `transformer_only` 的跨场景结果已经完整，但目前主要是与 4 个 GNN 配置的对照
3. `near_multiple_vehicles` 的 `transformer_only` 还没补跑
4. `VectorNet` 结果暂时缺失

## 9. 最终建议

如果你现在就要写论文，我建议优先采用下面的结果结构：

1. **主表**
   - 使用 `medium_magnitude_speed` 上的：
     - `GNN-Trans nearby_only`
     - `Transformer-only`
     - `LSTM`
     - `IDM`
2. **补充表**
   - 使用当前 4 个已完成场景的 `GNN + transformer_only` 总表
3. **不要先写入正文**
   - `VectorNet`
   - `near_multiple_vehicles` 的 `transformer_only`
   - 未统一协议的旧小样本 LSTM 数字

如果后续需要，我下一步最适合做的是：

1. 直接把这份 `summary.md` 再改写成论文里的 `Results` 小节草稿
2. 或者把这里的表格转成 LaTeX 版本
