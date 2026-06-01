# GNN_Trans results 目录实验结果分析报告

> 适用项目：`项目 A（轨迹预测 / baseline / planning）`
>
> 定位：阶段性大报告。用于保留完整统计过程，不作为第一次上手的主入口。

分析范围：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/results*`  
生成日期：2026-04-28  
输出位置：`GNN_Trans/docs/reports/results_analysis_report.md`

---

## 1. 数据来源与统计口径

本报告扫描了 `GNN_Trans` 根目录下所有以 `results` 开头的实验结果目录，共发现 **20 个顶层结果目录**，覆盖以下实验矩阵：

- 场景类型：`high_lateral_acceleration`、`high_magnitude_speed`、`medium_magnitude_speed`、`near_multiple_vehicles`、`traversing_intersection`
- 邻接配置：`default`、`ego_only`、`nearby_only`、`no_ego_road`

每个训练目录主要包含：

- `training_progress.json`：最后一个 epoch 的摘要
- `results/loss_record_*.json`：完整训练曲线、最终 ADE/FDE、最佳 ADE/FDE、最佳验证 loss 等
- `results/loss_components_*.json`：loss 组成项
- `models/best_model_*.pth` 与 `models/final_model_*.pth`
- `logs/training_log.txt`

本报告优先使用 `loss_record_*.json` 中的字段，因为它包含完整曲线和最终统计。报告中的主要指标含义如下：

| 指标 | 含义 |
|------|------|
| `final_ade` | 最后一个 epoch 的 ADE |
| `final_fde` | 最后一个 epoch 的 FDE |
| `best_ade` | 训练过程中记录到的最低 ADE |
| `best_fde` | 训练过程中记录到的最低 FDE |
| `best_val_loss` | 训练过程中最低验证 loss |
| `best_val_epoch` | `best_val_loss` 对应 epoch |

需要注意：`best_val_loss` 与最低 ADE/FDE 不一定出现在同一个 epoch，因为训练 loss 是 ADE、FDE、smoothness、direction 的加权组合，权重为：

| loss 项 | 权重 |
|---------|------|
| ADE | 0.5 |
| FDE | 3.0 |
| Smoothness | 0.3 |
| Direction | 0.5 |

---

## 2. 实验目录总览

共解析到 **22 条训练记录**。多出的 2 条来自 `high_magnitude_speed/default`，该配置下保留了 `ds900`、`ds450`、`ds4500` 三个历史训练版本。

| 场景类型 | 邻接配置 | 训练样本数 | Epoch | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Loss | 训练目录 |
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

## 3. 主对比表：按最大样本规模选择每个配置

为了做公平横向比较，本节对同一“场景类型 + 邻接配置”只保留一个主记录。选择规则是：优先选择训练样本数最大的记录；若样本数相同，则选择时间更新的记录。

这会将 `high_magnitude_speed/default` 的主记录确定为 `ds4500` 版本，而不是 `ds900` 或 `ds450`。

| 场景类型 | default ADE/FDE | ego_only ADE/FDE | nearby_only ADE/FDE | no_ego_road ADE/FDE | 最优最终 ADE/FDE |
|----------|-----------------|------------------|---------------------|---------------------|------------------|
| high_lateral_acceleration | 2.180 / 4.317 | 2.063 / 3.870 | **1.647 / 3.440** | 2.369 / 4.641 | nearby_only |
| high_magnitude_speed | 3.717 / 7.583 | 3.671 / 7.700 | **2.831 / 5.917** | 3.318 / 6.733 | nearby_only |
| medium_magnitude_speed | 2.739 / 5.396 | 2.734 / 5.413 | **1.447 / 3.694** | 1.925 / 4.350 | nearby_only |
| near_multiple_vehicles | 1.552 / 2.871 | **1.398 / 2.445** | 1.838 / 3.443 | 1.623 / 2.945 | ego_only |
| traversing_intersection | **2.508 / 5.648** | 2.603 / 5.969 | 3.353 / 7.190 | 3.169 / 6.778 | default |

**直接观察**

1. `nearby_only` 在 5 个场景里赢了 3 个：`high_lateral_acceleration`、`high_magnitude_speed`、`medium_magnitude_speed`。
2. `ego_only` 在 `near_multiple_vehicles` 上最强，说明该场景里自车历史运动本身已经提供了非常强的预测信号。
3. `default` 在 `traversing_intersection` 上最强，说明路口场景可能更依赖完整图结构，而不是只依赖附近 agent 或自车信息。
4. `no_ego_road` 没有在任何场景中成为最终 ADE/FDE 第一，但在 `medium_magnitude_speed` 上验证 loss 最低，说明 loss 与 ADE/FDE 指标之间并非完全一致。

---

## 4. 邻接配置整体对比

按 5 个主场景求平均：

| 邻接配置 | 平均最终 ADE | 平均最终 FDE | 排名 |
|----------|--------------|--------------|------|
| nearby_only | **2.223** | **4.737** | 1 |
| no_ego_road | 2.481 | 5.089 | 2 |
| ego_only | 2.494 | 5.079 | 3 |
| default | 2.539 | 5.163 | 4 |

**解读**

`nearby_only` 是目前整体最稳定的邻接配置，平均 ADE/FDE 都最低。这个结果有点有趣：完整 `default` 图并没有稳定胜出，说明当前模型可能还没有充分利用复杂图结构，或者完整图引入了额外噪声。

`ego_only` 与 `no_ego_road` 的平均结果非常接近。`ego_only` 在多车近邻场景表现最好，但在高速和路口场景并不占优；`no_ego_road` 在中速场景较强，但整体没有一个场景拿到最终指标第一。

`default` 平均排名靠后，但它在 `traversing_intersection` 上最好，这一点很重要。不能简单说 default 没用，更合理的判断是：完整图结构对某些场景有帮助，但当前训练或特征融合方式没有让它在所有场景中稳定获益。

---

## 5. 场景难度对比

按每个场景下 4 种邻接配置的最终指标求平均：

| 场景类型 | 平均最终 ADE | 平均最终 FDE | 难度判断 |
|----------|--------------|--------------|----------|
| near_multiple_vehicles | **1.603** | **2.926** | 最容易 |
| high_lateral_acceleration | 2.065 | 4.067 | 较容易 |
| medium_magnitude_speed | 2.211 | 4.713 | 中等 |
| traversing_intersection | 2.908 | 6.396 | 偏难 |
| high_magnitude_speed | **3.384** | **6.983** | 最难 |

**解读**

`near_multiple_vehicles` 虽然听起来复杂，但在这些结果里是最容易的场景，尤其 `ego_only` 可以达到 `ADE 1.398 / FDE 2.445`。这可能说明该数据集中的 near multiple vehicles 场景并不一定包含强交互未来变化，或者自车轨迹惯性非常稳定。

`high_magnitude_speed` 是最困难的场景。即使最好的 `nearby_only` 也只有 `ADE 2.831 / FDE 5.917`，其他配置普遍在 `FDE 6.7-7.7m`。高速场景下，速度估计、长时积分误差和末端偏差都会被放大。

`traversing_intersection` 是第二难场景，且 `nearby_only` 和 `no_ego_road` 明显退化。路口场景对道路拓扑和完整上下文更敏感，因此 `default` 反而成为最优。

---

## 6. 各场景详细分析

### 6.1 high_lateral_acceleration

| 邻接配置 | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 2.180 | 4.317 | 2.134 | 4.200 | 89 |
| ego_only | 2.063 | 3.870 | 2.059 | 3.870 | 100 |
| nearby_only | **1.647** | **3.440** | **1.581** | **3.306** | 95 |
| no_ego_road | 2.369 | 4.641 | 2.337 | 4.566 | 93 |

`nearby_only` 明显最强，说明高横向加速度场景下，附近动态对象或局部交互信息对预测更有价值。`no_ego_road` 最弱，可能是去掉 ego-road 关系后损失了关键道路约束。

值得注意的是 `ego_only` 的 `best_val_epoch=100`，最终指标和最佳指标几乎重合，说明该配置仍在稳定收敛，没有明显过拟合。

### 6.2 high_magnitude_speed

| 邻接配置 | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 3.717 | 7.583 | 2.588 | **5.394** | 15 |
| ego_only | 3.671 | 7.700 | 3.006 | 6.379 | 26 |
| nearby_only | **2.831** | **5.917** | **2.526** | 5.479 | 25 |
| no_ego_road | 3.318 | 6.733 | 2.922 | 5.995 | 45 |

这是整体最难场景。`nearby_only` 最终指标最好，但 `default` 在训练中曾达到非常好的最佳 FDE（5.394），只是最终 epoch 明显回退到 7.583。

这说明 `high_magnitude_speed/default` 的 `ds4500` 版本存在明显的后期退化：最佳验证点出现在第 15 个 epoch，而最终训练到第 30 个 epoch 时指标变差。因此使用该实验时应优先考虑 `best_model_default.pth`，不要直接用 final model。

另外，`high_magnitude_speed/default` 保留了三个历史规模：

| 样本数 | Epoch | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Loss |
|--------|-------|----------|----------|----------|----------|---------------|
| 450 | 64 | 3.373 | 6.901 | 2.529 | 5.600 | 0.5343 |
| 900 | 42 | 3.386 | 7.234 | 3.093 | 6.888 | 0.5793 |
| 4500 | 30 | 3.717 | 7.583 | 2.588 | 5.394 | 0.4664 |

`ds4500` 的最佳验证 loss 和最佳 FDE 更好，但最终指标更差。这提示训练后半段可能需要更强 early stopping，或者降低学习率/增加验证监控。

### 6.3 medium_magnitude_speed

| 邻接配置 | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 2.739 | 5.396 | 2.712 | 5.326 | 57 |
| ego_only | 2.734 | 5.413 | 2.684 | 5.308 | 66 |
| nearby_only | **1.447** | **3.694** | **1.345** | **3.506** | 56 |
| no_ego_road | 1.925 | 4.350 | 1.725 | 4.095 | 76 |

`nearby_only` 在中速场景有压倒性优势，相比 `default`：

- ADE 从 2.739 降到 1.447，下降约 47.2%
- FDE 从 5.396 降到 3.694，下降约 31.5%

这说明中速场景下，完整图结构不一定是收益来源；更干净的附近对象关系可能更适合当前模型学习。

`no_ego_road` 的 `best_val_loss=0.4415` 是该场景最低，但最终 ADE/FDE 不如 `nearby_only`。这再次说明 weighted loss 与 ADE/FDE 评估目标之间存在差异。

### 6.4 near_multiple_vehicles

| 邻接配置 | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | 1.552 | 2.871 | 1.514 | 2.767 | 97 |
| ego_only | **1.398** | **2.445** | **1.380** | **2.427** | 99 |
| nearby_only | 1.838 | 3.443 | 1.795 | 3.349 | 83 |
| no_ego_road | 1.623 | 2.945 | 1.623 | 2.945 | 100 |

这是整体表现最好的场景。`ego_only` 取得最优，说明该场景下自车历史轨迹已经足够强，额外加入 nearby 或完整图结构反而没有带来收益。

`nearby_only` 在这里表现最差，与它在其他场景的强势形成反差。这可能意味着 near multiple vehicles 数据集中的“多车近邻”不一定意味着强交互，或者 nearby-only 图缺少道路/ego 信息后容易误读附近对象关系。

### 6.5 traversing_intersection

| 邻接配置 | 最终 ADE | 最终 FDE | 最佳 ADE | 最佳 FDE | 最佳 Val Epoch |
|----------|----------|----------|----------|----------|----------------|
| default | **2.508** | **5.648** | **2.453** | **5.473** | 40 |
| ego_only | 2.603 | 5.969 | 2.504 | 5.712 | 65 |
| nearby_only | 3.353 | 7.190 | 3.178 | 6.678 | 48 |
| no_ego_road | 3.169 | 6.778 | 3.057 | 6.401 | 42 |

路口穿越场景最依赖完整上下文。`default` 是最终 ADE/FDE 最优配置，`ego_only` 接近但略差，`nearby_only` 和 `no_ego_road` 明显落后。

这说明路口场景不能只看近邻动态对象，也不能简单去掉 ego-road 关系。道路拓扑、车道连接、路口几何对未来 6 秒预测很重要。

---

## 7. 收敛与 early stopping 观察

从 `epochs_trained` 和 `best_val_epoch` 看，很多实验不是在最后一个 epoch 达到最佳验证 loss。

典型例子：

| 场景 | 邻接配置 | Epoch 总数 | 最佳 Val Epoch | 最终 ADE/FDE | 最佳 ADE/FDE | 观察 |
|------|----------|------------|----------------|--------------|--------------|------|
| high_magnitude_speed | default | 30 | 15 | 3.717 / 7.583 | 2.588 / 5.394 | 后半段退化明显 |
| high_magnitude_speed | ego_only | 41 | 26 | 3.671 / 7.700 | 3.006 / 6.379 | 后期退化 |
| high_magnitude_speed | nearby_only | 40 | 25 | 2.831 / 5.917 | 2.526 / 5.479 | 小幅退化 |
| medium_magnitude_speed | no_ego_road | 91 | 76 | 1.925 / 4.350 | 1.725 / 4.095 | 小幅退化 |
| traversing_intersection | default | 55 | 40 | 2.508 / 5.648 | 2.453 / 5.473 | 小幅退化 |

**结论**

这些结果更支持使用 `best_model_*.pth`，而不是 `final_model_*.pth`。尤其是高速场景，final model 与 best model 的差距较大，后续实验报告应明确引用哪个 checkpoint。

---

## 8. 关键结论

### 8.1 最优邻接配置不是固定的

当前结果不支持“某一种邻接配置全场景最优”的结论。

- `nearby_only`：整体平均最好，赢下 3 个场景
- `ego_only`：在 `near_multiple_vehicles` 上最好
- `default`：在 `traversing_intersection` 上最好
- `no_ego_road`：没有最终指标第一，但在部分场景验证 loss 很低

更合理的判断是：邻接配置应按场景类型选择，而不是全局固定。

### 8.2 nearby_only 是当前最强的平均配置

按 5 个主场景平均，`nearby_only` 达到：

- 平均 ADE：2.223
- 平均 FDE：4.737

它明显优于 `default` 的 2.539 / 5.163。这说明当前模型更容易从局部近邻关系中学习有效模式，而完整图结构可能引入了噪声或训练难度。

### 8.3 高速场景是最需要继续优化的场景

`high_magnitude_speed` 平均指标为：

- 平均 ADE：3.384
- 平均 FDE：6.983

这是所有场景中最差的。高速下末端误差更容易放大，未来需要重点看：

- 初始速度估计是否准确
- 预测是否存在系统性低估/高估速度
- long-horizon FDE 是否主要来自方向误差还是速度积分误差
- best checkpoint 是否显著优于 final checkpoint

### 8.4 路口场景需要完整上下文

`traversing_intersection` 中 `default` 最优，而 `nearby_only` 明显变差。这说明路口场景不能只依赖附近车辆，必须保留道路结构、ego-road 关系和更完整的图上下文。

### 8.5 near_multiple_vehicles 的结果反直觉但稳定

`near_multiple_vehicles` 中 `ego_only` 最优，且整体指标最低。这可能有两种解释：

1. 该数据集虽然有多车近邻，但自车未来主要由自身历史运动决定，交互影响有限
2. 当前 nearby/default 图构建方式没有有效利用多车信息，反而引入干扰

这个场景值得后续做 case study，因为它能帮助判断“多车近邻”是否真的对应强交互预测难度。

---

## 9. 建议的后续实验

1. **统一使用 best checkpoint 做正式对比**

   报告中最终 epoch 指标适合看训练收尾状态，但模型选择应优先用 `best_model_*.pth`。尤其高速场景 final 指标退化明显。

2. **按场景选择邻接配置**

   当前推荐：

   | 场景类型 | 推荐邻接配置 |
   |----------|--------------|
   | high_lateral_acceleration | nearby_only |
   | high_magnitude_speed | nearby_only |
   | medium_magnitude_speed | nearby_only |
   | near_multiple_vehicles | ego_only |
   | traversing_intersection | default |

3. **对高速场景单独做误差分解**

   建议拆分速度误差、方向误差、末端横向/纵向误差，确认 FDE 高的主要来源。

4. **检查 default 图是否存在噪声**

   default 没有在多数场景胜出，说明完整图可能没有被模型有效利用。可以检查：

   - 远距离 lane / agent 是否过多
   - edge type 是否过粗
   - road-agent 连接是否过密
   - pooling 后特征是否损失关键方向信息

5. **补充同批样本的 LSTM / IDM / GNN-Trans 对比**

   目前本报告只分析 `results*` 训练目录。LSTM 与 IDM 已有独立记录，但要做正式结论，最好在同一批样本、同一指标恢复逻辑下重新汇总。

---

## 10. 总结

这批 `results*` 目录反映的是一组比较完整的场景类型与邻接配置消融实验。整体上，`nearby_only` 是目前平均表现最好的配置，说明局部动态邻接对大多数场景最有帮助；但它不是万能解，在 `near_multiple_vehicles` 和 `traversing_intersection` 上分别被 `ego_only` 和 `default` 超过。

从场景角度看，`near_multiple_vehicles` 反而是当前最容易的场景，`high_magnitude_speed` 和 `traversing_intersection` 是主要难点。高速场景的末端误差最大，且多个实验出现 final model 比 best model 明显退化的现象；路口场景则更依赖完整图上下文，证明道路拓扑信息仍然不可替代。

下一步最有价值的工作不是继续盲目扩大训练，而是把模型选择、场景分组和邻接配置策略做得更精确：正式评估时使用 best checkpoint；针对高速场景做误差分解；针对路口场景保留 default 图；对多车近邻场景做 case study，确认它是否真的包含强交互。这样才能把这批消融实验从“结果列表”推进到“可指导下一轮模型设计的证据”。

---

## 11. 我们的模型与两个 Baseline 的详细对比

本节在主模型消融结果之外，结合现有的 `lstm_edit_record.md` 与 `idm_edit_record.md`，对 **GNN-Trans 主模型**、**LSTM baseline**、**IDM baseline** 做更直接的横向分析。

### 11.1 先说明：当前结果并不是完全同一标准

这是理解三组结果时最重要的前提。**现在手上的数字已经足够支持方向性判断，但还不是“完全统一评测协议”下的最终对比表。**

主要存在 4 类口径差异：

| 维度 | GNN-Trans 主模型 | LSTM baseline | IDM baseline | 影响 |
|------|------------------|---------------|--------------|------|
| 数据来源 | `results*` 训练目录中的训练统计 | 独立评估脚本输出 | 独立评估脚本输出 | 主模型结果直接来自训练记录，baseline 来自额外推理评估 |
| checkpoint 选择 | 报告里多数使用 `final_ade/fde`，但同时保留 `best_ade/fde` | 使用重训练后的 `trained_lstm_model.pth` 做推理评估 | 使用修复后的当前版本 `idm_predictor.py` 做推理评估 | 主模型当前表格偏向“最终 epoch 结果”，baseline 偏向“实际部署推理结果” |
| 样本规模 | 每个场景用各自训练/验证过程统计，非统一固定样本列表 | `medium_magnitude_speed` 上 100 / 500 样本显式评估 | `medium_magnitude_speed` 上 3000 样本显式评估 | 三者样本数量与具体样本集不完全一致 |
| 指标恢复链路 | 训练目录中的现成 ADE/FDE 统计 | 修复后的 `lstm_predictor.py` 恢复轨迹后计算 | 修复后的 `idm_predictor.py` 恢复轨迹后计算 | 虽然都是 ADE/FDE，但并非完全同一脚本同一批样本统一重算 |

因此，**现在的结果有“标准差异”**，但这些差异主要影响“严格数值排序的精确度”，**不会推翻模型能力层级的总体判断**。

更具体地说：

1. **主模型当前报告使用的是训练结果汇总口径**
   - `results_analysis_report.md` 里的主表，大多取自 `loss_record_*.json` 中的 `final_ade / final_fde`
   - 这更像“训练收尾状态总结”
   - 但很多实验其实 `best_model` 明显优于 `final_model`，尤其是高速场景

2. **LSTM / IDM 使用的是离线推理评估口径**
   - 它们是把 checkpoint 或预测器真正跑在 processed `.pkl` 样本上，再恢复轨迹并计算 ADE/FDE
   - 这更像“实际使用时的模型表现”

3. **样本集没有完全统一**
   - LSTM 新模型结果来自 `processed_medium_magnitude_speed` 的 100 / 500 样本
   - IDM 主要最终结果来自同场景的 3000 样本
   - 主模型结果来自对应训练目录的总结，不是同一批固定样本离线重算

4. **场景覆盖范围也不同**
   - 主模型报告覆盖 5 类场景
   - 当前已整理好的 LSTM / IDM 可靠结果，主要集中在 `medium_magnitude_speed`

**因此，当前比较最可靠的使用方式是：**
- 把它理解为“在 `medium_magnitude_speed` 上，三类模型的大致能力层级”
- 而不是把它直接当作“所有场景、完全统一协议下的最终排行榜”

### 11.2 在 `medium_magnitude_speed` 上的直接对比

为了尽量减少口径差异，本节只比较目前三条线都已有可靠结果的场景：`medium_magnitude_speed`。

| 模型 | 结果来源 | 样本数 | ADE | FDE | 备注 |
|------|----------|--------|-----|-----|------|
| GNN-Trans `default` | `results_analysis_report` 主记录 | 训练统计 | 2.739 | 5.396 | 完整图 |
| GNN-Trans `ego_only` | `results_analysis_report` 主记录 | 训练统计 | 2.734 | 5.413 | 仅自车相关图 |
| GNN-Trans `nearby_only` | `results_analysis_report` 主记录 | 训练统计 | **1.447** | **3.694** | 当前主模型最优配置 |
| GNN-Trans `no_ego_road` | `results_analysis_report` 主记录 | 训练统计 | 1.925 | 4.350 | 去掉 ego-road 连接 |
| LSTM baseline | `lstm_edit_record` | 500 | **2.19** | **6.39** | 修复后重训练版本 |
| IDM baseline | `idm_edit_record` | 3000 | **9.38** | **18.54** | 修复后完整版本 |

从这张表可以直接看到三层能力差异：

1. **主模型最优配置优于 LSTM**
   - `nearby_only` 相比 LSTM：
   - ADE 从 `2.19` 降到 `1.447`，约下降 **33.9%**
   - FDE 从 `6.39` 降到 `3.694`，约下降 **42.2%**

2. **LSTM 明显优于 IDM**
   - LSTM 相比 IDM：
   - ADE 从 `9.38` 降到 `2.19`，约下降 **76.7%**
   - FDE 从 `18.54` 降到 `6.39`，约下降 **65.5%**

3. **主模型与规则 baseline 之间存在数量级差距**
   - `nearby_only` 相比 IDM：
   - ADE 下降约 **84.6%**
   - FDE 下降约 **80.1%**

这说明：
- **IDM** 提供的是规则下界
- **LSTM** 提供的是可信的轻量学习基线
- **GNN-Trans** 才是当前真正有效利用交互结构信息的主模型

### 11.3 我们的模型和 LSTM baseline 到底差在哪里

LSTM 在修复后已经是一个可信 baseline，但它的本质仍然是**单体时序模型**。

它的主要特征是：
- 重点利用 ego 历史轨迹
- 善于学习平滑运动趋势
- 能形成稳定的短中期预测基线
- 对复杂交互、道路拓扑和图结构信息利用有限

这也是为什么 LSTM 在 `medium_magnitude_speed` 上已经能做到 `ADE ≈ 2.2m / FDE ≈ 6.4m`，说明它并不弱；但相比主模型最优配置，**末端误差 FDE 仍明显更大**。这通常意味着：

- LSTM 能较好拟合“整体运动惯性”
- 但对未来 6 秒末端位置的长期偏移控制不足
- 在需要理解近邻交互的样本上更容易累积误差

而 GNN-Trans 主模型的优势正体现在这里。报告已经表明：

- `nearby_only` 在 `medium_magnitude_speed` 上显著优于 `default`
- ADE 从 `2.739` 降到 `1.447`
- FDE 从 `5.396` 降到 `3.694`

这说明当前主模型真正受益的不是“图越复杂越好”，而是**合适的局部交互图**。与 LSTM 相比，主模型的本质增益来自：

1. **不只看 ego 自己怎么动，还看附近关键对象怎么动**
2. **把未来轨迹预测建模为交互问题，而不只是单序列外推**
3. **对末端偏差更敏感的 FDE 改善更大，说明长时预测更稳**

换句话说，**LSTM 更像“学自车惯性”，GNN-Trans 更像“学自车与邻居的相互作用”**。

### 11.4 我们的模型和 IDM baseline 到底差在哪里

IDM baseline 修复后已经从“逻辑不成立”变成了“能力有限但可信”的规则模型，但它依然属于**传统规则范式**。

它的强项是：
- 可解释
- 行为稳定
- 有明确的 lane-following 约束
- 适合作为工程下界参考

它的弱项是：
- 主要依赖手工规则与纵向跟驰逻辑
- 对复杂交互几乎没有学习能力
- 对未来多样性与场景变化适应弱
- 长时末端误差大

所以 IDM 的价值不在于“追平学习模型”，而在于回答一个问题：

> 如果只用规则运动学 + 基础车道约束，这个任务大致能做到什么水平？

从 `ADE 9.38 / FDE 18.54` 看，它已经可以作为一个**合理下界**，但和学习模型的差距是结构性的，不是简单调参就能补齐的。

因此，主模型相对 IDM 的优势，不应该只理解成“分数更低”，而应该理解成：

- 主模型学到了规则模型难以显式编码的时空关系
- 交互信息、局部上下文和数据驱动建模，才是当前性能跃升的主要来源

### 11.5 为什么我们的模型不是“全场景统一最优结构”

报告里一个很重要的结论是：**主模型虽然整体最强，但它内部也不是单一配置全场景最优。**

不同场景最优配置分别是：

| 场景类型 | 当前最优邻接配置 |
|----------|------------------|
| high_lateral_acceleration | nearby_only |
| high_magnitude_speed | nearby_only |
| medium_magnitude_speed | nearby_only |
| near_multiple_vehicles | ego_only |
| traversing_intersection | default |

这说明主模型的真正优势不是“图结构更大”，而是：

- 在合适场景下用对图结构
- 避免把无关节点、远距离 lane、过密连接当成噪声灌给模型

这点也正是它和两个 baseline 的关键区别：

- **IDM** 基本没有图结构学习能力
- **LSTM** 几乎不显式建模图结构
- **GNN-Trans** 的性能上限来自图结构，但前提是图构建方式合理

所以对主模型的正确理解不是“我们永远赢”，而是：

> 我们已经证明了交互图建模是有效方向，但还需要继续按场景选择图结构，不能把所有配置一锅端。

### 11.6 现阶段最合理的模型定位

综合现在的结果，这三类模型在项目中的定位已经比较清晰：

| 模型 | 当前定位 | 核心价值 |
|------|----------|----------|
| IDM | 规则下界 baseline | 可解释、可复现、提供传统方法下界 |
| LSTM | 轻量学习 baseline | 验证“只靠 ego 历史”能达到的水平 |
| GNN-Trans | 主模型 | 验证交互图建模对轨迹预测的增益 |

因此，现阶段最合理的结论不是“baseline 没意义”，而是：

- **IDM** 证明了规则方法的下界位置
- **LSTM** 证明了学习型单序列模型已经能把误差压到较低水平
- **GNN-Trans** 进一步证明了：当任务需要更强交互建模时，图结构模型能明显改善 ADE/FDE，尤其是 FDE

### 11.7 结论：现在的结果该怎么解读

如果只问一句“现在结果有没有不同标准”，答案是：

**有。**

主要差异在于：
- 主模型目前更多是训练结果汇总口径
- LSTM / IDM 是离线推理评估口径
- 样本规模和样本列表还没有完全统一
- checkpoint 选择也没有完全统一成“全部 best model”或“全部 final model”

但如果再问一句“这些差异会不会推翻总体判断”，答案是：

**不会。**

当前数据已经足够稳地支持以下判断：

1. 在 `medium_magnitude_speed` 上，**GNN-Trans 主模型最优配置明显优于 LSTM 与 IDM**
2. **LSTM 是可信而且有竞争力的学习基线**，但仍弱于主模型
3. **IDM 只能作为规则下界**，不具备与学习模型同级竞争力
4. 主模型的优势主要来自**合适的局部交互建模**，而不是简单堆更多图结构

### 11.8 下一步建议：如何把三者做成严格公平对比

如果后续要形成正式论文表格或最终实验结论，建议补一轮**统一评测协议**：

1. 固定同一批 `medium_magnitude_speed` 样本文件列表
2. 对 GNN-Trans、LSTM、IDM 全部用同一段 ADE/FDE 恢复与统计代码
3. 明确规定全部使用 `best checkpoint` 还是 `final checkpoint`
4. 在同一批样本上额外输出：
   - mean / median ADE
   - mean / median FDE
   - 长尾样本比例
   - 按场景子类型拆分的误差

这样才能把“方向性优势”升级为“严格可发表的对比结论”。

---

## 12. `medium_magnitude_speed` 统一公平对比执行计划

为了把 GNN-Trans、LSTM、IDM 放到**尽可能统一的评估标准**下，本项目接下来采用如下协议，只在 `medium_magnitude_speed` 场景上做正式对比。

### 12.1 对比对象

- **GNN-Trans 主模型**：`nearby_only` 配置，使用 `best_model_nearby_only.pth`
- **LSTM baseline**：使用修复后重训练得到的 `trained_lstm_model.pth`
- **IDM baseline**：使用修复后的当前版本 `idm_predictor.py`

### 12.2 样本池

统一从：

`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_data_by_type/processed_medium_magnitude_speed/*.pkl`

中抽取 **3000 个样本**。

### 12.3 样本选择方式

不直接使用“前 3000 个文件”，而是使用固定随机种子 `42` 做**分层抽样**。

分层依据：

1. **未来总位移长度**
2. **未来方向变化量**

具体流程：

- 先对所有 `.pkl` 按文件名排序
- 读取每个样本的 `GROUND_TRUTH`
- 计算：
  - `total_distance`：未来 60 步增量的总位移长度
  - `direction_change`：未来方向角变化累计量
- 分别按三分位切成 `低 / 中 / 高`
- 形成 `3 × 3 = 9` 个桶
- 按桶内样本占比抽取，最终生成一份固定 manifest

这样做的目的：

- 避免“前 3000 个文件”带来的顺序偏差
- 避免纯随机抽样导致直行样本或易样本过多
- 让 3000 样本同时覆盖短位移 / 长位移、低转向 / 高转向场景

### 12.4 统一评估协议

三种模型都必须：

- 使用同一份 manifest
- 预测 60 步未来轨迹
- 使用统一的 ADE / FDE 统计方式
- 输出统一 json 结果

需要说明的是：

- **LSTM / IDM** 将直接读取 manifest 中列出的 `.pkl` 文件进行评估
- **GNN-Trans** 为了复用现有训练期图构建逻辑，将：
  - 使用 `GraphDataset`
  - 在对应邻接配置下加载 / 生成 `processed/config_<adj>/dataset_<adj>.pt`
  - 再按 manifest 文件名映射到具体样本索引

### 12.5 资源分配

为了减少对现有任务的干扰，并并行完成统一对比，采用如下资源分配：

| 模型 | 设备 |
|------|------|
| GNN-Trans | `GPU 2` |
| LSTM | `GPU 0` |
| IDM | `CPU` |
| `GPU 1` | 保留，不占用 |

说明：

- GNN-Trans 是最重的模型，优先使用空闲度更高的 `GPU 2`
- LSTM 可独立放在 `GPU 0`
- IDM 属于规则推理，GPU 收益有限，因此放在 CPU

需要补充说明的是：

- 从当前 Codex 可实际调用到的运行环境看，`torch.cuda` 不可用，快速探针会直接报：
  - `RuntimeError: No CUDA GPUs are available`
- 因此本轮正式统一对比已临时切换为 **CPU 执行版本**，以优先保证结果可完整跑通。
- 如果后续需要严格复用宿主机 GPU，需要在能够直接访问 CUDA 设备的终端环境中重跑同一份 manifest。

### 12.6 后台执行方式

正式任务将通过 `nohup` 启动，以保证 SSH 断开后继续运行。

在实际调试后，后台执行策略已经从“单个总控脚本统一拉起三条任务”调整为“分步执行”：

1. 先单独生成正式 `3000` 样本 manifest
2. 再分别独立启动：
   - `GNN-Trans`
   - `LSTM`
   - `IDM`
3. 三条任务各写各自日志与结果文件

这样做的原因是：

- 便于判断到底卡在 manifest、GNN、LSTM 还是 IDM
- 避免总控脚本任意一步异常导致整条任务静默退出
- 更适合长时间远程后台运行与断线续查

进一步调试后，又补充了一层实际执行约束：

- 在 Codex 这层自动化环境里，直接裸用 `nohup` 启动子进程时，子进程存在被快速回收的风险。
- 因此最终采用的持久化方式是：
  - **`tmux` detached session + `nohup` 父级脚本**
- 这样既保留了 `nohup` 后台执行，也提高了断线后存活的稳定性。

计划生成的核心文件包括：

- manifest：
  - `experiments/analysis/comparisons/<run>/medium_speed_manifest_3000.json`
- GNN-Trans 结果：
  - `experiments/analysis/comparisons/<run>/gnn_nearby_only_3000.json`
- LSTM 结果：
  - `experiments/analysis/comparisons/<run>/lstm_3000.json`
- IDM 结果：
  - `experiments/analysis/comparisons/<run>/idm_3000.json`
- 日志：
  - `experiments/analysis/comparisons/<run>/master.log`
  - `experiments/analysis/comparisons/<run>/gnn_eval.log`
  - `experiments/analysis/comparisons/<run>/lstm_eval.log`
  - `experiments/analysis/comparisons/<run>/idm_eval.log`

### 12.7 当前状态

- [x] 已确认 `medium_magnitude_speed nearby_only` 的主模型 best checkpoint 路径
- [x] 已补充 LSTM / IDM manifest 评估脚本
- [x] 已补充 GNN-Trans manifest 评估脚本
- [x] 已补充 3000 样本 manifest 生成脚本
- [x] 已完成语法检查
- [x] 已完成 manifest 单独生成验证
- [x] 已完成 LSTM / IDM manifest 小样本验证
- [x] 已完成 GNN-Trans `nearby_only` 图缓存生成与小样本验证
- [x] 已定位 GPU 版本失败原因为当前执行环境 `CUDA` 不可用
- [x] 已切换为 CPU 版本正式评估方案
- [x] 已通过 `tmux + nohup` 重新启动正式任务
- [ ] 任务完成后将结果回填到本报告

已清理的失败中间结果：

- 已删除以下失败运行目录与旧日志，避免新旧结果混杂：
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_142805`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143615`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143641`
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_143752`
  - `experiments/analysis/comparisons/medium_speed_manifest_3000_test.json`
  - `experiments/analysis/comparisons/nohup_launch_20260429.log`

首轮后台任务状态：

- 首次 `nohup` 启动时间：`2026-04-29 14:28:05`
- 首次运行目录：
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_142805`
- 首次主日志：
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_142805/master.log`

调试结论：

- 首轮 `nohup` 任务在非常早的阶段退出，未进入正式评估。
- 之后拆分验证发现：
  - `generate_medium_speed_manifest.py` 可正常生成 3000 样本 manifest；
  - `run_lstm_pkl_eval.py --manifest` 可正常运行；
  - `run_idm_pkl_eval.py --manifest` 可正常运行；
  - `run_gnn_manifest_eval.py` 首次运行时会先完整生成 `processed/config_nearby_only/dataset_nearby_only.pt`，这一过程耗时较长，是之前误判“任务无进展”的主要风险点之一。

已完成的关键验证结果：

- 3000 样本 manifest 已成功生成测试版：
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_manifest_3000_test.json`
- LSTM 在统一 manifest 前 50 样本上的测试结果：
  - `ADE = 2.27 m`
  - `FDE = 6.79 m`
- IDM 在统一 manifest 前 50 样本上的测试结果：
  - `ADE = 8.83 m`
  - `FDE = 15.78 m`
- GNN-Trans 在统一 manifest 前 12 样本上的测试结果：
  - `ADE = 1.36 m`
  - `FDE = 3.43 m`
- `nearby_only` 图缓存已成功生成：
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_data_by_type/processed_medium_magnitude_speed/processed/config_nearby_only/dataset_nearby_only.pt`

说明：

- 由于 `nearby_only` 缓存已经生成完成，后续正式 3000 样本评估不再需要重复构图，GNN-Trans 启动速度会明显快于首次测试。
- 当前时间：`2026-04-29 15:23:34`
- 当前执行计划为：
  1. 生成正式 manifest
  2. 通过 `tmux + nohup` 启动正式 CPU 评估
  3. 回收三份 json，生成统一对比结论

当前正式运行信息：

- 运行目录：
  - `/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise`
- `tmux` session：
  - `medium3000_stepwise_150505`
- 主日志：
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/master.log`
- GNN 日志：
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/gnn_eval.log`
- LSTM 日志：
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/lstm_eval.log`
- IDM 日志：
  - `experiments/analysis/comparisons/medium_speed_compare_20260429_150505_stepwise/idm_eval.log`

当前已确认的实时状态：

- `GNN-Trans` 已经开始在 `3000` 样本上推进，日志中已出现进度条
- `LSTM` 与 `IDM` 子任务也已由父脚本拉起，后续继续观察日志落盘情况

- 本节后续将补充三模型最终统一对比结果。
