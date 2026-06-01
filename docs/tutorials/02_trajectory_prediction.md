# 教程 02：轨迹预测

本教程介绍如何训练、评估 GNN+Transformer 轨迹预测模型，并在 NuPlan 仿真器中运行。

---

## 工作目录说明

```bash
# 大多数训练/评估命令在 GNN_Trans 根目录执行
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
conda activate nuplan

# NuPlan 仿真命令在 nuplan-devkit 根目录执行（有专门说明）
cd /mnt/local/Xiao_Yi/nuplan-devkit
```

---

## 概览：三种模型

| 模型 | 入口脚本 | 特点 | 建议用途 |
|------|----------|------|----------|
| **GNN + Transformer** | `train.py` | 完整图神经网络，精度最高 | 主模型 |
| **Transformer Only** | `train_transformer_only.py` | 去掉 GNN，只用 Transformer | 消融实验 |
| **LSTM** | `train_lstm.py` | 经典序列模型 | 基线对比 |

已有训练好的 GNN+Transformer 权重：`data/model_weights/best_model.pth`，**可以跳过训练直接评估**。

---

## Step 1：确认训练数据

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
ls data/processed_data_by_type/
```

输出：
```
processed_high_lateral_acceleration     ← 高横向加速度场景
processed_high_magnitude_speed          ← 高速场景
processed_medium_magnitude_speed        ← 中速场景（默认训练集）
processed_near_multiple_vehicles        ← 多车交互场景
processed_traversing_intersection       ← 路口场景
```

查看默认训练集的样本数：
```bash
ls data/processed_data_by_type/processed_medium_magnitude_speed/ | wc -l
# 输出：20002（约 2 万个场景）
```

每个 `.pkl` 文件是一个独立的驾驶场景，包含 ego 车和周围最多 15 辆车的历史轨迹（过去 4 秒）及地图特征。

---

## Step 2：（可选）修改训练配置

打开 `config.py`，按需修改以下参数：

```python
# 换场景类型（默认 medium_magnitude_speed）
TRAIN_PATH = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")
TEST_PATH  = os.path.join(PROCESSED_DATA_BY_TYPE_DIR, "processed_medium_magnitude_speed")

# 训练超参数
EPOCHS        = 100     # 训练轮数（从头训建议 50-100）
BATCH_SIZE    = 8       # 显存不够就改成 4
LEARNING_RATE = 1e-3    # 初始学习率

# 模型结构（一般不用改）
EMBED_DIM  = 128
NUM_LAYERS = 3
NUM_HEADS  = 8
```

---

## Step 3：训练 GNN+Transformer 主模型

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
python train.py
```

启动后会出现交互式菜单，选择邻接矩阵配置：

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

**直接回车**使用默认配置 1（Full connectivity），适合大多数情况。

训练过程中的输出示例：
```
Epoch [1/100] | Train Loss: 2.3451 | Val Loss: 2.1203 | ADE: 1.8432 | FDE: 3.2105
Epoch [2/100] | Train Loss: 1.9823 | Val Loss: 1.8341 | ADE: 1.6201 | FDE: 2.8934
...
```

**ADE**（Average Displacement Error）和 **FDE**（Final Displacement Error）越低越好，单位是米。

训练完成后，权重自动保存到：
```
data/model_weights/
└── best_model_<场景类型>_<邻接配置>.pth    ← 验证集最佳
└── final_model_<场景类型>_<邻接配置>.pth   ← 最终 epoch
```

训练记录保存到：
```
experiments/analysis/training_records/<实验名>/
```

---

## Step 4：训练 Transformer-Only 基线（消融实验用）

```bash
python train_transformer_only.py \
    --train_path data/processed_data_by_type/processed_medium_magnitude_speed \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --device cuda:0
```

权重保存到：
```
experiments/baselines/transformer_only/
└── best_model_transformer_only.pth
```

---

## Step 5：训练 LSTM 基线

```bash
python train_lstm.py
```

同样是交互式启动，直接回车用默认配置即可。权重保存到：
```
experiments/baselines/lstm/
```

---

## Step 6：NuPlan 规划仿真（最直观的端到端测试）

这个测试把训练好的 GNN+Transformer 当作 NuPlan 规划器运行，验证模型在真实仿真环境中是否正常工作。

> **注意**：这个命令必须在 `nuplan-devkit` 根目录执行，不是 GNN_Trans 目录。

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
python GNN_Trans/run_nuboard_simulation.py --no-nuboard
```

成功输出：
```
...
Number of successful simulations: 2
Number of failed simulations: 0
Finished running simulation!
```

### 可视化仿真结果（NuBoard）

如果想在浏览器里看仿真轨迹的可视化，去掉 `--no-nuboard`：

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
python GNN_Trans/run_nuboard_simulation.py
```

然后在浏览器访问 `http://localhost:5006`，可以看到 ego 车的轨迹和周围车辆。

---

## Step 7：评估模型性能（ADE/FDE 指标）

`run_gnn_manifest_eval.py` 在指定测试集上批量计算 ADE/FDE。

首先需要准备一个 manifest JSON 文件，记录测试样本列表：

```bash
# 查看是否已有 manifest 文件
find experiments/ -name "*.json" | head -10
find data/ -name "manifest*.json" | head -5
```

如果没有，可以快速生成一个（用 Python 脚本列出测试目录下的 pkl 文件）：

```python
# 在 GNN_Trans 目录下运行这段代码生成 manifest
import os, json
data_dir = "data/processed_data_by_type/processed_medium_magnitude_speed"
files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
# 取后 10% 作为测试集
test_files = files[int(len(files)*0.9):]
manifest = {"sample_files": test_files}
with open("data/test_manifest.json", "w") as f:
    json.dump(manifest, f)
print(f"生成 manifest，共 {len(test_files)} 个测试样本")
```

然后运行评估：

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

参数说明：

| 参数 | 说明 |
|------|------|
| `--model_path` | 模型权重路径 |
| `--data_dir` | 测试数据目录（pkl 文件所在目录）|
| `--manifest` | 测试样本列表 JSON |
| `--adjacency_config` | 邻接矩阵配置（与训练时保持一致）|
| `--device` | `cpu` 或 `cuda:0` |
| `--output_json` | 结果保存路径（可选）|

---

## Step 8：评估 LSTM 和 IDM 基线

```bash
# 评估 LSTM
python run_lstm_pkl_eval.py

# 评估 IDM（智能驾驶模型，基于规则）
python run_idm_pkl_eval.py
```

---

## 常见问题

### Q: train.py 报 `GraphDataset` import 失败

```
ModuleNotFoundError: No module named 'dataset'
```

原因：没有在正确的 conda 环境下运行，或 `torch_geometric` 未安装。

解决：
```bash
conda activate nuplan
python -c "from torch_geometric.data import Data; print('OK')"
```

### Q: 训练很慢

- 检查是否用了 GPU：`python -c "import torch; print(torch.cuda.is_available())"`
- `config.py` 里的 `BATCH_SIZE` 过小会导致 GPU 利用率低，适当增大

### Q: 训练 loss 不收敛

- 降低 `LEARNING_RATE`（从 `1e-3` 改为 `5e-4` 或 `1e-4`）
- 检查数据路径是否正确：`ls data/processed_data_by_type/processed_medium_magnitude_speed/ | head -5`

### Q: NuPlan 仿真报 Hydra 配置错误

确认从 `nuplan-devkit` 根目录（而非 `GNN_Trans` 目录）运行：
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit  # ← 必须是这个目录
python GNN_Trans/run_nuboard_simulation.py --no-nuboard
```

---

## 参考：各脚本对应的输出目录

| 脚本 | 权重/结果保存位置 |
|------|-----------------|
| `train.py` | `data/model_weights/` + `experiments/analysis/training_records/` |
| `train_transformer_only.py` | `experiments/baselines/transformer_only/` |
| `train_lstm.py` | `experiments/baselines/lstm/` |
| `run_gnn_manifest_eval.py` | `--output_json` 指定路径 |
| `run_nuboard_simulation.py` | `/mnt/local/Xiao_Yi/gnn_trans_simulation_results/` |
