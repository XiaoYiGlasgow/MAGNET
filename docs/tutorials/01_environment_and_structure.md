# 教程 01：环境配置与项目结构

## 1. 运行环境

### Conda 环境

本项目所有命令必须在 `nuplan` 环境下运行：

```bash
conda activate nuplan
```

验证环境是否正确：

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

正常输出示例：
```
PyTorch: 2.x.x
CUDA: True
```

### 工作目录约定

**轨迹预测**相关命令，在 **nuplan-devkit 根目录** 运行（NuPlan Hydra 仿真需要）：
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit
```

**危险场景生成**相关命令，在 **GNN_Trans 根目录** 运行：
```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
```

> **记住这个区别**，大多数报路径错误的问题都是工作目录搞错了。

---

## 2. 项目目录结构

```
GNN_Trans/
│
├── config.py                        ← 全局配置（路径、超参数）★ 最常修改
│
│   【轨迹预测入口】
├── train.py                         ← GNN+Transformer 轨迹预测训练
├── train_transformer_only.py        ← Transformer-only 基线训练
├── train_lstm.py                    ← LSTM 基线训练
├── run_gnn_manifest_eval.py         ← GNN+Trans 评估
├── run_nuboard_simulation.py        ← NuPlan 规划仿真
├── idm_predictor.py / lstm_predictor.py  ← 模型推理模块
│
│   【危险场景生成入口】
├── train_diffusion_gnn_trans.py     ← Diffusion 训练（当前主入口）★
├── generate_danger_scenes.py        ← 批量生成危险场景（推理）★
├── run_full_training_DEPRECATED.py  ← 旧版训练脚本（v3.x，已废弃）
│
├── danger_gen_ablation/             ← 消融实验（危险场景生成）
│   ├── README.md                    ← 消融实验说明
│   ├── train_ablation.py            ← 消融训练（待写）
│   ├── eval_ablation.py             ← 消融评估（待写）
│   └── results/                     ← 各消融组评估输出
│
├── configs/                         ← NuPlan Hydra 配置文件
│   ├── simple_simulation.yaml       ← 仿真入口配置
│   └── planner/                     ← planner 配置
│
├── data/                            ← 所有数据（不进 Git）
│   ├── model_weights/               ← GNN+Trans 轨迹预测权重
│   │   └── best_model.pth           ★
│   ├── processed_data_by_type/      ← 按场景类型分类的训练数据
│   │   └── processed_medium_magnitude_speed/  ← 默认训练集（20002 个场景）
│   └── ...
│
├── diffusion_danger_model/          ← 危险场景生成专用数据
│   ├── processed_data/              ← Diffusion 训练数据（803 个 pkl）★
│   ├── danger_pair_scenarios/       ← 危险对场景（带标签，390 个）★
│   └── ttc_filter_results/
│
├── experiments/                     ← 所有实验结果（不进 Git）
│   ├── danger_generation/
│   │   ├── training_runs/
│   │   │   ├── full_training_v3_10/ ← 当前最优模型 ★（Margin Met 87.5%）
│   │   │   │   └── best_model.pt
│   │   │   └── full_training_v3_1 ~ v3_11/  ← 历史版本
│   │   └── ablation/               ← 消融实验结果（运行后生成）
│   ├── analysis/gnn_trans_ablation/ ← GNN+Trans 轨迹预测消融
│   └── baselines/                  ← Transformer-only / LSTM 训练记录
│
├── src/                             ← 库代码（两条研究线共用）
│   ├── danger_generation/
│   │   ├── models/                  ← danger_diffusion.py, scene_condition_encoder.py
│   │   └── visualization/           ← run_evaluation.py ★
│   ├── ego_planning/
│   │   ├── planner/                 ← GNNTransPlanner
│   │   └── nuboard/
│   └── shared/utils/                ← danger_score.py, dataset.py 等
│
└── docs/
    ├── DANGER_GENERATION.md                 ← 危险场景生成研究线全部入口导航 ★
    ├── TRAJECTORY_PREDICTION.md             ← 轨迹预测研究线全部入口导航 ★
    ├── reference/                           ← 技术说明与排错文档
    ├── reports/                             ← 结果总结、汇报稿、技术报告
    └── tutorials/                           ← 你现在在这里
```

> `utils/`, `models/`, `visualization/` 等根目录下的目录均为 **symlink**，
> 指向 `src/` 下真实位置，不需要手动处理。
> 两条研究线的入口说明现在统一放在 `docs/` 根下，按研究线查阅 `docs/DANGER_GENERATION.md` 或 `docs/TRAJECTORY_PREDICTION.md`。

---

## 3. 关键配置文件详解

### `config.py` — 全局参数（最常需要修改）

```python
# 数据路径（一般不需要改，除非挂载点变了）
DATA_PATH   = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
MAP_PATH    = '/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps'

# 训练数据路径 ← 换场景类型时改这里
TRAIN_PATH  = ".../data/processed_data_by_type/processed_medium_magnitude_speed"
TEST_PATH   = ".../data/processed_data_by_type/processed_medium_magnitude_speed"

# 训练超参数
EPOCHS      = 100
BATCH_SIZE  = 8
LEARNING_RATE = 1e-3

# 模型架构
EMBED_DIM   = 128      # Transformer 嵌入维度
NUM_LAYERS  = 3        # Transformer 层数
NUM_HEADS   = 8        # 多头注意力头数
```

修改 `config.py` 后直接运行对应训练脚本即可生效，不需要命令行参数。

### `train_diffusion_gnn_trans.py` — Diffusion 训练配置（推荐用 CLI）

当前训练入口不再改脚本内的 `CUSTOM_CONFIG`，而是直接传命令行参数：

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

python train_diffusion_gnn_trans.py \
    --output_dir experiments/danger_generation/training_runs/v4_0_debug \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --device cuda
```

如果只想检查数据和前向流程，不正式训练，可以加：

```bash
python train_diffusion_gnn_trans.py --test_only
```

---

## 4. 两条研究线的数据流向

### 轨迹预测

```
NuPlan 原始数据库 (.db 文件)
    │
    ↓ scripts/preprocessing/process_nuplan_to_12d.py
    │  （提取场景图特征，输出 .pkl 文件）
    │
data/processed_data_by_type/<场景类型>/   ← 20002 个 .pkl
    │
    ↓ train.py
    │  （GNN 编码 + Transformer 预测）
    │
data/model_weights/best_model.pth          ← 保存的权重
    │
    ↓ run_nuboard_simulation.py / run_gnn_manifest_eval.py
    │
仿真结果 / ADE·FDE 评估指标
```

> **好消息**：`processed_data_by_type/` 和 `model_weights/best_model.pth` 已经存在，
> 可以直接跳到训练或评估步骤。

### 危险场景生成

```
NuPlan 原始数据库
    │
    ↓ diffusion_danger_model/process_pittsburgh.py
    │  （提取 12D 场景特征）
    │
diffusion_danger_model/processed_data/      ← 804 个 .pkl（已有）
    │
    ↓ utils/ttc_filter.py
    │  （TTC 筛选，找出存在碰撞风险的配对）
    │
diffusion_danger_model/danger_pair_scenarios/ ← 81 个 .pkl（已有）
    │
    ↓ train_diffusion_gnn_trans.py
    │  （DDPM Diffusion 训练，v4.0：TrajectoryPredictionHead + 动态 danger_target）
    │
experiments/.../full_training_v3_10/best_model.pt ← 当前最优（已有，Margin Met 87.5%）
    │
    ├─→ visualization/run_evaluation.py   ← 统计评估
    │
    └─→ generate_danger_scenes.py         ← 批量生成新场景
```

> **好消息**：所有数据和模型都已准备好，可以直接运行评估和生成。
