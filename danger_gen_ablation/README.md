# 危险场景生成 — 消融实验

本目录包含危险场景生成系统（Diffusion + GNN+Trans）的全部消融实验代码。
与根目录的主体脚本完全隔离，不影响现有代码运行。

## 消融组

| 编号 | 名称 | 对比点 | 需重新训练 |
|------|------|--------|-----------|
| A1 | `no_danger_loss` | 去掉 DangerExceedLoss，验证危险监督的必要性 | ✅ |
| A2 | `no_svo` | SVO 输入置零，验证行为意图条件的有效性 | ✅ |
| A3 | `no_gnn` | GNN 特征置零，验证图神经网络分支的贡献 | ✅ |
| A4 | `n1_inference` | 推理时 N=1（不重新训练），验证 Best-of-8 策略的价值 | ❌ |

## 基准模型

对比基准：`experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt`
- danger_weight=5.0，epoch 12，Margin Met=87.5%，Valid Rate=100%

## 文件说明

```
danger_gen_ablation/
├── train_ablation.py     训练脚本（支持 --ablation 参数驱动 A1/A2/A3）
├── eval_ablation.py      评估脚本（支持所有消融组，输出汇总对比表）
├── results/              各消融组评估输出目录
│   ├── no_danger_loss/
│   ├── no_svo/
│   ├── no_gnn/
│   ├── n1_inference/
│   └── ablation_summary.json   最终对比表
└── README.md
```

## 路径约定

所有脚本使用 `os.path.dirname(os.path.dirname(__file__))` 定位 GNN_Trans 根目录，
依赖的 `models/`、`utils/`、`config.py` 均从根目录解析，无需修改现有文件。

## 运行方式

> 工作目录：`/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans`，激活 `nuplan` 环境。

```bash
# 训练消融组（各约 50 epochs，每组约 30~60 分钟）
python danger_gen_ablation/train_ablation.py --ablation no_danger_loss
python danger_gen_ablation/train_ablation.py --ablation no_svo
python danger_gen_ablation/train_ablation.py --ablation no_gnn

# 快速调试（只用 20 个样本，跑 5 epoch）
python danger_gen_ablation/train_ablation.py --ablation no_danger_loss --max_samples 20 --epochs 5

# 评估所有组（含 A4 N=1，Full 参考值）
python danger_gen_ablation/eval_ablation.py --num_samples 40 --device cuda

# 只评估特定组
python danger_gen_ablation/eval_ablation.py --groups A1 A4 Full
```

消融模型保存到 `danger_gen_ablation/results/<name>/best_model.pt`，
评估汇总输出到 `danger_gen_ablation/results/ablation_summary.json`。

## 实现细节

| 消融组 | 实现方式 |
|--------|---------|
| A1 no_danger_loss | `DangerExceedLoss(danger_weight=0.0)` |
| A2 no_svo | 训练循环中 `svo_angle = torch.zeros_like(svo_angle)` |
| A3 no_gnn | `get_encoder_features()` 后将后 128 维清零（indices 256:384） |
| A4 N=1   | `eval_ablation.py` 使用 v3.10 模型，`num_inference_samples=1` |
