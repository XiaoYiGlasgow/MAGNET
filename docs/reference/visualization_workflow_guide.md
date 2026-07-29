# 可视化排错与代码位置速查

> 完整使用命令见 [tutorials/03_danger_scene_generation.md](tutorials/03_danger_scene_generation.md)，
> 本文档只记录排错要点和关键代码位置。

---

## 常见问题排查

### `No module named torch` / `No module named transformer_predictor`

未激活正确环境，或在错误目录下运行：

```bash
conda activate nuplan
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
```

### `Processed graph file not found`

使用 `gnn_trans_encoder` checkpoint 时，需要对应的 `processed_data/features_*.pkl`：

```bash
ls diffusion_danger_model/processed_data/ | wc -l   # 应为 804
```

若为空，需先运行 `diffusion_danger_model/process_pittsburgh.py` 重新预处理。

### Valid Rate 为 0%

模型版本与代码不匹配。v3.9 之前的模型不支持运动学参数化输出，需换用当前最佳模型：

```bash
# 始终使用这个路径
experiments/danger_generation/training_runs/full_training_v3_9/best_model.pt
```

### `generate() got an unexpected keyword argument` / JSON 序列化报错

这些 bug 在当前代码中已修复，如仍报错说明 import 了旧版缓存模块，重启 Python 进程即可。

### `Warning: nuPlan imports not available`

非阻塞警告，`quick` / `bokeh` / `stats` / `full` 四种模式均不依赖 NuPlan；
只有 `--mode nuboard` 才需要 NuPlan 环境，可忽略此警告。

### CUDA out of memory

降低推理批量，在 `run_evaluation.py` 中减小 `--num_inference_samples`（默认 8，改为 1 或 2）：

```bash
python visualization/run_evaluation.py --mode stats \
    --num_inference_samples 2 ...
```

### 首次运行建议

先用 `--num_samples 1` 做单样本冒烟，确认输出正常后再扩大：

```bash
python visualization/run_evaluation.py --mode quick --num_samples 1 ...
```

---

## 关键代码位置

| 功能 | 文件 |
|------|------|
| 评估统一入口 | `src/danger_generation/visualization/run_evaluation.py` |
| Quick / Stats 可视化 | `src/danger_generation/visualization/matplotlib_visualizer.py` |
| Bokeh / NuBoard 可视化 | `src/danger_generation/visualization/nuboard_visualizer.py` |
| 轨迹格式转换（运动学解码）| `src/danger_generation/visualization/trajectory_converter.py` |
| 条件编码器 | `src/danger_generation/models/scene_condition_encoder.py` |
| Diffusion 主模型 | `src/danger_generation/models/danger_diffusion.py` |
| 危险分计算（numpy）| `src/shared/utils/danger_score.py` |
| 危险分计算（torch，训练用）| `src/shared/utils/danger_score_torch.py` |
| GNN+Trans 编码器 | `/mnt/local/Xiao_Yi/nuplan-devkit/models/transformer_predictor.py` |
