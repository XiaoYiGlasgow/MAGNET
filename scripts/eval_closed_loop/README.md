# 官方 nuPlan 闭环评测 harness（重建版）

> 建立：2026-07-28。归属：VT-2026-04191 MAGNET 重投，阶段⑤/⑧ 的闭环评测链路重建。

## 为什么重建

阶段⑤/⑧ 的闭环 harness 全部放在会话级 `scratchpad/` 下，现已丢失：

| 丢失资产 | 原用途 |
|---|---|
| `eval_matrix.sh` / `eval_gnn_pgh.sh` / `eval_baselines_pgh.sh` | 3 planner × 2 模式矩阵驱动 |
| `gpu_watcher.sh`（v3 Bark） | 等独占卡自动触发 |
| `extract_scores.py` / `bark_summary.py` | 从 aggregator parquet 提分 |
| `boston_clean/`（300 symlink）、`pittsburgh_clean/`（300 symlink） | 深筛过的评测 DB 子集 |
| `eval_full/`、`eval_full_pgh/` | 全部原始 simulation 产物 |
| `step0_heldout_pgh.py` / `pittsburgh_validate.py` | held-out 池构造与深筛 |

**后果**：Boston-250 的 `0.122/0.095` 与阶段⑧ 的 Pittsburgh 基线 `IDM 0.591/0.564`、
`LogFuture 0.815/0.543` 目前**只有阶段记录文本，没有可复现产物**。这是 D0 provenance
（阶段15 §3.1 #5）的实质缺口：这些数字在重投稿里可以引用，但**不能声称可复现**，
除非用本目录的 harness 重跑。

本目录是可审计的重建版，放在版本可追踪路径下，不再放 `scratchpad/`。

## 文件

| 文件 | 作用 |
|---|---|
| `build_pittsburgh_heldout.py` | 重建阶段⑧ 的 in-dist 评测池：`train_pittsburgh` 1560 DB − E1 用过的 600 log = 960 held-out（含零重叠硬断言），深筛后取 300 clean |
| `run_closed_loop.sh` | 单个 (场景集 × planner × 模式) 的 hydra 调用，全绝对路径，每 run 落 provenance JSON |
| `extract_scores.py` | 读 `aggregator_metric/*weighted_average*.parquet` → final score + 6 子指标 + 按 `scenario_type` 分解 |
| `queue_pittsburgh_indist.sh` | GPU 队列排程：轮询三卡，第一张连续 3 次近乎独占（used < 4000 MiB）即跑整批 6 格，完成后自动提分并 Bark |
| `logs_pittsburgh_heldout.txt` | 960 个 held-out log 名（`--build` 生成） |
| `logs_pittsburgh_heldout_clean.txt` | 300 个深筛通过的 log 名（`--validate` 生成） |
| `logs_pittsburgh_{val,test}60.txt` | split3 的 val/test log 名，**当前闭环不用**，见下方"场景集选择" |

## 场景集选择（重要）

| 场景集 | 定义 | 用途 |
|---|---|---|
| `pittsburgh_heldout` ✅ **当前使用** | 960 held-out 深筛后的 300 clean DB，**与 E1 的 600 log 零重叠** | in-dist 闭环对照，沿用阶段⑧ 防泄漏红线 |
| `pittsburgh_val` ❌ 不用 | split3 的 60 val log | val 参与过 checkpoint 选择，闭环诊断上属轻度泄漏 |
| `pittsburgh_test` ❌ 不用 | split3 的 60 test log | R4-7 锁定集，只在方法冻结后一次性使用，不得用于诊断 |
| `boston` ⚠️ 待重建 | 跨城 held-out | `boston_clean/` 已丢失，重跑前须先按同样方式深筛 `train_boston` |

## 复现命令

```bash
# 1. 重建 in-dist 评测池（CPU，秒级）
PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit \
  /home/team/anaconda3/envs/nuplan/bin/python \
  /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/build_pittsburgh_heldout.py \
  --build --validate --limit 300 --workers 8

# 2. 排队等空闲 GPU 后自动跑整批（跨会话存活，勿用 pkill -f 杀）
SCENARIO_SET=pittsburgh_heldout \
GROUP=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/closed_loop/pittsburgh_heldout_N20 \
setsid bash /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/queue_pittsburgh_indist.sh &

# 3. 单格手动跑
bash /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/run_closed_loop.sh \
  --scenario-set pittsburgh_heldout --planner gnn_trans --mode nonreactive \
  --n 20 --device cuda --cuda 0 --group <group_dir>

# 4. 提分
/home/team/anaconda3/envs/nuplan/bin/python \
  /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/extract_scores.py <group_dir>
```

## 已验证

- `logs_pittsburgh_heldout.txt`：960 条，与 E1 600 log 零重叠（硬断言通过），**与阶段⑧ 的 960 一致**；
- 深筛：`clean=300`、`db_error=0`，**与阶段⑧ 的「300 clean、deep_bad=0」一致**；
  另有 412 个 log 无 challenge 类型场景 —— 这是正常过滤，不是 DB 损坏，两者在日志中分列；
- hydra 配置解析：`--cfg job --resolve` 已确认 `data_root` / `db_files` / `num_scenarios_per_type` /
  `planner.gnn_trans_planner.checkpoint_path` 全部按预期落位；
- checkpoint 与 Boston-250 完全相同：`experiments/magnet_imitation_e1/seed2_gpu2/models/best_model_default.pth`。

## 边界

- 本 harness 只做评测，不训练、不改 planner 逻辑、不碰 danger 线；
- `pittsburgh_heldout` 场景分布高速主导（阶段⑧ 实测 `high_magnitude_speed` ≈ 63%），
  与 Boston 池形态不同 → 跨场景集比较必须同时看 per-`scenario_type` 分解，不能只比总分；
- Pittsburgh 有 14 类而 Boston 只有 13 类（`traversing_pickup_dropoff` 天然缺）→
  两集对比总分前须限制到共同 13 类，否则等权聚合口径不一致。
