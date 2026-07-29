# 教程 03：危险场景生成

> **最后更新**: 2026-06-08（当前主线：`v7 map-conditioned DDPM prior + full guidance + E1-E6 evidence chain`）

---

## 工作目录

```bash
cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
conda activate nuplan
```

---

## 一、当前主线

这条研究线已经从“训练一个直接输出危险轨迹的模型”，转成更清晰的两层框架：

1. **先训练自然 joint future prior**：用 DDPM 学真实 `ego + key agent` 联合未来轨迹分布。
2. **再在推理期做反事实编辑**：用 SDEdit 和可微 guidance 把轨迹推向更危险但仍物理合理的 near-miss。

截至 2026-06-08，当前主线不是 6 月 4 日的 `v6.0 ego-only guidance 诊断`，而是：

- `v7_map_speed`
- `v7_map_intersection`
- `v7_nomap_speed`
- `v7_nomap_intersection`
- `sdedit_with_full_guidance_phase2`
- `compute_ego_svo_torch_timevarying`
- `eval_synergy_report.py`

一句话概括当前论文叙事：

> 危险 near-miss 不是被硬塞出来的碰撞，而是由 aggressive intent、social-force time-varying SVO、road-level physical constraint 和 kinematic realism 共同诱导出的交互结果。

---

## 二、任务定义

给定一个真实 nuPlan danger pair：

- `ego_history`
- `key_agent_history`
- 场景图特征
- 可选 `lane_polylines`
- GT joint future

生成新的 `ego + key agent` 联合未来轨迹，使结果满足：

1. 比 baseline future 更危险。
2. 尽量保持道路边界和运动学合理性。
3. 危险变化能由 SVO / social-force 机制解释。
4. 不把“直接撞上去”当作唯一目标，而是覆盖 collision 与 near-miss 的危险谱。

当前推荐报告口径是：

- **full split**：说明整体数据分布。
- **GT-anchored close subset**：用 `GT min_dist < 8m` 报告交互真正发生的样本。
- **原因**：TTC 筛出的 danger pairs 里存在远距离或弱交互样本，full split 会稀释 guidance 效果。

---

## 三、当前架构

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
            └── lane_encoder(lane_polylines)   # map prior 可选
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

当前四个关键变化：

1. `train_diffusion_gnn_trans.py` 支持 `--use_map_conditioning`。
2. `LaneEncoder` 在 map 模型中把 `lane_polylines` 融入 diffusion condition。
3. `sdedit_with_full_guidance_phase2` 非破坏式保留旧 v6.0，同时接入 full guidance。
4. `compute_ego_svo_torch_timevarying` 用逐时刻 social force 替代旧的全时域平均压力。

---

## 四、数据与表示

### 4.1 生成空间

当前 joint future 是 240 维：

```text
x0 = ego_delta_xy[60 * 2] || agent_delta_xy[60 * 2]
```

也就是：

- ego 未来 60 步 `(Δx, Δy)`
- key agent 未来 60 步 `(Δx, Δy)`
- 两者拼成一个 joint diffusion sample

### 4.2 danger pair 数据

当前稳定使用两类数据：

| 数据集 | 样本数 | 用途 |
|---|---:|---|
| `danger_pairs_speed` | 533 | 高速 / 追赶 / 近距离交互 |
| `danger_pairs_intersection` | 719 | 路口 / 复杂几何交互 |

注意：训练或评估时必须显式传入 danger pair 目录。

```text
diffusion_danger_model/dataset/danger_pairs_speed
diffusion_danger_model/dataset/danger_pairs_intersection
```

不要误用默认 danger pair 路径。6 月 5 日日志已经确认默认路径里 intersection 样本只有 2 个，容易造成假实验。

### 4.3 lane 类型修复

6 月 5 日已修复一个关键数据 bug：

- 旧逻辑误把 `pf[:, 9]` 当作 lane type。
- 实际上 `pf[:, 9]` 是 `subgraph_id`。
- 当前应通过 `SUBGRAPH_ID_TO_TYPE` 映射恢复 road / lane 语义。

这个修复直接影响 map-conditioned prior 和 road guidance 的可信度。

---

## 五、v7 训练状态

v7 当前有四个完整 checkpoint：

| checkpoint | 数据集 | map conditioning | epoch100 loss | 状态 |
|---|---|---:|---:|---|
| `v7_map_speed` | speed | yes | 0.226 | 完成 |
| `v7_map_intersection` | intersection | yes | 0.208 | 完成 |
| `v7_nomap_speed` | speed | no | 0.213 | 完成 |
| `v7_nomap_intersection` | intersection | no | 0.233 | 完成 |

训练时间已经被确认比最初预估短很多：

- GPU 训练约 23 秒 / epoch。
- 100 epoch 约 38-40 分钟。
- 曾出现一次 host RAM / DataLoader 并发导致的 OOM，后来单独重跑 `v7_map_intersection` 已完成。

map prior 的结论不能简单说“永远更好”：

- intersection 几何复杂，map prior 明显降低 baseline offroad。
- speed 场景几何更简单，map prior 冗余甚至略负。

因此论文口径应是：

> Map prior 是复杂几何下的 enabling component，不是单独制造 danger 的来源。

---

## 六、SVO 与 social force 机制

当前核心贡献是 **time-varying SVO**。

旧版本 SVO 的问题是：social force 被压成一个全时域平均量，无法解释“远距离时不必避让、近距离时突然让行”的动态行为。

新版本在 `src/shared/utils/svo_torch.py` 中实现：

```text
force[t] = relu(8 - dist[t]) / 8
```

直观解释：

- 两车距离远时，social force 近似 0。
- 两车进入 8m 内，social force 上升。
- ego 的 SVO 随近距离压力变得更 prosocial。
- 这会形成“先 aggressive approach，后 near-zone avoidance”的 near-miss 机制。

6 月 7 日 S2 机制验证给出了具体样本：

| 最近距离 | far SVO | near SVO | 解释 |
|---:|---:|---:|---|
| 0.16m | 约 1-2.6° | 4.10° | 近距离后开始让行 |
| 0.19m | 约 1-2.6° | 5.43° | social force 升高 |
| 0.42m | 约 1-2.6° | 7.58° | near-miss 机制明显 |

当前 aggressive target 使用固定初始承诺：

```text
target = baseline_svo - relative_delta_deg
relative_delta_deg = 15
```

这对应论文叙事中的 aggressive intent。

---

## 七、推理：SDEdit + full guidance

当前主入口是：

```text
scripts/eval_v5_svo.py
mode = full
```

对应内部实现：

```text
sdedit_with_full_guidance_phase2
```

full guidance 包含四类约束：

| 组件 | 作用 |
|---|---|
| symmetric time-varying SVO | ego 和 agent 都进入交互约束，不再只推 ego |
| soft-min proximity | 鼓励形成可控近距离交互 |
| road-level `L_lane` | 约束 road/on-road，不再把正常换道误当失败 |
| kinematic + lowpass | 控制加速度和轨迹平滑性 |

6 月 7 日选定的物理工作点：

```text
lambda_kinematic = 10.0
lowpass_cutoff_ratio = 0.15
```

这个工作点的作用：

- 把平均加速度从约 12 降到约 9.3，低于 1g。
- danger 基本不下降。
- collision / near-miss 强度仍接近 GT 危险谱。

注意：jerk 当前不适合作为主 realism 指标。GT 轨迹本身由于 `dt=0.1` 有有限差分噪声，jerk 已经偏高。当前主报告指标应优先使用 accel。

---

## 八、评估协议

### 8.1 当前脚本

| 脚本 | 用途 |
|---|---|
| `scripts/eval_v5_svo.py` | baseline / full guidance 评估入口 |
| `scripts/analysis/eval_synergy_report.py` | 统一输出 GT、baseline、full 的 full split 与 close subset 指标 |
| `scripts/analysis/viz_near_miss.py` | 生成 near-miss 机制可视化 |

### 8.2 当前主指标

| 指标 | 用途 |
|---|---|
| `danger` | 危险强度主指标 |
| `collision_rate` | 危险谱中更激烈一端 |
| `min_distance` | 是否形成近距离交互 |
| `offroad` | 道路保持 |
| `accel` | 物理合理性主指标 |
| `ego_svo` | 行为语义解释 |

### 8.3 offroad 口径

需要同时说明两个口径：

- `offroad @1.75m`：严格 lane-level 偏离，容易把正常换道算失败。
- `offroad @3m`：更接近当前 road-level soft boundary 设计。

当前方法目标是 road keeping，不是禁止换道。

---

## 九、当前关键结果

### 9.1 S2 speed split：full guidance 是否有效

`v7_map_speed`，B=63，road 3m + time-varying SVO + proximity + kinematic。

| 方法 | danger | offroad | egoSVO | minDist | collide |
|---|---:|---:|---:|---:|---:|
| GT | 2.96 | 0.0% | 11.8° | 21.5m | 27.0% |
| baseline | 0.80 | 53.6% | 8.2° | 23.9m | 3.2% |
| full(timevar) | 2.19 | 8.0% | 13.5° | 21.2m | 19.0% |

结论：

- danger 明显上升。
- offroad 大幅下降。
- collision rate 朝 GT 危险谱靠近。
- time-varying SVO 能解释 near-miss 行为变化。

### 9.2 intersection close subset：复杂几何下是否成立

| 方法 | danger | offroad | egoSVO | minDist | collide |
|---|---:|---:|---:|---:|---:|
| GT | 3.097 | 1.8% | 5.7° | 3.68m | 40.5% |
| baseline | 1.763 | 35.2% | 5.2° | 6.18m | 25.7% |
| full(timevar) | 2.407 | 8.7% | 10.7° | 5.33m | 35.1% |

结论：

- close subset 中 full guidance 对 speed 和 intersection 都有效。
- full split 效果被远距离或弱交互样本稀释，因此论文要解释 close subset 的必要性。

### 9.3 E2 map prior 结论

baseline offroad 对比：

| 数据集 | map | nomap | 结论 |
|---|---:|---:|---|
| speed all | 53.4% | 49.1% | map 略差 |
| speed close | 58.0% | 48.3% | map 略差 |
| intersection all | 40.8% | 56.1% | map 明显更好 |
| intersection close | 37.8% | 55.5% | map 明显更好 |

结论：

- map prior 对 intersection 有显著价值。
- map prior 对 speed 冗余或略负。
- road guidance 接入后，map / nomap 的 offroad 都会降低，因此 map 的边际价值会被 guidance 掩盖。

### 9.4 E5 统计显著性

两条核心 claim 已有统计支持：

| claim | 结果 | 统计 |
|---|---|---|
| full guidance 提升 danger | baseline 1.648 -> full 5.136 | Wilcoxon p=1.74e-09，bootstrap diff 95% CI [2.556, 4.323] |
| map prior 降低 intersection baseline offroad | map median 33.3% vs nomap median 65.0% | Wilcoxon p=2.53e-09，median -31.7pt |

---

## 十、当前论文主张

当前最稳的论文结构是：

1. **方法主张**：用 map-conditioned DDPM prior 学真实 joint trajectory 分布。
2. **机制主张**：用 social-force time-varying SVO 表达 “aggressive approach + near-zone avoidance”。
3. **推理主张**：用 full guidance 在物理约束下生成危险 near-miss。
4. **实验证据**：full guidance 显著提高 danger，map prior 显著改善 intersection offroad。

更具体地说：

- danger 不是来自 map。
- danger 主要来自 SVO / social-force guidance。
- map 是复杂道路几何下减少 offroad 的 enabling component。
- kinematic / lowpass 是让轨迹物理合理的约束层。

---

## 十一、已知问题与 TODO List

TODO List:

- 量化 degenerate danger pair 的比例，尤其是 “min gap 只发生在 t=0” 或 “ego 几乎静止” 的样本。
- 检查并修复 `ttc_filter` 中 ego speed 使用瞬时速度而不是历史均值的问题。
- 设计更强的 danger pair 筛选条件：真实接近过程、mid-trajectory min distance、双方都在运动。
- 决定是否需要重新筛选 danger pairs 并重训 v7 prior。
- 报告 offroad 时同时给出 `@1.75m` 与 `@3m`，避免把正常换道误解释为道路失败。
- 继续处理 accel p95 约 21 的长尾物理合理性问题。
- 把 E1-E6 evidence chain 整理成论文实验章节。

---

## 十二、快速入口

### 12.1 训练入口

```text
train_diffusion_gnn_trans.py
--use_map_conditioning
--danger_pair_dir diffusion_danger_model/dataset/danger_pairs_speed
--danger_pair_dir diffusion_danger_model/dataset/danger_pairs_intersection
```

当前训练主线是 v7 四组 checkpoint。继续实验前先确认 checkpoint 与 danger pair 目录匹配。

### 12.2 评估入口

```text
scripts/eval_v5_svo.py
scripts/analysis/eval_synergy_report.py
```

推荐使用 `eval_synergy_report.py` 统一生成：

- GT
- baseline
- full guidance
- full split
- close subset
- JSON 结果

### 12.3 可视化入口

```text
scripts/analysis/viz_near_miss.py
docs/figures/near_miss/
```

当前可视化已经支持 speed / intersection representative near-miss，用于展示：

- aggressive approach
- distance V-shape
- near-zone SVO rise
- late avoidance

---

## 十三、一句话记忆

> 当前不是 v6.0 的 “ego-only guidance 还能不能调好”，而是 v7 prior + full guidance 已经形成一条可写论文的证据链；下一步重点是清理 danger pair 质量、统一 offroad/accel 口径，并把 E1-E6 写成方法与实验章节。
