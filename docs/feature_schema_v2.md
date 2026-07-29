# Feature Schema v2 — 冻结记录

> 冻结日期：2026-07-28 ｜ 归属：VT-2026-04191 阶段15 D0#4 / Z7
> **schema hash：`31bce53e29f9f0bd`**

---

## 0. 唯一真相源

本文档**不是** schema 的定义，只是冻结记录。定义在代码里：

```
scripts/analysis/result_manifest.py :: FEATURE_SCHEMA_V2
```

改那个字典，hash 就会变，所有引用旧 hash 的结果 JSON 会在
`validate_manifest()` 里被自动标记为「输入契约变过，不可直接并表」。

查看与取值：

```bash
GNN=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans; PY=/home/team/anaconda3/envs/nuplan/bin/python
$PY $GNN/scripts/analysis/result_manifest.py --print_schema        # 人读的列表
$PY $GNN/scripts/analysis/result_manifest.py --print_schema_hash   # 31bce53e29f9f0bd
$PY $GNN/scripts/analysis/result_manifest.py --self_test           # 自检
```

---

## 1. 为什么需要一个 hash

`col6/7/8` 的语义**按 node type 分岔**：

- 在 **ego / agent** 行上是 `timestamp / v_start / v_end`（v1 起就是如此，未变）；
- 在 **lane** 行上，v1 是三个恒 0 的占位槽，**v2 改为承载 route 三通道**。

这意味着：一份 `features_<token>.pkl` 是 v1 还是 v2、一个 checkpoint 训练时见过哪一种语义，
**从数据本身看不出来** —— 两者形状都是 `(N, 12)`、dtype 都是 float32，
`route_mode=none` 时甚至逐字节相同。

所以必须把 schema 的规范化序列化 hash 写进每一份结果 JSON。否则半年后无法回答
「这张表用的是哪套输入契约」，而这正是审稿人会问的问题。

---

## 2. 列定义（`--print_schema` 的快照）

```
 col name                      ego/agent                    lane
 ---------------------------------------------------------------------------------------
 0   x_start                   线段起点 x                     线段起点 x
 1   y_start                   线段起点 y                     线段起点 y
 2   x_end                     线段终点 x                     线段终点 x
 3   y_end                     线段终点 y                     线段终点 y
 4   heading_start             起点朝向 (rad)                 起点朝向 (rad)
 5   heading_end               终点朝向 (rad)                 终点朝向 (rad)
 6   ts_or_route_proximity     归一化 timestamp ∈ [0,1)      ★v2: route 邻近度 clip(1-d/radius,0,1)
 7   vstart_or_route_progress  v_start (m/s)                ★v2: 沿 route 归一化进度 ∈ [0,1]
 8   vend_or_is_on_route       v_end (m/s)                  ★v2: is_on_route ∈ {0,1}
 9   subgraph_id               子图 ID（cluster）             子图 ID（cluster）
 10  t_start                   相对 T_ref 起点时间 (s)         恒 0（占位）
 11  t_end                     相对 T_ref 终点时间 (s)         恒 0（占位）
```

- 坐标系：ego-centric，原点 = ego 当前 `rear_axle` 位姿，`x` = ego heading（前），`y` = 左。
- **行语义 = 一条 polyline 的一个线段（相邻两点），不是一个时刻。** 这点常被误读。
- 代码对应：`src/shared/utils/data_utils.py::encoding_vectornet_features`
  的三处 `np.hstack`（ego :826-837 / agent :869-878 / lane :904-915）。

`route_mode` 两种取值：

| route_mode | lane 行的 col6/7/8 | 保证 |
|---|---|---|
| `none` | **全 0** | 与 route 引入前的产出**逐字节一致**（E0 判据 0，10/10 已复验） |
| `three_channel` | route 三通道 | ego/agent 行不变（E0 判据 1-6，10/10 已复验） |

模型侧：`_extract_core_features` 只取**前 9 列**，因此 route 三通道经 Transformer 分支
**逐样本**生效（`transformer_predictor.py:232-243` 按 `data.batch == i` 掩码）。

---

## 3. 已知边界（必须进论文 limitation，不得省略）

1. route 三通道是**复用 lane 节点的占位通道**，**不是独立 route 分支**
   —— 论文不得写成 reference-line 条件化；若保留 route，Eq.(8) 必须同步改写。
2. **route 信号为空的样本约 11.8%**（95% CI [10.7%, 13.0%]），
   源自 devkit `prune_route_by_connectivity` 的半径连通性裁剪。
3. **约 19.9% 样本的 lane 数 > `LANE_NUM=40`**，被按到 ego 的距离排序后截断。
   截断本身不影响 route 通道的正确性（三通道用同一 permutation 对齐），但应进 limitation。
4. `EGO_HISTORY` 是 `[20, 2]` 局部逐帧增量，**2 s @ 10 Hz** —— 论文**不得写 4 s history**。

---

## 4. 执行契约常量（与论文原稿不符，Z10 已按授权改正）

| 项 | 执行值 | 提交稿写的 |
|---|---|---|
| `NUM_AGENTS` | **20** | 15 ❌ |
| `LANE_NUM` × points | **40 × 50** | 30 ❌ |
| route lanes | **0（不存在）** | 10 ❌ |
| crosswalks | **0（不存在）** | 5 ❌ |
| `QUERY_RADIUS` | **50 m** | 30 m ❌ |
| nearby-only 阈值 | **30 m** | 15 m ❌ |
| `PAST_TIME_HORIZON` | 2 s | 2 s ✅ |

`config.py` 里 `ROUTE_LANES_NUM=10` / `CROSSWALKS_NUM=5` 两个常量**确实存在**，
但按阶段1 决策 Option A，它们**仅供 `src/danger_generation/` 旧线消费，MAGNET LANE-only 不使用**
（`config.py:60` 注释原文）。原稿很可能照抄了常量而非执行路径。

---

## 5. 冻结策略

- 本 schema 在 **2026-07-28 冻结**，供 G3 final（4 cells × 5 seeds）及其后全部正式实验使用。
- **任何**对 `FEATURE_SCHEMA_V2` 的改动都必须：
  1. 递增 `SCHEMA_VERSION`；
  2. 在本文件追加一节记录 v2→v3 的 delta 与新 hash；
  3. 重跑 E0 十样本验收（`e0_verify.py` + `e0_route_diff.py` + `e0_infer.py`，
     四拓扑 edge hash 须恒为 `861/41/40/377`）；
  4. **不得**把不同 schema hash 的结果并进同一张表。
- 每份正式结果 JSON 由 `build_result_manifest()` 生成，必含 `schema_hash` 字段；
  `validate_manifest()` 会在 hash 漂移时报错。
