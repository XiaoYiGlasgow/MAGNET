"""agent 侧 label 的**唯一定义源**（D1/D3/D4，2026-08-06 裁定）。

背景
----
现状预处理在建图前用 `avg_speed < 0.01` **硬删** 72.91% 的近邻车辆，
且顺序是「先取最近 20 → 再删静止」。这带来两个后果：

1. 被删的车**不在图里** ⇒ 连「留车切边」都做不到（§0.0-e 8.1 的枢纽洞察要求车必须在图里）；
2. **29.53% 的场景图里一辆车都没有** ⇒ 对这些场景任何 agent 级干预的 response
   **结构上恒为 0**，近 1/3 样本在稀释 estimand。

新方案（§0.0-i）：**预处理全量保留 + 打 label，训练/干预时再按 label mask**。

🔴 两条硬约束
-------------
1. **label 存独立字段**（`SUBGRAPH_ID_TO_LABEL`），**不混进 `POLYLINE_FEATURES` 的列**
   ⇒ 模型输入 schema 不变、受管文件 `transformer_predictor.py` 不必改。
2. **输入侧 / 审计侧严格隔离**：用 GT future 定义的 label（如 `near_ego_future`）
   **只能离线审计，不得进模型输入**（§7 硬约束）。本模块**只产出输入侧 label**，
   审计侧 label 由单独的离线脚本产出，从物理上杜绝泄漏。

D4 裁定的 label 清单
--------------------
====================  ========  ==============================================
label                 类别      用途
====================  ========  ==============================================
``is_moving``         输入侧    复现现状档／定义 L-2 运动×运动边子集／旋钮档位
``dist_rank``         输入侧    复现截断档／剂量扫描／R\\* 精确控制删几个
``was_truncated``     输入侧    标出旧 20 槽之外的车／新旧口径对照／等价门验证
``agent_type``        输入侧    🟡 降级：实测留下与删掉的 100% 都是 vehicle，
                                零区分力；留字段仅为 provenance 自洽
====================  ========  ==============================================

已删除（D4）：``is_parked_far`` —— 可由 ``is_moving=False AND near_ego_future>8m`` 派生。
lane 侧 label（``is_route_lane`` / ``lane_dist_rank``）见 `lane_labels.py`，不在本模块。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

#: 静止判定阈值（m/s）。🔴 **唯一定义源** —— `data_utils.py` 原先是就地硬编码的局部变量，
#: 现改为从本模块导入。值逐字保持 0.01，改它会同时改变新旧两条路径。
SPEED_THRESHOLD: float = 0.01

#: 旧口径的 agent 槽位上限。`was_truncated` 用它划线；等价门用它复现旧视图。
#: 🔴 不是新档的上限（新档 64，见主控表 §1.1），本常量只用于「旧口径是什么」。
LEGACY_NUM_AGENTS: int = 20

#: 输入侧 label 的字段名，顺序即 `LABEL_SCHEMA` 的列序（写盘与单测都认它）。
LABEL_SCHEMA: tuple = ("is_moving", "dist_rank", "was_truncated", "agent_type")

#: `agent_type` 的取值（末 3 维 one-hot 的解码结果）
AGENT_TYPES: tuple = ("vehicle", "pedestrian", "other")


def agent_avg_speed(agent_hist: np.ndarray) -> float:
    """逐字复刻 `data_utils.py` 原有的平均速度算法（相邻两帧速度的均值再取均值）。

    🔴 **不得"顺手改进"**：等价门要求新旧两条路径对同一辆车给出**逐位相同**的静止判定。
    原实现取的是 `mean((speed[:-1] + speed[1:]) / 2)`，不是 `mean(speed)` ——
    两者在数值上不同（首尾帧权重减半），改了就过不了门。

    :param agent_hist: 形状 (M, >=5) 的 agent 历史，列序 [x, y, heading, vx, vy, ...]。
    :returns: 平均速度（m/s）；M < 2 时返回 0.0（调用方本就会跳过这类 agent）。
    """
    if agent_hist.shape[0] < 2:
        return 0.0
    speed = np.sqrt(agent_hist[:, 3] ** 2 + agent_hist[:, 4] ** 2)
    return float(np.mean((speed[:-1] + speed[1:]) / 2))


def decode_agent_type(agent_hist: np.ndarray) -> str:
    """从末 3 维 one-hot 解出类型；缺列或全零时返回 ``"other"``。

    列约定见 `data_utils.py:180-195`：`agents[i, :, -3:]` 依次是
    vehicle / pedestrian / other 的 one-hot。
    """
    if agent_hist.shape[1] < 8:          # 5 个状态列 + 3 个 one-hot
        return "other"
    onehot = agent_hist[0, -3:]
    if not np.any(onehot):
        return "other"
    return AGENT_TYPES[int(np.argmax(onehot))]


def is_usable_agent(agent_hist: np.ndarray) -> bool:
    """逐字复刻 `data_utils.py` 的可用性判据：``M < 2 or np.all(agent_hist == 0)`` 即不可用。

    🔴 这一条**与静止无关** —— 它筛掉的是补零占位的空槽，新旧两条路径都必须照样筛掉，
    否则空槽会变成图里的节点。
    """
    return not (agent_hist.shape[0] < 2 or np.all(agent_hist == 0))


def compute_agent_labels(agents_past: Sequence[np.ndarray],
                         legacy_num_agents: int = LEGACY_NUM_AGENTS,
                         speed_threshold: float = SPEED_THRESHOLD) -> List[Optional[Dict]]:
    """给 `agents_past` 里每一辆车算输入侧 label。

    ⚠️ **返回值与 `agents_past` 逐位对齐**（同样长度、同样下标），
    不可用的槽位返回 ``None`` —— 这样调用方可以用同一个下标去对 subgraph_ID，
    不必再维护第二套映射（第二套映射正是历史上错位 bug 的来源）。

    :param agents_past: 长度 = 槽位数；每项形状 (M, >=5)，**已按到 ego 的距离升序**
        （`data_utils.py:183` 的 `argsort(distance_to_ego)[:num_agents]`）
        ⇒ 下标即 ``dist_rank``。
    :param legacy_num_agents: 旧口径槽位数，用于 ``was_truncated`` 划线。
    :param speed_threshold: 静止判定阈值；默认取本模块的唯一定义源。
    :returns: 与 `agents_past` 等长的列表，每项是 label dict 或 ``None``。
    """
    labels: List[Optional[Dict]] = []
    for rank, agent_hist in enumerate(agents_past):
        arr = np.asarray(agent_hist)
        if not is_usable_agent(arr):
            labels.append(None)
            continue
        labels.append({
            "is_moving": bool(agent_avg_speed(arr) >= speed_threshold),
            "dist_rank": int(rank),
            "was_truncated": bool(rank >= legacy_num_agents),
            "agent_type": decode_agent_type(arr),
        })
    return labels


def legacy_kept_ranks(labels: Sequence[Optional[Dict]],
                      legacy_num_agents: int = LEGACY_NUM_AGENTS) -> List[int]:
    """由 label 复现**旧口径实际入图的那批 agent**（等价门的核心）。

    旧口径的两步，顺序不可交换（`data_utils.py:183` 与 `:867`）：

    1. **先**取最近 `legacy_num_agents` 个（``dist_rank < N``）；
    2. **再**删掉静止的（``is_moving == False``）。

    🔴 顺序反过来会多留车（同口径实测 3.71 → 4.26，+14.9%），
    所以这里必须先按 rank 截断再按 is_moving 过滤，**不能合并成一个条件**去写
    —— 合并写法在 `dist_rank < N` 与 `is_moving` 同时为真时结果相同，但语义不同，
    将来若有人把截断数改大就会静默偏离旧口径。

    :returns: 旧口径下会入图的 `dist_rank` 列表，**升序**（即入图顺序）。
    """
    truncated = [lb for lb in labels if lb is not None and lb["dist_rank"] < legacy_num_agents]
    return [lb["dist_rank"] for lb in truncated if lb["is_moving"]]


def label_stats(labels: Sequence[Optional[Dict]]) -> Dict[str, float]:
    """一批 label 的汇总统计，供预处理落盘时写进 manifest、以及冒烟核对用。"""
    usable = [lb for lb in labels if lb is not None]
    n = len(usable)
    if n == 0:
        return {"n_usable": 0, "n_moving": 0, "n_static": 0,
                "static_frac": float("nan"), "n_truncated": 0}
    n_moving = sum(1 for lb in usable if lb["is_moving"])
    return {
        "n_usable": n,
        "n_moving": n_moving,
        "n_static": n - n_moving,
        "static_frac": (n - n_moving) / n,
        "n_truncated": sum(1 for lb in usable if lb["was_truncated"]),
    }
