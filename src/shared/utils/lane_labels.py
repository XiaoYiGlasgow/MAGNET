"""lane 侧 label 的**唯一定义源**（D4 补录的两个，2026-08-06 裁定）。

为什么需要 lane 侧 label
------------------------
D4 原草案只列了 agent 侧，**漏了整个 lane 侧**。而 L-1 的 `R\\*`（剂量口径批判）
需要**lane 侧的分母** —— 现在「删 25% lane = 10 条」这种粗口径，正是两个 primary cell
符号相反的原因（主控表 §1.9）。没有 lane 侧 label 就没法做精确剂量控制。

🔴 一条必须先讲清的限制：`is_route_lane` 在主线档拿不到
--------------------------------------------------------
route 信息只在 `route_mode='three_channel'` 时才会被算出来并写进 lane 的 col6/7/8
（`process_nuplan_to_12d.py:215-222`）。**主线一直跑 `route_mode='none'`**
（主控表 §1.4），此时 `lane_route_channels is None`、三列恒为 0
⇒ **`is_route_lane` 在主线档没有信息可依，只能缺省**。

处理方式（不硬造）：
* 有 route 通道 ⇒ 产出 `is_route_lane`（取 col8 = `is_on_route`）；
* 没有 ⇒ 该键**不出现**在 label 里（而不是填一个假的 False）。
  🔴 填 False 会让下游把「不知道」误读成「不在 route 上」，是典型的静默类错误。

⇒ 若 L-1 确实需要 lane 侧 route 分母，**必须先裁定是否把新档预处理切到
`route_mode='three_channel'`** —— 那是**另一个变量**（改变模型输入），
不能和本次上限/label 改动混做（Q45 教训）。

坐标系
------
`vector_map['lanes']` 进入 `encoding_vectornet_features` 时**已是 ego 局部系**，
原点 = ego 当前位姿（实测：lane 节点到原点距离中位 50.21 m、max 90.85 m，
ego 历史首帧在 -12.58 m 处）⇒ `lane_dist_rank` 直接按到原点的最小距离排序即可。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

#: lane 侧 label 的字段名。`is_route_lane` 是**条件产出**，故不在此列（见模块 docstring）。
LANE_LABEL_SCHEMA: tuple = ("lane_dist_rank",)

#: 条件产出的字段名（仅当调用方给了 route 通道时出现）
LANE_LABEL_OPTIONAL: tuple = ("is_route_lane",)


def lane_min_distance(lane: np.ndarray) -> float:
    """lane 折线上离 ego（局部系原点）最近的那一点的距离。

    用**最小值**而不是首点/中点：lane 是一条几十米长的折线，
    用首点会让一条正好从 ego 身边穿过、但起点在远处的 lane 排到很后面。

    :param lane: 形状 (L, >=2)，前两列是局部系 x/y。
    :returns: 最近点距离（米）；空折线返回 ``inf``（排序时自然落到最后）。
    """
    arr = np.asarray(lane)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return float("inf")
    return float(np.min(np.linalg.norm(arr[:, :2], axis=1)))


def compute_lane_labels(lanes: Sequence[np.ndarray],
                        lane_route_channels: Optional[Sequence] = None) -> List[Dict]:
    """给每条 lane 算 label，返回列表与 `lanes` **逐位对齐**。

    :param lanes: `vector_map['lanes']`，已是 ego 局部系。
    :param lane_route_channels: 可选，与 `lanes` 同序的 (N, 3) route 三通道；
        第 3 列（col8）是 `is_on_route`。**传 None 时不产出 `is_route_lane`**。
    :returns: 与 `lanes` 等长的 label dict 列表。

    ⚠️ `lane_dist_rank` 是**按最近距离升序的名次**，不是 `lanes` 的原始下标
    （这一点和 agent 侧相反：agent 侧 `agents_past` 本身已按距离排好序，**下标即 rank**）。
    把两边当成同一个约定会静默地把 lane 的 rank 算错。

    🔬 **实测证据（2026-08-07，L1 val 前 30 个样本）**：交付几何里 lane 子图的出现顺序
    与「按最近点距离升序」**一次都没有重合（0/30）** ⇒ 本 rank 携带的是**原始下标之外的信息**，
    不是把下标换个名字。
    ⚠️ 注意 `convert_feature_layer_to_fixed_size` 的 docstring 说它「按到 ego 的最小距离
    重排后截断」—— 那个排序发生在**坐标变换与插值之前**、用的是另一套距离键，
    与本函数在交付几何上量到的名次**不是同一个量**。两者都对，只是口径不同，**不得混用**。
    """
    dists = [lane_min_distance(ln) for ln in lanes]
    order = np.argsort(dists, kind="stable")          # stable：距离相同时保持 map_process 的原序
    rank_of = {int(idx): r for r, idx in enumerate(order)}

    labels: List[Dict] = []
    for i in range(len(lanes)):
        lb: Dict = {"lane_dist_rank": int(rank_of[i])}
        if lane_route_channels is not None and i < len(lane_route_channels):
            lb["is_route_lane"] = bool(float(lane_route_channels[i][2]) > 0.5)
        labels.append(lb)
    return labels


def lane_label_stats(labels: Sequence[Dict]) -> Dict[str, float]:
    """汇总统计，供落盘 manifest 与冒烟核对。"""
    n = len(labels)
    has_route = [lb for lb in labels if "is_route_lane" in lb]
    out: Dict[str, float] = {"n_lanes": n, "has_route_info": len(has_route) > 0}
    if has_route:
        n_on = sum(1 for lb in has_route if lb["is_route_lane"])
        out.update({"n_on_route": n_on, "on_route_frac": n_on / len(has_route)})
    return out
