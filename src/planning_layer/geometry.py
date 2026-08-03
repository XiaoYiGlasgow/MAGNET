"""C4 规划层 —— 纯几何工具（弧长参数化、法向偏移、曲率）。

只依赖 numpy（见 `config.py` 的依赖纪律）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_EPS = 1e-9


def arc_length(xy: np.ndarray) -> np.ndarray:
    """逐点累计弧长，`s[0] = 0`。"""
    d = np.diff(xy, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_by_arc_length(xy: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    """按弧长重采样折线；超出末端时沿末端切向**外推**（不夹住，否则车会假性停住）。"""
    s = arc_length(xy)
    total = float(s[-1])
    out = np.empty((len(s_query), 2), dtype=float)

    inside = s_query <= total
    if np.any(inside):
        out[inside, 0] = np.interp(s_query[inside], s, xy[:, 0])
        out[inside, 1] = np.interp(s_query[inside], s, xy[:, 1])
    if np.any(~inside):
        tangent = _end_tangent(xy)
        extra = (s_query[~inside] - total)[:, None]
        out[~inside] = xy[-1][None, :] + extra * tangent[None, :]
    return out


def _end_tangent(xy: np.ndarray) -> np.ndarray:
    """末端单位切向；折线退化（全部重合）时返回 +x。"""
    for i in range(len(xy) - 1, 0, -1):
        d = xy[i] - xy[i - 1]
        n = float(np.hypot(d[0], d[1]))
        if n > _EPS:
            return d / n
    return np.array([1.0, 0.0])


def unit_normals(xy: np.ndarray) -> np.ndarray:
    """逐点左法向（切向逆时针转 90°）。"""
    t = np.gradient(xy, axis=0)
    n = np.hypot(t[:, 0], t[:, 1])
    n = np.where(n < _EPS, 1.0, n)
    t = t / n[:, None]
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def lateral_offset(xy: np.ndarray, offset_m: float, ramp_len_m: float = 0.0) -> np.ndarray:
    """沿左法向偏移 `offset_m`（正 = 左）。

    :param ramp_len_m: 偏移在多少弧长内**从 0 线性升到满值**。

    🔴 `ramp_len_m=0`（整体平移）**不能用于生成可执行轨迹**：
    首点会立刻偏出 `offset_m`，等于让 ego 在 t0 横跳 1 m，
    官方 `two_stage_controller`（LQR 系）会看到一个阶跃横向误差。
    08-01 复核由判据 [H] 抓到。
    """
    if abs(offset_m) < _EPS:
        return xy.copy()
    if ramp_len_m <= _EPS:
        return xy + offset_m * unit_normals(xy)
    s = arc_length(xy)
    u = np.clip(s / ramp_len_m, 0.0, 1.0)
    # 🔴 必须用 C² 连续的 smootherstep，不能用线性 clip：
    #    线性爬升在起点和终点各有一个折角 ⇒ 曲率出现尖峰，
    #    实测把 a_lat 顶到 6.54（界 4.89），而真实横移只需要约 0.9 m/s²。
    w = (u ** 3 * (u * (6.0 * u - 15.0) + 10.0))[:, None]
    return xy + (offset_m * w) * unit_normals(xy)


def curvature(xy: np.ndarray) -> np.ndarray:
    """逐点曲率 |kappa|，用一阶/二阶差分估计。退化处返回 0。"""
    d1 = np.gradient(xy, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = np.power(d1[:, 0] ** 2 + d1[:, 1] ** 2, 1.5)
    return np.where(den < _EPS, 0.0, num / np.maximum(den, _EPS))


def path_direction(xy: np.ndarray, fallback_heading: float) -> np.ndarray:
    """折线整体前进方向（首末连线）；折线退化时用 `fallback_heading`。

    🔴 这条 fallback 是为**上游塌陷**准备的：输出恒 bias ⇒ 参考轨迹几乎不动，
    此时必须仍能给出一个前进方向，否则进展项无从算起（判据 [E]）。
    """
    d = xy[-1] - xy[0]
    n = float(np.hypot(d[0], d[1]))
    if n > 1e-3:
        return d / n
    return np.array([np.cos(fallback_heading), np.sin(fallback_heading)])


def project_progress(xy: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """终点相对起点在 `direction` 上的投影长度（m），即「沿参考方向推进了多少」。"""
    return float(np.dot(xy[-1] - origin, direction))


def trim_to_nearest(xy: np.ndarray, point: np.ndarray) -> np.ndarray:
    """把折线裁到「离 `point` 最近的那一点」之后，并把该投影点作为新起点。

    🔴 **不做这一步会瞬移 ego**（08-01 复核抓到的灾难级 bug）：
    上游塌陷时会改用 route 中心线当基准路径，而 route 起点通常在 ego **后方几十米**。
    直接拿它当基准 ⇒ 输出轨迹的首点落在 route 起点 ⇒ ego 被瞬移回去。
    实测：ego 在 x=50、route 起于 x=0 时，输出首点 x=0.508（倒退 50 m）。
    """
    if len(xy) < 2:
        return xy
    seg = xy[1:] - xy[:-1]
    seg_len2 = np.sum(seg ** 2, axis=1)
    seg_len2 = np.where(seg_len2 < _EPS, _EPS, seg_len2)
    t = np.clip(np.sum((point[None, :] - xy[:-1]) * seg, axis=1) / seg_len2, 0.0, 1.0)
    proj = xy[:-1] + t[:, None] * seg
    i = int(np.argmin(np.sum((proj - point[None, :]) ** 2, axis=1)))
    return np.vstack([proj[i][None, :], xy[i + 1:]])


def min_pairwise_distance(
    ego_xy: np.ndarray, agents_xy: Optional[np.ndarray]
) -> Tuple[float, int]:
    """ego 轨迹与逐时刻 agent 位置的最小间距。

    :param ego_xy: ``[T, 2]``
    :param agents_xy: ``[T, N, 2]``（逐时刻已前推），或 None / 空
    :returns: ``(最小间距, 该间距出现的时间步)``；无 agent 时返回 ``(inf, -1)``
    """
    if agents_xy is None or agents_xy.size == 0:
        return float("inf"), -1
    diff = agents_xy - ego_xy[:, None, :]
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    t = int(np.argmin(np.min(dist, axis=1)))
    return float(np.min(dist)), t
