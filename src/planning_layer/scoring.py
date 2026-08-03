"""C4 规划层 —— 显式规划目标（R4-1 件②③④）。

🔴 §7 写作硬约束（必须随结果一起声明）：本目标函数的分项与 nuPlan 官方 score 的分项
**同源** —— 进展 / 碰撞 / 可行驶区域 / 限速既是这里的优化目标，也是评测指标。
⇒ 闭环提升**部分是同义反复**，论文必须显式声明，并用 leave-one-out 消融把
「优化了被评的那一项」与「真的改善了驾驶」分开报。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from . import config as C
from . import geometry as G
from .candidates import Candidate

#: 可行驶区域查询：给一批全局点，返回逐点 bool（True = 在可行驶区域内）
DrivableFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SceneContext:
    """打分需要的场景信息（全部为纯数据 / 可选闭包，不引 devkit 类型）。"""

    #: ``[T, N, 2]`` 逐时刻已前推的 agent 位置；None = 没有 agent
    agents_xy: Optional[np.ndarray] = None
    #: 限速（m/s）；None = 该场景没有限速信息
    speed_limit: Optional[float] = None
    #: 可行驶区域查询闭包；None = 不做该项检查
    drivable_fn: Optional[DrivableFn] = None
    #: 🔴 进展方向（单位向量）。**必须由调用方给定**（`layer.refine` 会兜底成 ego 朝向）；
    #: 若留 None 让每个候选按自身方向计分，倒车候选也会拿正分（08-01 复核抓到的真 bug）。
    progress_direction: Optional[np.ndarray] = None
    #: 进展起点（通常是 ego 当前位置）
    origin: Optional[np.ndarray] = None
    #: ego 半宽 / 前后悬（m）；给了就按**四角**查可行驶区域（对齐官方 metric），否则退化为质心
    ego_half_width: Optional[float] = None
    ego_front_m: Optional[float] = None
    ego_rear_m: Optional[float] = None


def _footprint_points(cand: Candidate, ctx: SceneContext) -> np.ndarray:
    """按路径切向把轨迹展开成四角点集 ``[4T, 2]``。

    ⚠️ 官方 `drivable_area_compliance` 判的是 **ego 的四个角**（FL/RL/RR/FR）
    加 **0.3 m** 容差（`drivable_area_compliance_statistics.yaml:8`），**不是质心**。
    只查质心会选出「质心在路上、车角出界」的候选。
    """
    if ctx.ego_half_width is None or ctx.ego_front_m is None or ctx.ego_rear_m is None:
        return cand.xy
    t = np.gradient(cand.xy, axis=0)
    nrm = np.hypot(t[:, 0], t[:, 1])
    nrm = np.where(nrm < 1e-9, 1.0, nrm)
    t = t / nrm[:, None]
    nvec = np.stack([-t[:, 1], t[:, 0]], axis=1)
    hw, fr, rr = ctx.ego_half_width, ctx.ego_front_m, ctx.ego_rear_m
    return np.concatenate([
        cand.xy + fr * t + hw * nvec,   # FL
        cand.xy + fr * t - hw * nvec,   # FR
        cand.xy - rr * t + hw * nvec,   # RL
        cand.xy - rr * t - hw * nvec,   # RR
    ], axis=0)


def progress_term(cand: Candidate, ctx: SceneContext, ego_heading: float) -> float:
    """沿 route/参考方向推进的距离，**已归一化到 [~0, 1]**。

    **这是针对「不作为失效」的靶。**

    🔴 归一化不是装饰（第一版实现的真 bug，单测判据 [D] 抓到）：
    进展项以「米」计（可达 180 m），其余项都是 O(1) ⇒ 进展会压倒一切，
    限速 8 m/s 的场景里选中了 22.6 m/s 的候选。所有分项必须先无量纲化，
    `W_*` 才是真正的权衡系数。
    """
    origin = ctx.origin if ctx.origin is not None else cand.xy[0]
    direction = ctx.progress_direction
    if direction is None:
        # 🔴 兜底成 **ego 朝向**，绝不能用候选自身方向 —— 否则倒车候选按自己的倒退方向
        #    也拿正分（实测：参考朝 −x 时全部候选 progress 都是正的，最高 39.66）。
        direction = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    return G.project_progress(cand.xy, origin, direction)


def progress_scale(cands, ctx: SceneContext, ego_heading: float) -> float:
    """归一化尺度 = **本候选集内**最大推进距离。

    与 nuPlan 官方 `ego_progress_along_expert_route`（相对 expert 归一）同构：
    进展项因此在每个场景里都张满 [0, 1]，与其余无量纲项可比。
    ⚠️ 用「V_MAX × horizon」当尺度不行 —— 实际推进通常只有它的 1/5，
    进展项被压到 0.2 以下，舒适项反而主导 ⇒ 规划层会变得畏缩，
    正好复现我们要修的「不作为失效」。
    """
    vals = [progress_term(c, ctx, ego_heading) for c in cands]
    # ⚠️ 下界取 1 m 而非 1e-6：全部候选都倒退时 max 为负，除以 1e-6 会把分数放大到 1e7
    #    并让符号翻转（越倒退分越高）。1 m 的地板既避免爆炸，也保住排序。
    return max(max(vals), 1.0)


def _validate_agents(cand: Candidate, agents_xy: Optional[np.ndarray]) -> None:
    """agent 前推的时间步数必须与候选一致，否则 numpy 会在别处抛难懂的广播错。"""
    if agents_xy is None or agents_xy.size == 0:
        return
    if agents_xy.ndim != 3 or agents_xy.shape[0] != len(cand.xy):
        raise ValueError(
            f"agents_xy 形状应为 [T={len(cand.xy)}, N, 2]，实际 {agents_xy.shape}；"
            "前推步数必须与规划 horizon 一致"
        )


def collision_term(cand: Candidate, ctx: SceneContext) -> float:
    """避障惩罚，**无量纲**（越大越差）。硬碰撞 ≥ 1.0，软惩罚 ∈ (0, 1)。

    ⚠️ 官方 `no_ego_at_fault_collisions` 判的是**包围盒相交**（阈值 0），
    这里用**质心距离**做保守代理 —— 该近似必须写进论文（设计文档 §7）。
    """
    _validate_agents(cand, ctx.agents_xy)
    d, _ = G.min_pairwise_distance(cand.xy, ctx.agents_xy)
    if not np.isfinite(d):
        return 0.0
    if d < C.COLLISION_RADIUS_M:
        # 硬碰撞：1.0 起步 + 侵入深度；配 W_COLLISION=4 后，任何进展（≤1）都换不来碰撞
        return 1.0 + float((C.COLLISION_RADIUS_M - d) / C.COLLISION_RADIUS_M)
    if d < C.NEAR_MISS_RADIUS_M:
        span = C.NEAR_MISS_RADIUS_M - C.COLLISION_RADIUS_M
        return float((C.NEAR_MISS_RADIUS_M - d) / span)  # 0..1 软惩罚
    return 0.0


def ttc_term(cand: Candidate, ctx: SceneContext, dt: float = C.DT) -> float:
    """TTC 惩罚，对齐官方 `time_to_collision_within_bound` 的语义。

    官方判据（`time_to_collision_within_bound_statistics.yaml`）：
    在 `time_horizon = 3.0 s` 内，最小 TTC **低于 `least_min_ttc = 0.95 s`** 即算违规。
    这里用「相对位置 / 相对接近速率」估 TTC —— agent 位置已经前推过，
    逐步差分即可得相对速度，不需要额外输入。

    🔴 为什么单独加这一项：C3 实测**只有 `ttc` 这个通道对我们的干预有反应**（+0.031），
    只用「全程最小间距」会把 6 s 后的远期接近也算进来，与官方 3 s 窗口不是一回事。
    """
    _validate_agents(cand, ctx.agents_xy)
    a = ctx.agents_xy
    if a is None or a.size == 0:
        return 0.0
    n = min(int(round(C.TTC_TIME_HORIZON_S / dt)), len(cand.xy) - 1)
    if n < 1:
        return 0.0
    rel = a[: n + 1] - cand.xy[: n + 1, None, :]            # [n+1, N, 2]
    dist = np.hypot(rel[:, :, 0], rel[:, :, 1])             # [n+1, N]
    closing = -(np.diff(dist, axis=0) / dt)                 # [n, N] 正 = 正在接近
    gap = np.maximum(dist[:-1] - C.COLLISION_RADIUS_M, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing > 1e-3, gap / closing, np.inf)
    min_ttc = float(np.min(ttc)) if ttc.size else float("inf")
    if not np.isfinite(min_ttc) or min_ttc >= C.TTC_LEAST_MIN_S:
        return 0.0
    return float((C.TTC_LEAST_MIN_S - min_ttc) / C.TTC_LEAST_MIN_S)   # 0..1


def rules_term(cand: Candidate, ctx: SceneContext, ego_heading: float = 0.0) -> float:
    """交规惩罚，**无量纲**（越大越差）。三项都按官方判据的容差口径写：

    * 限速：超出 `max_overspeed_value_threshold = 2.23 m/s` 之后才计惩罚；
    * 可行驶区域：容差 `0.3 m`，且**按四角**查（给了车身尺寸时）；
    * 行驶方向：沿参考方向的**倒退量**超过 `2 m` 起罚、`6 m` 记满分
      （官方 `driving_direction_compliance` 的两个阈值）。
    """
    pen = 0.0
    if ctx.speed_limit is not None:
        over = np.maximum(cand.speed - (ctx.speed_limit + C.OVERSPEED_TOLERANCE_MPS), 0.0)
        pen += float(np.mean(over)) / max(ctx.speed_limit, 1.0)
    if ctx.drivable_fn is not None:
        pts = _footprint_points(cand, ctx)
        inside = np.asarray(ctx.drivable_fn(pts), dtype=bool)
        pen += float(np.mean(~inside))       # 越界点比例 ∈ [0,1]
    pen += _backward_penalty(cand, ctx, ego_heading)
    return pen


def _backward_penalty(cand: Candidate, ctx: SceneContext, ego_heading: float) -> float:
    """逆行惩罚：沿参考方向的最大倒退量，按官方 2 m / 6 m 两档归一到 [0, 1]。"""
    direction = ctx.progress_direction
    if direction is None:
        direction = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    origin = ctx.origin if ctx.origin is not None else cand.xy[0]
    s_along = (cand.xy - origin[None, :]) @ direction
    back = float(-np.min(s_along))           # 最深的倒退量（m）
    if back <= C.BACKWARD_TOLERANCE_M:
        return 0.0
    span = C.BACKWARD_VIOLATION_M - C.BACKWARD_TOLERANCE_M
    return float(min((back - C.BACKWARD_TOLERANCE_M) / span, 1.0))


def comfort_term(cand: Candidate) -> float:
    """舒适性正则：加速度 / jerk / 侧向加速度的均方量级。"""
    return float(
        np.mean(np.abs(cand.accel)) / C.A_LON_MAX
        + np.mean(np.abs(cand.jerk)) / C.JERK_MAX
        + np.mean(np.abs(cand.lat_accel)) / C.A_LAT_MAX
    )


def score(
    cand: Candidate,
    ctx: SceneContext,
    cfg: C.PlanningLayerConfig,
    ego_heading: float,
    scale: float = 1.0,
) -> float:
    """总分（越大越好）。四个开关对应 R4-1 的四件缺失件，可独立关掉做 leave-one-out。

    :param scale: 进展项的归一化尺度，由 `progress_scale()` 在候选集上算出。
    """
    total = 0.0
    if cfg.progress:
        total += C.W_PROGRESS * progress_term(cand, ctx, ego_heading) / max(scale, 1e-6)
    if cfg.collision:
        total -= C.W_COLLISION * collision_term(cand, ctx)
        total -= C.W_TTC * ttc_term(cand, ctx, cfg.dt)
    if cfg.rules:
        total -= C.W_RULES * rules_term(cand, ctx, ego_heading)
    total -= C.W_COMFORT * comfort_term(cand)
    return total


def score_breakdown(
    cand: Candidate,
    ctx: SceneContext,
    cfg: C.PlanningLayerConfig,
    ego_heading: float,
    scale: float = 1.0,
) -> Dict[str, float]:
    """逐项分解 —— 只用于诊断与单测，不参与选择。"""
    return {
        "progress": progress_term(cand, ctx, ego_heading) / max(scale, 1e-6),
        "collision": collision_term(cand, ctx),
        "ttc": ttc_term(cand, ctx, cfg.dt),
        "rules": rules_term(cand, ctx, ego_heading),
        "comfort": comfort_term(cand),
        "total": score(cand, ctx, cfg, ego_heading, scale),
    }
