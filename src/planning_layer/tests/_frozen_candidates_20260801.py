"""C4 规划层 —— 候选轨迹生成（R4-1 件① 车辆动力学约束）。

🔴 关键设计：候选**由运动学前推生成**，不是「先生成再投影」——
可行性由构造保证，因此 `dynamics=True` 时任何候选都必然满足 `a_lon / jerk / a_lat` 硬界。
`dynamics=False`（消融格）时退化为「沿参考路径按比例缩放速度」，不施加任何界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.planning_layer import config as C  # 冻结副本只冻结本文件；config/geometry 用同一份
from src.planning_layer import geometry as G


@dataclass(frozen=True)
class EgoSnapshot:
    """规划层需要的最小 ego 状态（**纯数据，不引 devkit 类型**，两个 env 通用）。"""

    x: float
    y: float
    heading: float
    speed: float = 0.0


@dataclass
class Candidate:
    """一个候选轨迹及其生成参数。"""

    xy: np.ndarray            # [T, 2] 全局坐标
    speed: np.ndarray         # [T] 纵向速度
    accel: np.ndarray         # [T] 纵向加速度
    jerk: np.ndarray          # [T] 纵向 jerk
    lat_accel: np.ndarray     # [T] 侧向加速度 v^2 * kappa
    lon_ratio: float
    lat_offset: float


def reference_speed(reference_xy: np.ndarray, dt: float) -> float:
    """参考轨迹的平均速度 —— 候选纵向档位以它为基准。

    用**平均**而不是末端瞬时值：上游塌陷时末端差分噪声大，均值更稳。
    """
    s = G.arc_length(reference_xy)
    if len(s) < 2:
        return 0.0
    return float(s[-1] / max((len(s) - 1) * dt, 1e-6))


def speed_cap_profile(path_s: np.ndarray, path_kappa: np.ndarray) -> np.ndarray:
    """沿路径的速度上限剖面（**含反向传播**）。

    两条曲率限速都取自官方 comfort 判据：
      ① `v <= sqrt(a_lat_max / kappa)`   ② `v <= yaw_rate_max / kappa`

    🔴 **必须做反向传播**（08-01 复核抓到）：只在「当前弧长处」查限速是不够的 ——
    速度不能瞬变，等车开到弯里再减速已经晚了，实测 a_lat 会冲到 11。
    反向一遍 `v[i] <= sqrt(v[i+1]^2 + 2·|a_min|·ds)` 才能保证**到达前**就减到位。
    这是轨迹生成里速度剖面的标准做法（前向-反向两遍）。
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        v_curve = np.where(
            path_kappa > 1e-6,
            np.minimum(
                np.sqrt(C.A_LAT_MAX / np.maximum(path_kappa, 1e-12)),
                C.YAW_RATE_MAX / np.maximum(path_kappa, 1e-12),
            ),
            C.V_MAX,
        )
    v_cap = np.minimum(v_curve, C.V_MAX).astype(float)
    a_brake = abs(C.A_LON_MIN)
    for i in range(len(v_cap) - 2, -1, -1):
        ds = max(float(path_s[i + 1] - path_s[i]), 0.0)
        v_cap[i] = min(v_cap[i], float(np.sqrt(v_cap[i + 1] ** 2 + 2.0 * a_brake * ds)))
    return v_cap


def _rollout_speed_profile(
    v0: float, v_target: float, n: int, dt: float, cap_at_arc, limited: bool
):
    """从 `v0` 向 `v_target` 收敛的速度剖面。

    `limited=True` 时逐步施加 `a_lon` / `jerk` / 弯道限速（`v <= sqrt(a_lat_max / kappa)`）
    三重硬界；`limited=False` 时直接线性过渡，不施加任何界（消融格用）。

    🔴 **jerk 感知的加速度包络**（第一版实现的真 bug，单测判据 [B] 抓到）：
    只在相邻两步之间夹 `|Δa| <= J·dt` 是不够的 —— 速度在 0 处被 `max(0, ·)` 截断时，
    **实际**加速度会从 −5 突跳到 0，实测 jerk 冲到 32（> J=8）。
    正解是用「要把 Δv 降完并把 a 平滑收回 0，此刻 a 最多能到多大」这个物理包络：
        |a| <= sqrt(2 · J · |v_goal − v|)
    它保证 a 在 v 抵达 v_goal 时自然归零，不需要硬截断。

    🔴 **弧长必须在滚动内部自洽地跟踪并一并返回**（08-01 复核抓到的第二处）：
    第一版由调用方用 `cumsum(v)` 另算一遍 s，而滚动内部查限速时用的是**更新前**的速度，
    两者错开一步 ⇒ 限速永远查在车的**后面一步**，弯道里 kappa 上升快时就压不住，
    实测 a_lat 冲到 5.52（界 4.89）。

    :returns: ``(v, s)`` —— 逐步速度与**与之严格对应**的累计弧长（``s[0] = 0``）。
    """
    v = np.empty(n, dtype=float)
    s = np.empty(n, dtype=float)
    if not limited:
        v[:] = np.maximum(np.linspace(v0, v_target, n), 0.0)
        s[0] = 0.0
        s[1:] = np.cumsum(v[:-1]) * dt
        return v, s

    cur_s, cur_v, prev_a = 0.0, float(v0), 0.0
    for i in range(n):
        s[i] = cur_s
        # 弯道限速由 `speed_cap_profile` 事先算好（含反向传播），这里查表。
        # 🔴 必须在一个**前瞻窗口**上取 min，只查当前点（或只前瞻一步）都不够：
        #    加速中的候选要把 a 从正翻到负受 jerk 限制，等看见限速再减速已经晚了，
        #    实测 a_lat 越界到 5.38（界 4.89）。窗口取 `LOOKAHEAD_S` 秒的行程，
        #    实测 0.5 s 恰好压到界上，且直线推进能力不变（50.63 m，无畏缩代价）；
        #    再拉长反而变差（1.5 s→4.97、2.0 s→5.32），因为会先过度减速再重新加速。
        look = max(cur_v * C.LOOKAHEAD_S, 2.0)
        v_goal = float(min(
            v_target, C.V_MAX,
            min(cap_at_arc(cur_s + u) for u in np.linspace(0.0, look, C.LOOKAHEAD_SAMPLES)),
        ))

        dv = v_goal - cur_v
        # jerk 感知包络：留出把 a 平滑收回 0 的余量。
        # 🔴 必须用**离散时间**的包络，不能用连续版 sqrt(2·J·|dv|)：
        #    连续版在 v→0 附近塌缩得比 J·dt 还快（实测最后一步 jerk 冲到 15.7）。
        #    离散版由「a 以 J·dt 阶梯降到 0 期间共减掉多少速度」反解：
        #        |dv| = a²/(2J) + a·m  ⇒  a = -m + sqrt(m² + 2·J·|dv|)
        #    余量 m 取**一整步** J·dt（而非半步）：给「最后一步落到 v=0」留出 jerk 权限，
        #    否则终末步实测 jerk 会溢出到 8.78（> J=8）。保守方向，不放宽任何判据。
        margin = C.JERK_MAX * dt
        env = -margin + np.sqrt(margin * margin + 2.0 * C.JERK_MAX * abs(dv))
        a_ceil = min(C.A_LON_MAX, env)
        a_floor = max(C.A_LON_MIN, -env)

        a = float(np.clip(dv / dt, a_floor, a_ceil))
        a = float(np.clip(a, prev_a - C.JERK_MAX * dt, prev_a + C.JERK_MAX * dt))
        # 🔴 jerk 带**不得**把 a 压到包络之外：压出去后 v 掉得比包络假设的快，
        #    包络随之塌缩得比 J·dt 还快，下一步就追不上 ⇒ 停车瞬间实测 jerk 冲到 18。
        #    再夹一次，把 a 拉回包络内，两个约束才自洽。
        a = float(np.clip(a, a_floor, a_ceil))
        new_v = cur_v + a * dt
        if new_v < 0.0:              # 不允许倒车；此步实际加速度随之修正
            new_v = 0.0
            a = (new_v - cur_v) / dt
        cur_v = new_v
        v[i] = cur_v
        cur_s += cur_v * dt
        prev_a = a
    return v, s


def generate(
    reference_xy: np.ndarray,
    ego: EgoSnapshot,
    cfg: C.PlanningLayerConfig,
    route_xy: Optional[np.ndarray] = None,
) -> List[Candidate]:
    """生成候选集。

    :param reference_xy: 上游 planner 给的参考轨迹 ``[T, 2]``（全局坐标）
    :param ego: 当前 ego 快照（后轴中心）
    :param route_xy: 可选的 route 中心线；**上游塌陷时用它兜底几何**，否则参考轨迹退化成一个点
    """
    n = cfg.horizon
    dt = cfg.dt

    ego_xy = np.array([ego.x, ego.y], dtype=float)
    base = reference_xy
    # 上游塌陷（参考轨迹几乎不动）⇒ 几何退化，改用 route 中心线；再没有就用 ego 朝向直线
    if G.arc_length(base)[-1] < 0.5:
        if route_xy is not None and len(route_xy) >= 2 and G.arc_length(route_xy)[-1] > 0.5:
            # 🔴 必须先把 route 裁到 ego 的投影点，否则 ego 会被瞬移回 route 起点
            base = G.trim_to_nearest(np.asarray(route_xy, dtype=float), ego_xy)
            if len(base) < 2 or G.arc_length(base)[-1] < 0.5:
                base = reference_xy  # 裁完没剩几何 ⇒ 退回下面的直线兜底
    if G.arc_length(base)[-1] < 0.5:
        # 连 route 也没有 ⇒ 沿 ego 朝向拉一条直线
        direction = np.array([np.cos(ego.heading), np.sin(ego.heading)])
        span = max(ego.speed, 1.0) * n * dt
        base = np.stack([ego_xy, ego_xy + span * direction])
    else:
        # 🔴 上游轨迹本身也要裁到 ego 投影点：MAGNET/PlanTF 的首点通常≈ego，
        #    但 CV 上游或未来接别的 planner 时未必，统一裁一次杜绝瞬移。
        base = G.trim_to_nearest(np.asarray(base, dtype=float), ego_xy)

    v_ref = reference_speed(reference_xy, dt)
    if v_ref < 0.5:  # 参考本身没速度（含塌陷）⇒ 用 ego 当前速度兜底，再不行给一个可前进的下限
        v_ref = max(ego.speed, 3.0)

    # 横向偏移的爬升长度：约 1.5 s 的行程，且不短于 5 m ——
    # 保证首点严格等于 ego、偏移平滑加进去（判据 [H]）
    ramp_len = max(ego.speed * 1.5, 5.0)

    out: List[Candidate] = []
    for lat in cfg.lat_offsets:
        path = G.lateral_offset(base, lat, ramp_len_m=ramp_len)
        path_s = G.arc_length(path)
        path_kappa = G.curvature(path)

        path_v_cap = speed_cap_profile(path_s, path_kappa)

        def kappa_at_arc(s_val: float, _s=path_s, _k=path_kappa) -> float:
            """按**弧长**取路径曲率。

            🔴 不能改用「重采样后折线的数值曲率」—— 制动候选的点会挤在一起，
            `np.gradient` 估出的曲率会爆掉（第一版就是这么把 a_lat 算成 10 的）。
            """
            return float(np.interp(min(s_val, _s[-1]), _s, _k))

        def cap_at_arc(s_val: float, _s=path_s, _c=path_v_cap) -> float:
            return float(np.interp(min(s_val, _s[-1]), _s, _c))

        for ratio in cfg.lon_ratios:
            v, s = _rollout_speed_profile(
                ego.speed, v_ref * ratio, n, dt, cap_at_arc, limited=cfg.dynamics
            )
            # s 由滚动内部返回（s[0]=0，与 v 严格对应）。
            # ⚠️ 第一版在这里用 `np.cumsum(v)*dt` 另算一遍 ⇒ s[0]=v[0]·dt>0 ⇒
            #    **整条轨迹超前一步**（实测首点比 ego 前移 0.6 m），控制器在 t0 会看到阶跃输入。
            xy = G.resample_by_arc_length(path, s)
            a = np.gradient(v, dt)
            j = np.gradient(a, dt)
            k_along = np.array([kappa_at_arc(float(si)) for si in s])
            out.append(
                Candidate(
                    xy=xy, speed=v, accel=a, jerk=j,
                    lat_accel=v ** 2 * k_along,
                    lon_ratio=ratio, lat_offset=lat,
                )
            )
    return out
