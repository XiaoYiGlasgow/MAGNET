"""
Differentiable SVO (Social Value Orientation) estimator in PyTorch.

Mirrors compute_agent_svo() from danger_score.py with sigmoid soft approximations
replacing discrete counting, making the full computation graph differentiable
w.r.t. ego_future_dxdy for DPS-style gradient guidance.

Note on lateral speed: when heading is derived from atan2(dy, dx), the
perpendicular velocity component is identically zero. Lateral avoidance/approach
terms therefore do not contribute gradients in practice; the dominant gradient
sources are acceleration patterns (yield/aggressive) and social proximity force.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SvoTorchConfig:
    """Mirrors DangerScoreConfig SVO fields."""
    dt: float = 0.1
    sigmoid_scale: float = 5.0          # steepness of soft threshold; higher -> closer to hard

    svo_decel_threshold: float = 3.0    # m/s^2
    svo_accel_threshold: float = 2.0    # m/s^2
    svo_lateral_speed_threshold: float = 0.5   # m/s
    svo_risk_distance_threshold: float = 8.0   # m

    svo_yield_delta_x: float = -0.10
    svo_yield_delta_y: float = +0.15
    svo_aggressive_delta_x: float = +0.15
    svo_aggressive_delta_y: float = -0.10
    svo_avoidance_delta_x: float = -0.08
    svo_avoidance_delta_y: float = +0.12
    svo_approach_delta_x: float = +0.05
    svo_approach_delta_y: float = -0.03
    svo_social_force_delta_y: float = +0.20

    svo_angle_min: float = -45.0
    svo_angle_max: float = 135.0


def compute_ego_svo_torch(
    ego_future_dxdy: torch.Tensor,
    ego_history: torch.Tensor,
    agent_future_pos: Optional[torch.Tensor] = None,
    config: Optional[SvoTorchConfig] = None,
) -> torch.Tensor:
    """Differentiable ego SVO angle estimator (scalar, degrees).

    Args:
        ego_future_dxdy: (T, 2) ego trajectory increments (dx, dy), typically
            requires_grad=True. Corresponds to x_0_pred[:120].reshape(60, 2).
        ego_history: (T_hist, 5) fixed historical trajectory [x, y, heading, vx, vy].
            Not differentiated.
        agent_future_pos: (T, 2) agent absolute positions in ego-centric frame,
            fixed (no grad). Derived from agent dxdy cumsum + agent_history[-1, :2].
            If None, social force and lateral detection are skipped.
        config: SvoTorchConfig, uses defaults if None.

    Returns:
        svo_deg: scalar tensor (degrees), differentiable w.r.t. ego_future_dxdy.
    """
    if config is None:
        config = SvoTorchConfig()

    device = ego_future_dxdy.device
    dtype = ego_future_dxdy.dtype
    T = ego_future_dxdy.shape[0]
    dt = config.dt
    sc = config.sigmoid_scale  # sigmoid steepness

    # ── Step 1: reconstruct ego future speeds ───────────────────────────────
    # speed[t] = ||dxdy[t]|| / dt
    speeds_future = ego_future_dxdy.norm(dim=-1) / dt  # (T,)

    # prepend history last-step speed for accel computation
    last_vx = ego_history[-1, 3].to(dtype=dtype, device=device)
    last_vy = ego_history[-1, 4].to(dtype=dtype, device=device)
    speed_last = (last_vx ** 2 + last_vy ** 2).sqrt()
    all_speeds = torch.cat([speed_last.unsqueeze(0), speeds_future], dim=0)  # (T+1,)
    accel = torch.diff(all_speeds) / dt  # (T,)

    # ── Step 2: longitudinal behaviour (soft yield / aggressive count) ──────
    yield_soft = torch.sigmoid((-accel - config.svo_decel_threshold) * sc).sum()
    aggressive_soft = torch.sigmoid((accel - config.svo_accel_threshold) * sc).sum()

    # ── Step 3: lateral behaviour (avoidance / approach) ───────────────────
    avoidance_soft = ego_future_dxdy.new_zeros(())
    approach_soft = ego_future_dxdy.new_zeros(())

    ego_future_pos: Optional[torch.Tensor] = None  # used also in Step 4

    if agent_future_pos is not None:
        # ego future absolute positions (ego-centric, origin = ego_history[-1, :2])
        ego_last = ego_history[-1, :2].to(dtype=dtype, device=device)
        ego_future_pos = ego_last.unsqueeze(0) + ego_future_dxdy.cumsum(dim=0)  # (T, 2)

        # heading from motion direction (lateral_speed will be ~0 algebraically,
        # but we include the path for correctness if histories provide independent heading)
        eps_norm = 1e-8
        heading = torch.atan2(
            ego_future_dxdy[:, 1],
            ego_future_dxdy[:, 0] + eps_norm,
        )  # (T,)
        vx = ego_future_dxdy[:, 0] / dt
        vy = ego_future_dxdy[:, 1] / dt
        lat_speed = -vx * heading.sin() + vy * heading.cos()  # (T,), ~0 in practice

        # neighbor lateral direction (sign of relative y-position), detached
        rel_pos = agent_future_pos.to(dtype=dtype, device=device) - ego_future_pos  # (T, 2)
        neighbor_lat_dir = rel_pos[:, 1].sign().detach()  # (T,)

        # distance weight: close neighbors weighted more
        dist = rel_pos.norm(dim=-1).clamp(min=0.1)  # (T,)
        dist_weight = torch.sigmoid((config.svo_risk_distance_threshold - dist) * sc)

        # speed above lateral threshold
        speed_weight = torch.sigmoid((lat_speed.abs() - config.svo_lateral_speed_threshold) * sc)

        # avoidance: moving away from neighbor (lat * dir < 0)
        avoidance_soft = (dist_weight * speed_weight
                          * torch.sigmoid(-lat_speed * neighbor_lat_dir * sc)).sum()
        approach_soft = (dist_weight * speed_weight
                         * torch.sigmoid(lat_speed * neighbor_lat_dir * sc)).sum()

    # ── Step 4: social force (proximity-weighted) ───────────────────────────
    social_force_soft = ego_future_dxdy.new_zeros(())

    if agent_future_pos is not None and ego_future_pos is not None:
        agent_pos = agent_future_pos.to(dtype=dtype, device=device)
        dists = (ego_future_pos - agent_pos).norm(dim=-1).clamp(min=0.1)  # (T,)
        avg_dist = dists.mean()
        # smooth relu: force = max(0, risk_dist - avg_dist) / risk_dist
        social_force_soft = F.relu(
            config.svo_risk_distance_threshold - avg_dist
        ) / config.svo_risk_distance_threshold

    # ── Step 5: accumulate SVO weights ─────────────────────────────────────
    x_raw = ego_future_dxdy.new_ones(())   # start at (1, 0) = 0° individualist
    y_raw = ego_future_dxdy.new_zeros(())

    x_raw = x_raw + config.svo_yield_delta_x * yield_soft
    y_raw = y_raw + config.svo_yield_delta_y * yield_soft

    x_raw = x_raw + config.svo_aggressive_delta_x * aggressive_soft
    y_raw = y_raw + config.svo_aggressive_delta_y * aggressive_soft

    x_raw = x_raw + config.svo_avoidance_delta_x * avoidance_soft
    y_raw = y_raw + config.svo_avoidance_delta_y * avoidance_soft

    x_raw = x_raw + config.svo_approach_delta_x * approach_soft
    y_raw = y_raw + config.svo_approach_delta_y * approach_soft

    y_raw = y_raw + config.svo_social_force_delta_y * social_force_soft

    x_raw = x_raw.clamp(min=0.1)  # differentiable min-clamp

    # ── Step 6: normalise to unit circle ───────────────────────────────────
    norm = (x_raw ** 2 + y_raw ** 2).sqrt().clamp(min=1e-8)
    x_unit = x_raw / norm
    y_unit = y_raw / norm

    # ── Step 7: SVO angle in degrees ───────────────────────────────────────
    svo_rad = torch.atan2(y_unit, x_unit)
    svo_deg = svo_rad * (180.0 / math.pi)
    svo_deg = svo_deg.clamp(config.svo_angle_min, config.svo_angle_max)

    return svo_deg


def compute_ego_svo_torch_timevarying(
    ego_future_dxdy: torch.Tensor,
    ego_history: torch.Tensor,
    agent_future_pos: Optional[torch.Tensor] = None,
    config: Optional[SvoTorchConfig] = None,
) -> torch.Tensor:
    """Time-varying ego SVO: per-step SVO angle (T,) in degrees.

    Same formulation as compute_ego_svo_torch, but every behaviour term keeps
    its per-step value (no sum/mean), and social force uses per-step dist[t]
    instead of avg_dist. Returns a (T,) SVO sequence rather than a scalar.

    Enables distance-driven time-varying guidance: when ego-agent distance is
    large, aggressive/longitudinal terms dominate (aggressive approach); when
    distance shrinks below svo_risk_distance_threshold, the per-step social
    force grows and pulls SVO toward yielding — avoidance emerges from distance,
    no hard time-window split needed.

    Args:
        ego_future_dxdy: (T, 2) ego trajectory increments, requires_grad for DPS.
        ego_history: (T_hist, 5) fixed history [x, y, heading, vx, vy].
        agent_future_pos: (T, 2) agent absolute positions (ego-centric), detached.
        config: SvoTorchConfig.

    Returns:
        svo_deg: (T,) per-step SVO angles in degrees, differentiable w.r.t.
            ego_future_dxdy.
    """
    if config is None:
        config = SvoTorchConfig()

    device = ego_future_dxdy.device
    dtype = ego_future_dxdy.dtype
    T = ego_future_dxdy.shape[0]
    dt = config.dt
    sc = config.sigmoid_scale

    # ── Step 1: ego future speeds + accel (T,) ──────────────────────────────
    speeds_future = ego_future_dxdy.norm(dim=-1) / dt
    last_vx = ego_history[-1, 3].to(dtype=dtype, device=device)
    last_vy = ego_history[-1, 4].to(dtype=dtype, device=device)
    speed_last = (last_vx ** 2 + last_vy ** 2).sqrt()
    all_speeds = torch.cat([speed_last.unsqueeze(0), speeds_future], dim=0)
    accel = torch.diff(all_speeds) / dt  # (T,)

    # ── Step 2: per-step longitudinal yield / aggressive (T,) — no sum ──────
    yield_t = torch.sigmoid((-accel - config.svo_decel_threshold) * sc)
    aggressive_t = torch.sigmoid((accel - config.svo_accel_threshold) * sc)

    # ── Step 3 + 4: per-step lateral + social force (T,) ────────────────────
    avoidance_t = ego_future_dxdy.new_zeros(T)
    approach_t = ego_future_dxdy.new_zeros(T)
    social_force_t = ego_future_dxdy.new_zeros(T)

    if agent_future_pos is not None:
        ego_last = ego_history[-1, :2].to(dtype=dtype, device=device)
        ego_future_pos = ego_last.unsqueeze(0) + ego_future_dxdy.cumsum(dim=0)  # (T, 2)
        heading = torch.atan2(ego_future_dxdy[:, 1], ego_future_dxdy[:, 0] + 1e-8)
        vx = ego_future_dxdy[:, 0] / dt
        vy = ego_future_dxdy[:, 1] / dt
        lat_speed = -vx * heading.sin() + vy * heading.cos()  # (T,) ~0 in practice

        rel_pos = agent_future_pos.to(dtype=dtype, device=device) - ego_future_pos  # (T, 2)
        neighbor_lat_dir = rel_pos[:, 1].sign().detach()  # (T,)
        dist = rel_pos.norm(dim=-1).clamp(min=0.1)  # (T,) per-step distance
        dist_weight = torch.sigmoid((config.svo_risk_distance_threshold - dist) * sc)
        speed_weight = torch.sigmoid((lat_speed.abs() - config.svo_lateral_speed_threshold) * sc)
        avoidance_t = dist_weight * speed_weight * torch.sigmoid(-lat_speed * neighbor_lat_dir * sc)
        approach_t = dist_weight * speed_weight * torch.sigmoid(lat_speed * neighbor_lat_dir * sc)

        # per-step social force: dist[t] (NOT avg) — distance-driven avoidance
        social_force_t = F.relu(
            config.svo_risk_distance_threshold - dist
        ) / config.svo_risk_distance_threshold  # (T,)

    # ── Step 5: per-step accumulate SVO weights (T,) ────────────────────────
    x_raw = ego_future_dxdy.new_ones(T)
    y_raw = ego_future_dxdy.new_zeros(T)
    x_raw = x_raw + config.svo_yield_delta_x * yield_t
    y_raw = y_raw + config.svo_yield_delta_y * yield_t
    x_raw = x_raw + config.svo_aggressive_delta_x * aggressive_t
    y_raw = y_raw + config.svo_aggressive_delta_y * aggressive_t
    x_raw = x_raw + config.svo_avoidance_delta_x * avoidance_t
    y_raw = y_raw + config.svo_avoidance_delta_y * avoidance_t
    x_raw = x_raw + config.svo_approach_delta_x * approach_t
    y_raw = y_raw + config.svo_approach_delta_y * approach_t
    y_raw = y_raw + config.svo_social_force_delta_y * social_force_t
    x_raw = x_raw.clamp(min=0.1)

    # ── Step 6-7: per-step SVO angle (T,) ───────────────────────────────────
    norm = (x_raw ** 2 + y_raw ** 2).sqrt().clamp(min=1e-8)
    x_unit = x_raw / norm
    y_unit = y_raw / norm
    svo_rad = torch.atan2(y_unit, x_unit)
    svo_deg = svo_rad * (180.0 / math.pi)
    svo_deg = svo_deg.clamp(config.svo_angle_min, config.svo_angle_max)
    return svo_deg  # (T,)


def compute_ego_svo_batch(
    ego_future_dxdy: torch.Tensor,
    ego_history: torch.Tensor,
    agent_future_pos: Optional[torch.Tensor] = None,
    config: Optional[SvoTorchConfig] = None,
) -> torch.Tensor:
    """Batched wrapper around compute_ego_svo_torch.

    Args:
        ego_future_dxdy: (B, T, 2) or (B, 120) flat; requires_grad for DPS.
        ego_history: (B, T_hist, 5) or (T_hist, 5) broadcast.
        agent_future_pos: (B, T, 2) or (T, 2) broadcast, optional.
        config: SvoTorchConfig.

    Returns:
        svo_deg: (B,) SVO angles in degrees.
    """
    if ego_future_dxdy.dim() == 2 and ego_future_dxdy.shape[-1] != 2:
        # flat (B, 120) → (B, 60, 2)
        B = ego_future_dxdy.shape[0]
        ego_future_dxdy = ego_future_dxdy.reshape(B, -1, 2)

    B = ego_future_dxdy.shape[0]
    results = []
    for i in range(B):
        hist_i = ego_history[i] if ego_history.dim() == 3 else ego_history
        agent_i = (agent_future_pos[i]
                   if agent_future_pos is not None and agent_future_pos.dim() == 3
                   else agent_future_pos)
        results.append(compute_ego_svo_torch(ego_future_dxdy[i], hist_i, agent_i, config))
    return torch.stack(results)  # (B,)


def svo_dps_loss(
    x0_pred: torch.Tensor,
    ego_history: torch.Tensor,
    agent_history: Optional[torch.Tensor] = None,
    target_svo_deg: float = -45.0,
    config: Optional[SvoTorchConfig] = None,
) -> torch.Tensor:
    """Compute DPS guidance loss for SVO-based trajectory guidance.

    Minimising this loss pushes ego SVO toward target_svo_deg (default -45°,
    maximally competitive), guiding the diffusion model toward aggressive behaviour.

    Args:
        x0_pred: (B, 240) predicted clean trajectory; first 120 = ego dx,dy (60 steps),
            last 120 = agent dx,dy (60 steps). Must retain grad graph.
        ego_history: (B, T_hist, 5) or (T_hist, 5).
        agent_history: (B, T_hist, 5) or (T_hist, 5), optional.
        target_svo_deg: desired SVO angle in degrees. Negative = competitive.
        config: SvoTorchConfig.

    Returns:
        loss: scalar, = mean(svo_deg - target_svo_deg) clamped so loss >= 0.
              When ego SVO > target, loss > 0, gradient pushes SVO down.
    """
    if config is None:
        config = SvoTorchConfig()

    B = x0_pred.shape[0]
    ego_dxdy_flat = x0_pred[:, :120]          # (B, 120)
    agent_dxdy_flat = x0_pred[:, 120:]        # (B, 120)

    ego_dxdy = ego_dxdy_flat.reshape(B, 60, 2)    # (B, 60, 2)
    agent_dxdy = agent_dxdy_flat.reshape(B, 60, 2)  # (B, 60, 2)

    # reconstruct agent future absolute positions (fixed, no grad)
    agent_future_pos_list = []
    for i in range(B):
        hist_i = agent_history[i] if (agent_history is not None and agent_history.dim() == 3) \
            else agent_history
        if hist_i is not None:
            agent_last = hist_i[-1, :2].to(dtype=x0_pred.dtype, device=x0_pred.device)
            apos = agent_last.unsqueeze(0) + agent_dxdy[i].detach().cumsum(dim=0)
        else:
            apos = None
        agent_future_pos_list.append(apos)

    svo_degs = []
    for i in range(B):
        hist_i = ego_history[i] if ego_history.dim() == 3 else ego_history
        svo = compute_ego_svo_torch(ego_dxdy[i], hist_i, agent_future_pos_list[i], config)
        svo_degs.append(svo)

    svo_batch = torch.stack(svo_degs)  # (B,)
    # loss = mean distance above target (push SVO down toward target)
    loss = F.relu(svo_batch - target_svo_deg).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 路线 A（2026-09-16）：可微软累加 SVO —— 用户 09-13 定义的累加式时变 SVO 的软化版。
# numpy 参照（逐项对齐）：scratchpad/svo_timevarying_20260911/svo_readout.py 的
#   alpha_cumulative / smooth_trunc / _events / HIST_REAL_START。
# 与上面 compute_ego_svo_torch_timevarying 的区别：
#   - 行为量按时间累加（α(t) 由「历史初始量 + 未来第 1..t 帧累加量」求角），不是逐帧独立；
#   - 计数前加速度做截断窗口居中滑动平均；未来第 0..seam_skip 帧事件置 0（接缝余波）；
#   - 不含横向/social force 项（用户定义只有纵向让行/抢行两类事件）。
# 只供 sdedit_with_full_guidance_phase2(svo_measure='soft_cumulative') 与离线读数使用，
# 不改动上面任何既有函数。
# ─────────────────────────────────────────────────────────────────────────────

SVO_SOFT_HIST_REAL_START = 20   # _full 数据集 history 前 20 帧为零填充，只用其后的真实段（同 svo_readout.HIST_REAL_START）


def default_svo_soft_cfg() -> Dict[str, Any]:
    """软累加 SVO 的默认参数（每次返回新 dict，调用方可放心修改）。

    必选键：
        dth        : 让行（减速）硬阈值 m/s²，加速度 < -dth 计一次让行（同 alpha_cumulative 默认 3.0）
        ath        : 抢行（加速）硬阈值 m/s²，加速度 > ath 计一次抢行（同 alpha_cumulative 默认 2.0）
        sc         : sigmoid 陡度；越大越接近硬计数（1.6 = 与实测加速度量程匹配的过渡带宽）
        acc_smooth : 计数前加速度截断窗口滑动平均窗长（帧，须为奇数；5 = 0.5 s；<=1 = 不平滑）
        seam_skip  : 未来第 0..seam_skip 帧的事件置 0（第 0 帧本来就不计，另跳过接缝余波）
    可选键（缺省即下列值，与 alpha_cumulative 一致）：
        dt=0.1, d_yield=(-0.10, 0.15), d_aggr=(0.15, -0.10), x_floor=0.1,
        hist_real_start=SVO_SOFT_HIST_REAL_START
    """
    return dict(dth=3.0, ath=2.0, sc=1.6, acc_smooth=5, seam_skip=4)


def _resolve_svo_soft_cfg(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """默认参数 + 调用方覆盖（只覆盖给出的键），返回新 dict。"""
    out = default_svo_soft_cfg()
    out.update(dict(dt=0.1, d_yield=(-0.10, 0.15), d_aggr=(0.15, -0.10), x_floor=0.1,
                    hist_real_start=SVO_SOFT_HIST_REAL_START))
    if cfg is not None:
        out.update(dict(cfg))
    return out


def smooth_trunc_torch(a: torch.Tensor, w: int) -> torch.Tensor:
    """等权滑动平均（可微），边缘截断窗口：只对窗口内实际存在的帧求平均，长度不变。

    逐项对齐 svo_readout.smooth_trunc：
        np.convolve(a, ones(w), 'same') / np.convolve(ones_like(a), ones(w), 'same')
    w 为奇数时 'same' 居中，等价于两侧各补 w//2 个 0 做求和、再除以窗口内实际帧数。

    Args:
        a: (..., L) 待平滑序列（可带梯度）。
        w: 窗长（帧）；w<=1 时原样返回。

    Returns:
        (..., L) 平滑后序列。

    Raises:
        ValueError: w 为偶数（'same' 卷积不居中，事件整体后移半帧）或 L < w（与 numpy 版断言一致）。
    """
    w = int(w)
    if w <= 1:
        return a
    if w % 2 != 1:
        raise ValueError(f'smooth_trunc_torch 窗口须为奇数，得到 {w}')
    length = a.shape[-1]
    if length < w:
        raise ValueError(f'序列长度 {length} 小于窗口 {w}')
    lead_shape = a.shape[:-1]
    flat = a.reshape(-1, 1, length)
    kernel = a.new_ones(1, 1, w)
    summed = F.conv1d(flat, kernel, padding=w // 2)                        # 窗口内求和（越界处补 0）
    count = F.conv1d(a.new_ones(1, 1, length), kernel, padding=w // 2)     # 窗口内实际帧数
    return (summed / count).reshape(*lead_shape, length)


@torch.no_grad()
def svo_soft_initial_counts(
    hist: torch.Tensor,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """历史真实段的硬阈值事件计数（初始累加量 Y0, A0），不带梯度。

    逐项对齐 alpha_cumulative 的历史部分：
        sp_h = hypot(hist[HIST_REAL_START:, 3], hist[HIST_REAL_START:, 4])
        acc_h = smooth_trunc(diff(sp_h) / dt, acc_smooth)
        Y0 = sum(acc_h < -dth)，A0 = sum(acc_h > ath)
    速度与差分在 hist 自身 dtype 下算（与 numpy 对 float32 输入的行为一致），平滑与计数升到 float64。

    Args:
        hist: (T_hist, 5) 或 (B, T_hist, 5) 历史 [x, y, heading, vx, vy]。
        cfg: 软累加参数（见 default_svo_soft_cfg），只用 dt/dth/ath/acc_smooth/hist_real_start。

    Returns:
        (Y0, A0)：形状 () 或 (B,)，float64，与 hist 同 device。
    """
    c = _resolve_svo_soft_cfg(cfg)
    h = hist.detach()[..., int(c['hist_real_start']):, :]
    sp = torch.hypot(h[..., 3], h[..., 4])                                 # (..., L_h)
    acc = torch.diff(sp, dim=-1) / float(c['dt'])
    acc = smooth_trunc_torch(acc.to(torch.float64), int(c['acc_smooth']))
    y0 = (acc < -float(c['dth'])).to(torch.float64).sum(dim=-1)
    a0 = (acc > float(c['ath'])).to(torch.float64).sum(dim=-1)
    return y0, a0


def compute_svo_soft_cumulative_torch(
    dxdy: torch.Tensor,
    hist: Optional[torch.Tensor] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """可微软累加 SVO：返回逐帧 α(t)（度），对 dxdy 可导。

    步骤（dt=0.1）：
      1) speeds = ||dxdy|| / dt；acc = [0, diff(speeds)/dt]；acc[1:] 做截断窗口居中滑动平均（窗长 acc_smooth）；
      2) 软事件 yield = sigmoid((−acc − dth)·sc)，aggr = sigmoid((acc − ath)·sc)，第 0..seam_skip 帧置 0；
      3) 初始量 (Y0, A0)：hist 给出时按历史真实段硬阈值计数（不带梯度，见 svo_soft_initial_counts）；
         hist=None 时为 0（agent 按用户 09-13 决定取初始 0）；
      4) Y = Y0 + cumsum(yield)，A = A0 + cumsum(aggr)；
         x = max(1 − 0.10·Y + 0.15·A, 0.1)，y = 0.15·Y − 0.10·A，α = degrees(atan2(y, x))。
    sc → ∞ 时与「alpha_cumulative 结果再把 yf/af 第 0..seam_skip 帧置 0 后重算」一致。
    x 的下限用 clamp：与 np.maximum 同值，x<0.1 处梯度为 0。

    Args:
        dxdy: (T, 2) 或 (B, T, 2) 逐步位移增量（可带梯度）。
        hist: (T_hist, 5) 或 (B, T_hist, 5) 历史 [x, y, heading, vx, vy]（不求导），或 None（初始量 0）。
        cfg: 软累加参数，None 或缺省键取 default_svo_soft_cfg() 的值。

    Returns:
        α(t)：(T,) 或 (B, T)，单位度，dtype/device 同 dxdy。
    """
    c = _resolve_svo_soft_cfg(cfg)
    squeeze = dxdy.dim() == 2
    d = dxdy.unsqueeze(0) if squeeze else dxdy                             # (B, T, 2)
    n_batch, n_t = d.shape[0], d.shape[1]
    dt = float(c['dt'])
    sc = float(c['sc'])

    # 1) 速度与加速度；第 0 帧不计接缝（置 0），acc[1:] 截断窗口平滑
    speeds = d.norm(dim=-1) / dt                                           # (B, T)
    acc_tail = torch.diff(speeds, dim=-1) / dt                             # (B, T-1)
    acc_tail = smooth_trunc_torch(acc_tail, int(c['acc_smooth']))
    acc = torch.cat([acc_tail.new_zeros(n_batch, 1), acc_tail], dim=-1)   # (B, T)

    # 2) 软事件 + 跳过第 0..seam_skip 帧
    yld = torch.sigmoid((-acc - float(c['dth'])) * sc)
    agg = torch.sigmoid((acc - float(c['ath'])) * sc)
    n_skip = min(max(int(c['seam_skip']), 0) + 1, n_t)
    keep = acc.new_ones(n_t)
    keep[:n_skip] = 0.0
    yld = yld * keep
    agg = agg * keep

    # 3) 初始累加量（历史硬计数，无梯度）
    if hist is None:
        y0 = acc.new_zeros(n_batch)
        a0 = acc.new_zeros(n_batch)
    else:
        if hist.dim() == 3 and hist.shape[0] != n_batch:
            raise ValueError(f'hist batch {hist.shape[0]} 与 dxdy batch {n_batch} 不一致')
        y0, a0 = svo_soft_initial_counts(hist, c)
        y0 = y0.to(dtype=acc.dtype, device=acc.device).expand(n_batch)
        a0 = a0.to(dtype=acc.dtype, device=acc.device).expand(n_batch)

    # 4) 累加 → 角度
    y_cum = y0.unsqueeze(1) + torch.cumsum(yld, dim=-1)                    # (B, T)
    a_cum = a0.unsqueeze(1) + torch.cumsum(agg, dim=-1)
    d_yield, d_aggr = c['d_yield'], c['d_aggr']
    x = 1.0 + float(d_yield[0]) * y_cum + float(d_aggr[0]) * a_cum
    y = float(d_yield[1]) * y_cum + float(d_aggr[1]) * a_cum
    x = x.clamp(min=float(c['x_floor']))
    alpha = torch.rad2deg(torch.atan2(y, x))
    return alpha[0] if squeeze else alpha
