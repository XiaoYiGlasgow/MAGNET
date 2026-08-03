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
from typing import Optional

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
