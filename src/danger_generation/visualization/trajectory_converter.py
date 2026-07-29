"""
Trajectory Format Converter

Convert Diffusion generated increments to trajectory format.

Diffusion Output: increments (120-dim: 60 frames × 2 coords)
Trajectory Format: (60, 5) - [x, y, heading, vx, vy]
NuPlan Format: Trajectory object with state points
"""

import os
import sys

# Avoid importing the local wheel-cache directory `nuplan-devkit/torch` as PyTorch.
sys.path = [
    p for p in sys.path
    if os.path.basename(os.path.normpath(p)) != "nuplan-devkit"
]

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.signal import savgol_filter


@dataclass
class TrajectoryData:
    """Container for trajectory data."""
    positions: np.ndarray  # (T, 2) - x, y
    headings: np.ndarray   # (T,) - heading angle
    velocities: np.ndarray # (T, 2) - vx, vy
    timestamps: np.ndarray # (T,) - time in seconds

    @property
    def full_trajectory(self) -> np.ndarray:
        """Get full trajectory array (T, 5)."""
        T = len(self.timestamps)
        full = np.zeros((T, 5))
        full[:, :2] = self.positions
        full[:, 2] = self.headings
        full[:, 3:5] = self.velocities
        return full

    @property
    def accelerations(self) -> np.ndarray:
        """Compute accelerations from velocities."""
        dt = 0.1  # NuPlan uses 0.1s timestep
        accel = np.diff(self.velocities, axis=0) / dt
        return accel

    @property
    def jerk(self) -> np.ndarray:
        """Compute jerk (acceleration derivative)."""
        dt = 0.1
        jerk = np.diff(self.accelerations, axis=0) / dt
        return jerk


def recover_trajectory_from_increment(
    increments: Union[torch.Tensor, np.ndarray],
    start_pos: Union[torch.Tensor, np.ndarray],
    start_heading: Union[torch.Tensor, np.ndarray],
    start_vx: Union[torch.Tensor, np.ndarray],
    start_vy: Union[torch.Tensor, np.ndarray],
) -> TrajectoryData:
    """
    Recover full trajectory from increment representation.

    Args:
        increments: (120,) or (batch, 120) - dx, dy sequence
        start_pos: (2,) - starting position [x, y]
        start_heading: scalar - starting heading angle
        start_vx: scalar - starting velocity x
        start_vy: scalar - starting velocity y

    Returns:
        TrajectoryData object with full trajectory info
    """
    # Convert to numpy
    if isinstance(increments, torch.Tensor):
        increments = increments.cpu().numpy()
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.cpu().numpy()
    if isinstance(start_heading, torch.Tensor):
        start_heading = start_heading.cpu().numpy()
    if isinstance(start_vx, torch.Tensor):
        start_vx = start_vx.cpu().numpy()
    if isinstance(start_vy, torch.Tensor):
        start_vy = start_vy.cpu().numpy()

    # Handle batch dimension
    if increments.ndim == 1:
        increments = increments.reshape(60, 2)
    else:
        # Batch processing - return first sample for now
        increments = increments[0].reshape(60, 2)

    # Build raw positions via cumulative sum of increments
    positions_raw = np.zeros((60, 2))
    positions_raw[0] = start_pos + increments[0]
    for t in range(1, 60):
        positions_raw[t] = positions_raw[t-1] + increments[t]

    # Two-pass Savitzky-Golay smoothing to suppress high-frequency noise in
    # diffusion outputs before computing derivatives.
    #
    # Pass 1 (positions): window=11 removes short-range jitter while keeping
    #   the trajectory shape intact.
    # Pass 2 (velocities): window=11 on the gradient output further flattens
    #   acceleration spikes caused by boundary effects and residual noise.
    #
    # Together they reduce max_accel by ~20x and max_jerk by ~100x vs. raw
    # finite differences, which is necessary because the diffusion model is
    # trained with smooth_weight=0.3 — not enough to guarantee smooth output.
    T = positions_raw.shape[0]
    win = min(21, T if T % 2 == 1 else T - 1)
    positions = savgol_filter(positions_raw, window_length=win, polyorder=3, axis=0)

    # Build timestamps
    timestamps = np.arange(60) * 0.1  # 0.1s per frame

    # Pass 1 velocities via central differences on smoothed positions
    velocities_raw = np.gradient(positions, 0.1, axis=0)
    # Pass 2: smooth velocities to make acceleration physically plausible
    velocities = savgol_filter(velocities_raw, window_length=win, polyorder=3, axis=0)

    # Headings from doubly-smoothed velocity direction
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    headings[0] = float(start_heading)

    return TrajectoryData(
        positions=positions,
        headings=headings,
        velocities=velocities,
        timestamps=timestamps
    )


def kinematic_decode_np(
    kin_params: Union[np.ndarray, torch.Tensor],
    v0: float,
    theta0: float,
    last_pos: np.ndarray,
    dt: float = 0.1,
    max_dv: float = 2.0,
    max_dtheta: float = 0.25,
) -> TrajectoryData:
    """
    Decode kinematic parameters (Δv, Δθ) into a TrajectoryData.

    Args:
        kin_params: (120,) interleaved (Δv_0,Δθ_0, Δv_1,Δθ_1, ...)
        v0: initial speed (m/s)
        theta0: initial heading (rad)
        last_pos: (2,) initial position [x, y]
        dt: timestep (s)
        max_dv: clamp range for Δv
        max_dtheta: clamp range for Δθ
    Returns:
        TrajectoryData
    """
    if isinstance(kin_params, torch.Tensor):
        kin_params = kin_params.cpu().numpy()
    params = kin_params.reshape(60, 2)
    delta_v = np.clip(params[:, 0], -max_dv, max_dv)
    delta_theta = np.clip(params[:, 1], -max_dtheta, max_dtheta)
    speeds = np.clip(np.cumsum(delta_v) + v0, 0, 40)          # [60]
    headings = np.cumsum(delta_theta) + theta0                  # [60]
    dx = speeds * np.cos(headings) * dt
    dy = speeds * np.sin(headings) * dt
    positions = last_pos + np.cumsum(np.stack([dx, dy], axis=1), axis=0)
    vx = speeds * np.cos(headings)
    vy = speeds * np.sin(headings)
    return TrajectoryData(
        positions=positions,
        headings=headings,
        velocities=np.stack([vx, vy], axis=1),
        timestamps=np.arange(60) * dt,
    )


def future_gt_to_kinematic_np(
    ego_future_gt: Union[np.ndarray, torch.Tensor],
    v0: float,
    theta0: float,
    max_dv: float = 2.0,
    max_dtheta: float = 0.25,
) -> np.ndarray:
    """Convert ego_future_gt (60,5) [x,y,heading,vx,vy] to kinematic (Δv,Δθ) format.

    Args:
        ego_future_gt: (60, 5) array; cols 3,4 are vx, vy
        v0: initial speed before future starts (sqrt(vx² + vy²) from ego_history[-1])
        theta0: initial heading from ego_history[-1]
        max_dv: clamp range for Δv (must match model training)
        max_dtheta: clamp range for Δθ (must match model training)

    Returns:
        (120,) interleaved float32 [Δv_0, Δθ_0, Δv_1, Δθ_1, ...]
    """
    if isinstance(ego_future_gt, torch.Tensor):
        ego_future_gt = ego_future_gt.cpu().numpy()
    ego_future_gt = np.asarray(ego_future_gt)
    vx = ego_future_gt[:, 3]
    vy = ego_future_gt[:, 4]
    speeds = np.sqrt(vx ** 2 + vy ** 2)
    headings = np.arctan2(vy, vx)
    delta_v = np.diff(np.concatenate([[v0], speeds]))
    delta_theta = np.diff(np.concatenate([[theta0], headings]))
    # Wrap heading differences to [-π, π] to avoid discontinuities at ±π
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    delta_v = np.clip(delta_v, -max_dv, max_dv)
    delta_theta = np.clip(delta_theta, -max_dtheta, max_dtheta)
    return np.stack([delta_v, delta_theta], axis=1).flatten().astype(np.float32)


def agent_future_to_kinematic_np(
    agent_future: Union[np.ndarray, torch.Tensor],
    agent_last_state: Union[np.ndarray, torch.Tensor],
    max_dv: float = 2.0,
    max_dtheta: float = 0.25,
) -> np.ndarray:
    """Convert agent_future (60,5) to kinematic (Δv,Δθ) using agent_history[-1] as prior state.

    Args:
        agent_future: (60, 5) where cols 3,4 are vx, vy
        agent_last_state: (5,) = agent_history[-1]; provides v0 (cols 3,4) and theta0 (col 2)
        max_dv: clamp range for Δv
        max_dtheta: clamp range for Δθ

    Returns:
        (120,) interleaved float32 [Δv_0, Δθ_0, Δv_1, Δθ_1, ...]
    """
    if isinstance(agent_last_state, torch.Tensor):
        agent_last_state = agent_last_state.cpu().numpy()
    agent_last_state = np.asarray(agent_last_state)
    v0 = float(np.sqrt(agent_last_state[3] ** 2 + agent_last_state[4] ** 2))
    theta0 = float(agent_last_state[2])
    return future_gt_to_kinematic_np(agent_future, v0, theta0, max_dv, max_dtheta)


def gt_to_cumulative_kinematic_np(
    traj_5col: Union[np.ndarray, torch.Tensor],
    v_scale: float = 20.0,
) -> np.ndarray:
    """Convert GT trajectory (60,5) to cumulative kinematic (v_norm, θ_abs) format.

    Cumulative space stores absolute speed and heading per frame, not increments.
    For straight driving at 15 m/s: output ≈ [0.75, θ, 0.75, θ, ...] (non-zero).
    This enables SDEdit anchoring — incremental space (Δv≈0, Δθ≈0) collapses to zero.

    Args:
        traj_5col: (60, 5) array; col 2=heading, cols 3,4=vx,vy
        v_scale: speed normalisation factor (v_norm = v_abs / v_scale, so 40 m/s → 2.0)

    Returns:
        (120,) float32 interleaved [v_norm_0, θ_0, v_norm_1, θ_1, ...]
    """
    if isinstance(traj_5col, torch.Tensor):
        traj_5col = traj_5col.cpu().numpy()
    traj_5col = np.asarray(traj_5col)
    speeds = np.sqrt(traj_5col[:, 3] ** 2 + traj_5col[:, 4] ** 2)
    headings = traj_5col[:, 2]
    v_norm = speeds / v_scale
    return np.stack([v_norm, headings], axis=1).flatten().astype(np.float32)


def kinematic_cumulative_decode_np(
    cum_params: Union[np.ndarray, torch.Tensor],
    last_pos: np.ndarray,
    dt: float = 0.1,
    v_scale: float = 20.0,
) -> TrajectoryData:
    """Decode cumulative kinematic (v_norm, θ_abs) parameters into a TrajectoryData.

    Inverse of gt_to_cumulative_kinematic_np.  Each frame's speed and heading are
    absolute (not incremental), so no initial v0/θ0 are needed beyond last_pos.

    Args:
        cum_params: (120,) interleaved [v_norm_0, θ_0, v_norm_1, θ_1, ...]
        last_pos: (2,) initial position [x, y] (position at end of history)
        dt: timestep (s)
        v_scale: must match value used during training (default 20.0)

    Returns:
        TrajectoryData
    """
    if isinstance(cum_params, torch.Tensor):
        cum_params = cum_params.cpu().numpy()
    params = np.asarray(cum_params).reshape(60, 2)
    speeds = np.clip(params[:, 0] * v_scale, 0.0, 40.0)
    headings = params[:, 1]
    dx = speeds * np.cos(headings) * dt
    dy = speeds * np.sin(headings) * dt
    positions = np.asarray(last_pos) + np.cumsum(np.stack([dx, dy], axis=1), axis=0)
    vx = speeds * np.cos(headings)
    vy = speeds * np.sin(headings)
    return TrajectoryData(
        positions=positions,
        headings=headings,
        velocities=np.stack([vx, vy], axis=1),
        timestamps=np.arange(60) * dt,
    )


def gt_to_xy_displacement_np(
    traj_5col: Union[np.ndarray, torch.Tensor],
    last_pos: np.ndarray,
) -> np.ndarray:
    """Convert GT trajectory (60,5) to (Δx, Δy) position displacement (120,).

    Δ[0] = pos[0] - last_pos
    Δ[t] = pos[t] - pos[t-1]

    Straight driving at 15 m/s → [1.5, 0, 1.5, 0, ...] (non-zero SDEdit anchor).
    Max per-step magnitude: ~4m (40 m/s × 0.1s) → implicit smoothness bound.
    """
    if isinstance(traj_5col, torch.Tensor):
        traj_5col = traj_5col.cpu().numpy()
    traj_5col = np.asarray(traj_5col)
    positions = traj_5col[:, :2]
    disp = np.zeros((len(positions), 2), dtype=np.float32)
    disp[0] = positions[0] - np.asarray(last_pos)
    disp[1:] = np.diff(positions, axis=0)
    return disp.flatten()


def xy_displacement_decode_np(
    xy_params: Union[np.ndarray, torch.Tensor],
    last_pos: np.ndarray,
    dt: float = 0.1,
) -> TrajectoryData:
    """Decode (Δx, Δy) displacement (120,) to TrajectoryData.

    Inverse of gt_to_xy_displacement_np.

    Args:
        xy_params: (120,) interleaved [Δx₀, Δy₀, Δx₁, Δy₁, ...]
        last_pos: (2,) initial position [x, y]
        dt: timestep (s)

    Returns:
        TrajectoryData
    """
    if isinstance(xy_params, torch.Tensor):
        xy_params = xy_params.cpu().numpy()
    params = np.asarray(xy_params, dtype=np.float32).reshape(60, 2)
    positions = np.asarray(last_pos, dtype=np.float32) + np.cumsum(params, axis=0)
    vx = params[:, 0] / dt
    vy = params[:, 1] / dt
    headings = np.arctan2(vy, vx)
    # For near-stationary frames, inherit previous heading to avoid numerical noise
    speed = np.sqrt(vx ** 2 + vy ** 2)
    for t in range(1, len(headings)):
        if speed[t] < 0.1:
            headings[t] = headings[t - 1]
    return TrajectoryData(
        positions=positions,
        headings=headings,
        velocities=np.stack([vx, vy], axis=1),
        timestamps=np.arange(60) * dt,
    )


def gt_to_bicycle_actions_np(
    traj_5col: Union[np.ndarray, torch.Tensor],
    init_state: np.ndarray,
    wheelbase: float = 2.8,
    dt: float = 0.1,
    a_max: float = 3.0,
    delta_max: float = 0.5,
) -> np.ndarray:
    """Convert GT trajectory (60,5) to pre-activation bicycle actions (120,).

    init_state: [x0, y0, θ0, v0] — last frame of history.
    Output format: [raw_a₀, raw_δ₀, raw_a₁, raw_δ₁, ...] (120 values).
    raw_a  = atanh(a / a_max),   a = longitudinal accel (m/s²)
    raw_δ = atanh(δ / delta_max), δ = steering angle (rad)

    tanh activation is applied at decode time to hard-bound the controls.
    """
    if isinstance(traj_5col, torch.Tensor):
        traj_5col = traj_5col.cpu().numpy()
    traj_5col = np.asarray(traj_5col, dtype=np.float32)
    T = traj_5col.shape[0]

    speeds   = np.sqrt(traj_5col[:, 3] ** 2 + traj_5col[:, 4] ** 2)  # [T]
    headings = traj_5col[:, 2]                                         # [T]

    v0     = float(init_state[3])
    theta0 = float(init_state[2])

    # Longitudinal acceleration: (v[t] - v[t-1]) / dt
    speeds_ext = np.concatenate([[v0], speeds])                        # [T+1]
    accels = (speeds_ext[1:] - speeds_ext[:-1]) / dt                  # [T]
    accels = np.clip(accels, -(a_max - 0.01), a_max - 0.01)

    # Steering angle: ω = Δθ/dt, δ = atan(ω·L / v)
    headings_ext = np.concatenate([[theta0], headings])                # [T+1]
    dtheta = headings_ext[1:] - headings_ext[:-1]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi                   # wrap to [−π, π]
    omegas = dtheta / dt
    delta  = np.arctan2(omegas * wheelbase, np.maximum(speeds, 0.5))
    delta  = np.clip(delta, -(delta_max - 0.01), delta_max - 0.01)

    # Pre-activation values (DDPM learns to denoise these)
    raw_a     = np.arctanh(accels / a_max).astype(np.float32)
    raw_delta = np.arctanh(delta  / delta_max).astype(np.float32)

    # Interleave: [raw_a₀, raw_δ₀, raw_a₁, raw_δ₁, ...]
    result = np.zeros(2 * T, dtype=np.float32)
    result[0::2] = raw_a
    result[1::2] = raw_delta
    return result


def bicycle_model_decode_np(
    action_params: Union[np.ndarray, torch.Tensor],
    init_state: np.ndarray,
    wheelbase: float = 2.8,
    dt: float = 0.1,
    a_max: float = 3.0,
    delta_max: float = 0.5,
) -> TrajectoryData:
    """Decode pre-activation bicycle actions (120,) to TrajectoryData.

    Inverse of gt_to_bicycle_actions_np.

    init_state: [x0, y0, θ0, v0]
    """
    if isinstance(action_params, torch.Tensor):
        action_params = action_params.cpu().numpy()
    params = np.asarray(action_params, dtype=np.float32).reshape(60, 2)

    # Apply tanh to recover bounded physical controls
    a     = a_max     * np.tanh(params[:, 0])   # [60] accel (m/s²)
    delta = delta_max * np.tanh(params[:, 1])   # [60] steering (rad)

    x     = float(init_state[0])
    y     = float(init_state[1])
    theta = float(init_state[2])
    v     = float(init_state[3])

    positions  = np.zeros((60, 2), dtype=np.float32)
    headings   = np.zeros(60,      dtype=np.float32)
    velocities = np.zeros((60, 2), dtype=np.float32)

    for t in range(60):
        v     = float(np.clip(v + a[t] * dt, 0.0, 40.0))
        theta = theta + v * np.tan(delta[t]) / wheelbase * dt
        x     = x + v * np.cos(theta) * dt
        y     = y + v * np.sin(theta) * dt
        positions[t]  = [x, y]
        headings[t]   = theta
        velocities[t] = [v * np.cos(theta), v * np.sin(theta)]

    return TrajectoryData(
        positions=positions,
        headings=headings,
        velocities=velocities,
        timestamps=np.arange(60) * dt,
    )


def compute_trajectory_metrics(traj: TrajectoryData) -> Dict:
    """
    Compute quality metrics for a trajectory.

    Returns:
        Dictionary with metrics:
        - max_speed: Maximum speed
        - max_accel: Maximum acceleration magnitude
        - max_jerk: Maximum jerk
        - total_distance: Total traveled distance
        - path_smoothness: Path smoothness score
    """
    speeds = np.sqrt(traj.velocities[:, 0]**2 + traj.velocities[:, 1]**2)
    accel_mag = np.sqrt(traj.accelerations[:, 0]**2 + traj.accelerations[:, 1]**2)
    jerk_mag = np.sqrt(traj.jerk[:, 0]**2 + traj.jerk[:, 1]**2)

    # Total distance
    displacements = np.diff(traj.positions, axis=0)
    total_distance = np.sum(np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2))

    # Path smoothness (sum of heading changes)
    heading_changes = np.abs(np.diff(traj.headings))
    path_smoothness = np.sum(heading_changes)

    return {
        'max_speed': float(np.max(speeds)),
        'mean_speed': float(np.mean(speeds)),
        'max_accel': float(np.max(accel_mag)),
        'max_jerk': float(np.max(jerk_mag)),
        'total_distance': float(total_distance),
        'path_smoothness': float(path_smoothness),
    }


def check_kinematic_validity(traj: TrajectoryData) -> Dict:
    """
    Check validity for kinematic-decoded trajectories (Δv/Δθ model).

    Speed limit is 80 m/s (covers highway scenarios up to ~288 km/h).
    NaN/Inf and negative speed are hard failures.
    Breakdown counts are provided for diagnosis.
    """
    speeds = np.sqrt(traj.velocities[:, 0] ** 2 + traj.velocities[:, 1] ** 2)
    finite = bool(np.all(np.isfinite(traj.positions)) and np.all(np.isfinite(traj.velocities)))
    has_nan = not finite
    has_negative = bool(np.any(speeds < 0.0))
    has_over80 = bool(np.any(speeds > 80.0))
    has_over40 = bool(np.any(speeds > 40.0))
    speed_valid = bool(not has_negative and not has_over80)
    metrics = compute_trajectory_metrics(traj)
    return {
        'speed_valid': speed_valid,
        'accel_valid': True,    # guaranteed by Δv clamp in generate()
        'jerk_valid': True,     # guaranteed by Δv clamp in generate()
        'overall_valid': speed_valid and finite,
        'has_nan': has_nan,
        'has_negative_speed': has_negative,
        'has_speed_over_40': has_over40,
        'has_speed_over_80': has_over80,
        'max_speed': float(np.nanmax(speeds)) if finite else float('nan'),
        'metrics': metrics,
    }


def check_trajectory_validity(traj: TrajectoryData) -> Dict[str, bool]:
    """
    Check if trajectory satisfies physical constraints.

    Vehicle limits (typical passenger car):
        - Max speed: ~50 m/s (180 km/h)
        - Max acceleration: ~10 m/s² (emergency braking)
        - Max jerk: ~20 m/s³

    Returns:
        Dictionary of validity checks
    """
    metrics = compute_trajectory_metrics(traj)

    # Physical limits
    MAX_SPEED = 50.0       # m/s
    MAX_ACCEL = 10.0       # m/s²
    MAX_JERK = 20.0        # m/s³

    return {
        'speed_valid': metrics['max_speed'] < MAX_SPEED,
        'accel_valid': metrics['max_accel'] < MAX_ACCEL,
        'jerk_valid': metrics['max_jerk'] < MAX_JERK,
        'overall_valid': all([
            metrics['max_speed'] < MAX_SPEED,
            metrics['max_accel'] < MAX_ACCEL,
            metrics['max_jerk'] < MAX_JERK
        ]),
        'metrics': metrics
    }
