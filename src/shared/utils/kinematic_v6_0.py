"""
[v6.0] Kinematic parameterization utilities for 六C retraining.

Re-exports existing NuPlan kinematic encode/decode with v6.0 labels and
provides the per-step clamp values that structurally enforce physical validity
at inference time.

Training space (六C): (Δv, Δθ) per step, interleaved: [Δv_0, Δθ_0, …, Δv_59, Δθ_59]
  - Δv  ∈ [-MAX_DV_V6_0,  +MAX_DV_V6_0]   (m/s speed change per step, dt=0.1s)
  - Δθ  ∈ [-MAX_DTH_V6_0, +MAX_DTH_V6_0]  (rad heading change per step)

These values match `kinematic_decode_np` defaults in trajectory_converter.py
and should NOT be changed without retraining.
"""

import math
import numpy as np
import torch
from typing import Union

# ── v6.0 kinematic clamp constants ─────────────────────────────────────────
MAX_DV_V6_0   = 2.0    # m/s per step — speed change; matches trajectory_converter
MAX_DTH_V6_0  = 0.25   # rad per step — heading change (~14.3°/step, ≈ 143°/s)

# Derived: maximum trajectory step length (m/step) at maximum cumulative speed
# The model output is (Δv, Δθ) so absolute speed is bounded only by the dataset
# distribution; the clamp on Δv prevents runaway acceleration.


# ── Encode: (x,y,heading,vx,vy) → kinematic flat ──────────────────────────

def encode_ego_kinematic_v6_0(
    ego_future_gt: Union[np.ndarray, torch.Tensor],
    ego_history: np.ndarray,
) -> np.ndarray:
    """[v6.0] Encode ego GT future trajectory as kinematic (Δv, Δθ) increments.

    Wrapper around `future_gt_to_kinematic_np` with v6.0 clamp constants.

    Args:
        ego_future_gt: (60, 5) [x,y,heading,vx,vy] ego future trajectory.
        ego_history: (T_hist, 5) ego history; last row provides v0 and theta0.

    Returns:
        (120,) float32 — interleaved [Δv_0,Δθ_0, Δv_1,Δθ_1, …] in m/s and rad.
    """
    from danger_generation.visualization.trajectory_converter import (
        future_gt_to_kinematic_np,
    )
    v0     = float(np.sqrt(ego_history[-1, 3] ** 2 + ego_history[-1, 4] ** 2))
    theta0 = float(ego_history[-1, 2])
    return future_gt_to_kinematic_np(
        ego_future_gt, v0, theta0,
        max_dv=MAX_DV_V6_0, max_dtheta=MAX_DTH_V6_0,
    )


def encode_agent_kinematic_v6_0(
    agent_future: Union[np.ndarray, torch.Tensor],
    agent_history: np.ndarray,
) -> np.ndarray:
    """[v6.0] Encode agent GT future trajectory as kinematic (Δv, Δθ) increments.

    Args:
        agent_future: (60, 5) [x,y,heading,vx,vy] agent future trajectory.
        agent_history: (T_hist, 5) agent history.

    Returns:
        (120,) float32 — interleaved [Δv_0,Δθ_0, …].
    """
    from danger_generation.visualization.trajectory_converter import (
        agent_future_to_kinematic_np,
    )
    return agent_future_to_kinematic_np(
        agent_future, agent_history[-1],
        max_dv=MAX_DV_V6_0, max_dtheta=MAX_DTH_V6_0,
    )


# ── Decode: kinematic flat → (Δx, Δy) per step ────────────────────────────

def decode_kinematic_to_dxdy_v6_0(
    kinematic_flat: Union[np.ndarray, torch.Tensor],
    ego_history: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """[v6.0] Decode kinematic (Δv, Δθ) into (Δx, Δy) position increments.

    Args:
        kinematic_flat: (120,) interleaved [Δv_0,Δθ_0, …, Δv_59,Δθ_59].
        ego_history: (T_hist, 5) — last row provides start speed and heading.
        dt: timestep in seconds (default 0.1).

    Returns:
        (60, 2) float32 — (Δx, Δy) per step in metres.
    """
    from danger_generation.visualization.trajectory_converter import (
        kinematic_decode_np,
    )
    if isinstance(kinematic_flat, torch.Tensor):
        kinematic_flat = kinematic_flat.cpu().numpy()

    v0     = float(np.sqrt(ego_history[-1, 3] ** 2 + ego_history[-1, 4] ** 2))
    theta0 = float(ego_history[-1, 2])
    last_pos = ego_history[-1, :2].astype(np.float32)

    traj_data = kinematic_decode_np(
        kinematic_flat, v0, theta0, last_pos, dt=dt,
        max_dv=MAX_DV_V6_0, max_dtheta=MAX_DTH_V6_0,
    )
    # traj_data.positions: (60, 2) absolute positions → convert to increments
    pos = traj_data.positions.astype(np.float32)        # (60, 2)
    dxdy = np.zeros_like(pos)
    dxdy[0]  = pos[0] - last_pos
    dxdy[1:] = pos[1:] - pos[:-1]
    return dxdy


# ── Inference-time clamp for kinematic model outputs ──────────────────────

def clamp_kinematic_output_v6_0(
    x: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """[v6.0] Per-step clamp on kinematic model output (Δv, Δθ) × 60 per agent.

    Structurally enforces physical validity at inference time.

    Args:
        x: (B, 240) model output in kinematic space — ego_120 ‖ agent_120,
           each 120 = interleaved (Δv_0,Δθ_0, …, Δv_59,Δθ_59).
        batch_size: B.

    Returns:
        (B, 240) clamped output.
    """
    x_view = x.view(batch_size, 60, 4)   # (B, 60, [Δv_ego, Δθ_ego, Δv_agent, Δθ_agent])
    x_view[:, :, 0].clamp_(-MAX_DV_V6_0,  MAX_DV_V6_0)   # Δv_ego
    x_view[:, :, 1].clamp_(-MAX_DTH_V6_0, MAX_DTH_V6_0)  # Δθ_ego
    x_view[:, :, 2].clamp_(-MAX_DV_V6_0,  MAX_DV_V6_0)   # Δv_agent
    x_view[:, :, 3].clamp_(-MAX_DTH_V6_0, MAX_DTH_V6_0)  # Δθ_agent
    return x_view.view(batch_size, 240)
