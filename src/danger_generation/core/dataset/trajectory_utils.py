"""
Trajectory utility functions for danger pair preprocessing.

Coordinate conventions (used throughout the pipeline):
  - Ego-centric frame: origin = ego position at reference time (last history frame),
    x-axis aligned with ego heading, y-axis to the left.
  - ego_future_gt[:, :2] stores per-frame displacement INCREMENTS (Δx, Δy).
    Frame 0 = displacement from origin to first future position.
    Frame t = displacement from position(t-1) to position(t).
  - key_agent_future[:, :2] stores ABSOLUTE positions in the ego-centric frame.
"""

import numpy as np
from typing import Optional


def global_to_ego_centric(
    x: float,
    y: float,
    heading: float,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> tuple[float, float, float]:
    """Transform a single pose from global to ego-centric coordinates."""
    dx = x - ego_x
    dy = y - ego_y
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    rel_x = dx * cos_h - dy * sin_h
    rel_y = dx * sin_h + dy * cos_h
    rel_heading = heading - ego_heading
    return rel_x, rel_y, rel_heading


def increments_to_positions(increments_2d: np.ndarray) -> np.ndarray:
    """Convert (Δx, Δy) increments to cumulative absolute positions.

    Args:
        increments_2d: shape (T, 2), per-frame displacements.
            increments[0] is displacement from origin (ego at t=0).

    Returns:
        positions: shape (T, 2), absolute positions in ego-centric frame.
    """
    return np.cumsum(increments_2d, axis=0)


def compute_velocity_from_increments(
    increments_2d: np.ndarray, dt: float = 0.1
) -> np.ndarray:
    """Compute velocity from per-frame displacement increments.

    Args:
        increments_2d: shape (T, 2), (Δx, Δy) per frame.
        dt: time step in seconds (default 0.1s = 10 Hz).

    Returns:
        velocity: shape (T, 2), (vx, vy) in m/s.
    """
    return increments_2d / dt


def positions_to_increments(
    positions: np.ndarray, start_pos: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convert absolute positions to per-frame displacement increments.

    Args:
        positions: shape (T, 2), absolute ego-centric positions.
        start_pos: shape (2,), position at t=-1 (before first frame).
            If None, assumes origin [0, 0] (ego at reference frame).

    Returns:
        increments: shape (T, 2), (Δx, Δy) per frame.
    """
    if start_pos is None:
        start_pos = np.zeros(2, dtype=positions.dtype)
    prev = np.vstack([start_pos[np.newaxis, :], positions[:-1]])
    return positions - prev


def build_ego_future_5col(
    ego_increments_2d: np.ndarray, dt: float = 0.1
) -> np.ndarray:
    """Build a (T, 5) ego future array from raw (Δx, Δy) increments.

    Columns: [Δx, Δy, heading, vx, vy]
    - heading: direction of motion from increment angle (atan2(Δy, Δx))
    - vx, vy: real velocity computed as increment / dt

    Args:
        ego_increments_2d: shape (T, 2), per-frame (Δx, Δy).
        dt: time step in seconds.

    Returns:
        ego_future: shape (T, 5).
    """
    T = len(ego_increments_2d)
    out = np.zeros((T, 5), dtype=np.float32)
    out[:, :2] = ego_increments_2d

    # heading from motion direction
    dx = ego_increments_2d[:, 0]
    dy = ego_increments_2d[:, 1]
    out[:, 2] = np.arctan2(dy, dx)

    # real velocity
    out[:, 3] = dx / dt  # vx
    out[:, 4] = dy / dt  # vy

    return out


def build_agent_future_5col(
    agent_positions: np.ndarray,
    agent_last_hist_pos: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """Build a (T, 5) agent future array from absolute ego-centric positions.

    Columns: [x, y, heading, vx, vy]
    Velocity is computed from consecutive position differences.

    Args:
        agent_positions: shape (T, 2) or (T, 3), absolute ego-centric positions.
            If (T, 3), the third column is heading.
        agent_last_hist_pos: shape (2,), agent's last history position (for
            computing velocity at the first future frame).
        dt: time step in seconds.

    Returns:
        agent_future: shape (T, 5).
    """
    T = len(agent_positions)
    out = np.zeros((T, 5), dtype=np.float32)
    out[:, :2] = agent_positions[:, :2]

    if agent_positions.shape[1] >= 3:
        out[:, 2] = agent_positions[:, 2]  # use NuPlan heading if available
    else:
        # compute heading from motion direction
        prev_xy = np.vstack([agent_last_hist_pos[:2][np.newaxis, :], agent_positions[:-1, :2]])
        dx = agent_positions[:, 0] - prev_xy[:, 0]
        dy = agent_positions[:, 1] - prev_xy[:, 1]
        out[:, 2] = np.arctan2(dy, dx)

    # velocity from consecutive differences
    prev_xy = np.vstack([agent_last_hist_pos[:2][np.newaxis, :], agent_positions[:-1, :2]])
    out[:, 3] = (agent_positions[:, 0] - prev_xy[:, 0]) / dt  # vx
    out[:, 4] = (agent_positions[:, 1] - prev_xy[:, 1]) / dt  # vy

    return out


def ego_future_5col_from_nuplan_positions(
    future_positions_global: np.ndarray,
    ego_ref_x: float,
    ego_ref_y: float,
    ego_ref_heading: float,
    dt: float = 0.1,
) -> np.ndarray:
    """Build ego future (T, 5) starting from raw NuPlan absolute global positions.

    Pipeline: global → ego-centric absolute → increments → (Δx, Δy, heading, vx, vy)

    Args:
        future_positions_global: shape (T, 2) or (T, 3), [x, y] or [x, y, heading]
            in global frame.
        ego_ref_x, ego_ref_y, ego_ref_heading: ego pose at reference time.
        dt: time step.

    Returns:
        ego_future: shape (T, 5).
    """
    T = len(future_positions_global)

    # transform to ego-centric absolute positions
    ego_centric = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        if future_positions_global.shape[1] >= 3:
            h = future_positions_global[t, 2]
        else:
            h = 0.0
        rx, ry, rh = global_to_ego_centric(
            future_positions_global[t, 0],
            future_positions_global[t, 1],
            h,
            ego_ref_x, ego_ref_y, ego_ref_heading,
        )
        ego_centric[t] = [rx, ry, rh]

    # convert absolute positions to per-frame increments
    # increment[0] = ego_centric[0] - origin(0,0)
    # increment[t] = ego_centric[t] - ego_centric[t-1]
    increments_2d = positions_to_increments(ego_centric[:, :2], start_pos=np.zeros(2))

    return build_ego_future_5col(increments_2d, dt=dt)
