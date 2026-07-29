"""
Safe-Sim data adapter: converts our pkl format to Safe-Sim obs_dict.

Safe-Sim model input (obs_dict) requires:
    image             : [B, C, 224, 224]  rasterized HD map + agent history
    agent_hist        : [B, T_hist, state_dim]
    neigh_hist        : [B, N, T_hist, state_dim]
    centroid          : [B, 2]
    yaw               : [B, 1]
    curr_speed        : [B, 1]
    world_from_agent  : [B, 3, 3]
    raster_from_agent : [B, 3, 3]
    dt                : float

Our pkl format:
    ego_history       : (40, 5) — [x, y, heading, vx, vy]
    key_agent_history : (40, 5)
    ego_future_gt     : (60, 5)

TODO: implement each stub below once Safe-Sim env is set up.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


# ── Map Rasterizer (Step 3 — main effort) ─────────────────────────────────────

def rasterize_map(
    ego_pos: np.ndarray,     # (2,) world coordinates
    ego_yaw: float,          # radians
    map_path: Optional[Path] = None,
    image_size: int = 224,
    resolution: float = 0.5,  # metres per pixel
) -> np.ndarray:
    """
    Render a top-down HD map centred on ego_pos.

    Returns:
        np.ndarray of shape (C, image_size, image_size), dtype float32
        Channels: [road, lane_divider, crosswalk, ...] (match Safe-Sim convention)

    TODO: integrate nuPlan map SDK to read lane/road geometry from map_path (.db file).
    """
    raise NotImplementedError(
        "Map rasterization not yet implemented. "
        "Requires nuPlan HD map files and the map SDK. "
        "See safe_sim/README.md Step 3."
    )


# ── Trajectory → agent_hist ────────────────────────────────────────────────────

def trajectory_to_agent_hist(traj: np.ndarray) -> np.ndarray:
    """
    Convert (T, 5) trajectory [x, y, heading, vx, vy] to Safe-Sim agent_hist format.
    Safe-Sim typically uses [x, y, heading] or [x, y, heading, speed] depending on config.

    Returns: (T, state_dim) float32
    """
    # Extract x, y, heading, speed
    x, y, heading = traj[:, 0], traj[:, 1], traj[:, 2]
    speed = np.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    return np.stack([x, y, heading, speed], axis=-1).astype(np.float32)


# ── World ↔ Agent transforms ───────────────────────────────────────────────────

def make_world_from_agent(pos: np.ndarray, yaw: float) -> np.ndarray:
    """3×3 homogeneous transform from agent frame to world frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, pos[0]],
        [s,  c, pos[1]],
        [0,  0,      1],
    ], dtype=np.float32)


def make_raster_from_agent(
    pos: np.ndarray,
    yaw: float,
    image_size: int = 224,
    resolution: float = 0.5,
) -> np.ndarray:
    """3×3 transform from agent frame to raster pixel coordinates."""
    cx, cy = image_size / 2.0, image_size / 2.0
    scale = 1.0 / resolution
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [ c * scale, -s * scale, cx - (c * pos[0] - s * pos[1]) * scale],
        [-s * scale, -c * scale, cy + (s * pos[0] + c * pos[1]) * scale],
        [0, 0, 1],
    ], dtype=np.float32)


# ── Main adapter ───────────────────────────────────────────────────────────────

def pkl_to_obs_dict(
    sample: Dict,
    map_path: Optional[Path] = None,
    image_size: int = 224,
    resolution: float = 0.5,
) -> Dict:
    """
    Convert a loaded pkl sample (from BaseBaseline.load_sample) to a Safe-Sim obs_dict.

    Args:
        sample     : dict with 'ego_history', 'key_agent_history', etc.
        map_path   : path to nuPlan .db map file (required for rasterize_map)
        image_size : raster image size in pixels
        resolution : metres per pixel

    Returns:
        obs_dict ready for Safe-Sim model forward pass (batch size = 1, no device move yet)

    NOTE: rasterize_map() is not yet implemented. This function will raise
    NotImplementedError until Step 3 (map rasterizer) is complete.
    """
    ego_hist = sample["ego_history"]   # (40, 5)

    # Current ego state (last history frame)
    ego_pos = ego_hist[-1, :2]
    ego_yaw = float(ego_hist[-1, 2])
    ego_speed = float(np.sqrt(ego_hist[-1, 3] ** 2 + ego_hist[-1, 4] ** 2))

    # Rasterized map (TODO: implement rasterize_map first)
    map_image = rasterize_map(ego_pos, ego_yaw, map_path, image_size, resolution)
    # map_image: (C_map, H, W)

    # Agent history tensors
    agent_hist = trajectory_to_agent_hist(ego_hist)           # (40, state_dim)
    neigh_hist = trajectory_to_agent_hist(sample["key_agent_history"])  # (40, state_dim)

    # Coordinate transforms
    world_from_agent = make_world_from_agent(ego_pos, ego_yaw)
    raster_from_agent = make_raster_from_agent(ego_pos, ego_yaw, image_size, resolution)

    return {
        "image":             map_image[None],            # (1, C, H, W)
        "agent_hist":        agent_hist[None],           # (1, T, state_dim)
        "neigh_hist":        neigh_hist[None, None],     # (1, 1, T, state_dim)
        "centroid":          ego_pos[None],              # (1, 2)
        "yaw":               np.array([[ego_yaw]]),      # (1, 1)
        "curr_speed":        np.array([[ego_speed]]),    # (1, 1)
        "world_from_agent":  world_from_agent[None],     # (1, 3, 3)
        "raster_from_agent": raster_from_agent[None],    # (1, 3, 3)
        "dt":                0.1,
    }
