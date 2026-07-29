"""
Safe-Sim standalone inference wrapper.

Bypasses the full rollout_episodes() closed-loop simulation and runs a single
model forward pass to generate adversarial trajectories from a batch of scenarios.

Prerequisites (see README.md):
  1. safe-sim repo cloned and installed (including trajdata fork)
  2. Pretrained checkpoint downloaded
  3. adapter.py rasterize_map() implemented (map rasterizer)

Usage (once prerequisites are met):
    python baselines/safe_sim/inference.py \\
        --ckpt_dir /path/to/safe_sim_checkpoint \\
        --data_dir data/danger_pair_scenarios \\
        --map_dir /path/to/nuplan/maps \\
        --num_samples 40 \\
        --output_dir experiments/danger_generation/baselines/safe_sim/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

_GNN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

_REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "config.py").exists()),
    _GNN_ROOT,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config


def load_safe_sim_model(ckpt_dir: str, device: str = "cpu"):
    """
    Load the Safe-Sim RasterizedDiffusionModel from a checkpoint.

    TODO: adapt once safe-sim repo is installed.
    Approximate code:
        from tbsim.models.RasterizedDiffusionModel import RasterizedDiffusionModel
        from tbsim.configs.algo_config import DiffusionConfig
        model = RasterizedDiffusionModel.load_from_checkpoint(ckpt_dir)
        model.eval()
        return model.to(device)
    """
    raise NotImplementedError(
        "Safe-Sim model loading not yet implemented. "
        "Install safe-sim repo first (see README.md Step 1-2)."
    )


def run_inference(model, obs_dict: Dict, n_samples: int = 8) -> np.ndarray:
    """
    Run a single forward pass of the Safe-Sim diffusion model.

    Args:
        model     : loaded RasterizedDiffusionModel
        obs_dict  : output of adapter.pkl_to_obs_dict() (numpy, batch=1)
        n_samples : number of diffusion samples (N best-of-N)

    Returns:
        trajectories : (n_samples, T, 3) — [x, y, yaw] in world frame

    TODO: move tensors to device, call model.sample() or guided_p_sample().
    Approximate code:
        import torch
        obs_tensors = {k: torch.tensor(v).to(device) for k, v in obs_dict.items()
                       if isinstance(v, np.ndarray)}
        with torch.no_grad():
            out = model.sample(obs_tensors, n_samples=n_samples)
        return out["trajectories"].cpu().numpy()
    """
    raise NotImplementedError("Inference wrapper not yet implemented.")


def convert_output_to_traj5(
    traj_xyz: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Convert Safe-Sim output (T, 3) [x, y, yaw] to our (T, 5) [x, y, heading, vx, vy].

    Velocities are estimated from finite differences of positions.
    """
    T = len(traj_xyz)
    traj5 = np.zeros((T, 5), dtype=np.float32)
    traj5[:, :2] = traj_xyz[:, :2]   # x, y
    traj5[:, 2] = traj_xyz[:, 2]     # heading (yaw)

    # Finite-difference velocity
    dx = np.diff(traj5[:, 0], prepend=traj5[0, 0])
    dy = np.diff(traj5[:, 1], prepend=traj5[0, 1])
    traj5[:, 3] = dx / dt  # vx
    traj5[:, 4] = dy / dt  # vy

    return traj5


# ── CLI entry point (placeholder) ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_dir", default=config.DANGER_PAIR_SCENARIOS_DIR)
    parser.add_argument("--map_dir", default=None, help="Path to nuPlan map .db files")
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--n_diffusion_samples", type=int, default=8)
    parser.add_argument("--output_dir", default=str(Path(config.DANGER_GENERATION_BASELINES_DIR) / "safe_sim"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise NotImplementedError(
        "Safe-Sim inference is not yet runnable. "
        "Complete README.md Steps 1-5 first."
    )
