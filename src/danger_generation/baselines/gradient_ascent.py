"""
Gradient Ascent baseline.

Starting from the GT ego trajectory, runs gradient ascent on danger_score
(via the differentiable PyTorch version) to find a more dangerous trajectory.

Optimises a delta over the GT trajectory so that the starting point is always
the ground-truth trajectory (delta = 0 initially).

After optimisation, speeds are clamped to [0, 80] m/s and vx/vy are
normalised to be consistent with the clamped speed and existing heading.
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

_GNN_ROOT = Path(__file__).resolve().parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

try:
    import torch
    from utils.danger_score_torch import compute_danger_score_torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available; GradientAscent will raise on generate()")

from baselines.base_baseline import BaseBaseline


class GradientAscent(BaseBaseline):
    """
    Gradient ascent on outcome_danger using danger_score_torch.

    Args:
        lr        Adam learning rate (default 0.05)
        n_steps   Gradient ascent steps (default 200)
        device    'cuda' | 'cpu' (default 'cpu')
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        lr: float = 0.05,
        n_steps: int = 200,
        device: str = "cpu",
    ):
        super().__init__(data_dir, output_dir)
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GradientAscent baseline.")
        self.lr = lr
        self.n_steps = n_steps
        self.device = torch.device(device)

    def _to_tensor(self, arr: np.ndarray) -> "torch.Tensor":
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def generate(self, sample: Dict) -> np.ndarray:
        """
        Run gradient ascent starting from sample['ego_future_gt'].
        Returns optimised trajectory (60, 5): [x, y, heading, vx, vy].
        """
        ego_h = self._to_tensor(sample["ego_history"]).unsqueeze(0)        # (1, 40, 5)
        agent_h = self._to_tensor(sample["key_agent_history"]).unsqueeze(0) # (1, 40, 5)

        # key_agent_future is always (60,5) after BaseBaseline.load_sample normalisation
        agent_f = self._to_tensor(sample["key_agent_future"]).unsqueeze(0)  # (1, 60, 5)

        ego_f_base = self._to_tensor(sample["ego_future_gt"])  # (60, 5), no grad

        # Learnable delta initialised at zero
        delta = torch.zeros_like(ego_f_base, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for _ in range(self.n_steps):
            optimizer.zero_grad()
            ego_f = (ego_f_base + delta).unsqueeze(0)   # (1, 60, 5)
            result = compute_danger_score_torch(ego_h, ego_f, agent_h, agent_f)
            loss = -result["outcome_danger"].mean()      # maximise danger
            loss.backward()
            optimizer.step()

        # Final trajectory (no batch dim)
        gen_traj = (ego_f_base + delta).detach().cpu().numpy()  # (60, 5)

        # Clamp speed to [0, 80] and normalise vx, vy
        speeds = np.sqrt(gen_traj[:, 3] ** 2 + gen_traj[:, 4] ** 2)
        speeds_clamped = np.clip(speeds, 0.0, 80.0)
        scale = np.where(speeds > 1e-6, speeds_clamped / speeds, 1.0)
        gen_traj[:, 3] *= scale
        gen_traj[:, 4] *= scale

        return gen_traj.astype(np.float32)
