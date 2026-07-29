"""
Random Perturbation baseline.

Adds independent Gaussian noise to each timestep of the GT ego trajectory.
- Position (x, y): σ = pos_sigma metres
- Speed magnitude: σ = speed_sigma m/s, clamped to [0, 80]
- Heading: unchanged (preserves directional consistency)
- vx, vy reconstructed from perturbed speed and original heading
"""

from typing import Dict

import numpy as np

from baselines.base_baseline import BaseBaseline


class RandomPerturbation(BaseBaseline):
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        pos_sigma: float = 0.5,
        speed_sigma: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(data_dir, output_dir)
        self.pos_sigma = pos_sigma
        self.speed_sigma = speed_sigma
        self.rng = np.random.default_rng(seed)

    def generate(self, sample: Dict) -> np.ndarray:
        """
        Perturb GT ego_future (60, 5) with Gaussian noise.

        Columns: [x, y, heading, vx, vy]
        """
        gt = sample["ego_future_gt"].copy()  # (60, 5)

        # Perturb positions
        gt[:, :2] += self.rng.normal(0.0, self.pos_sigma, gt[:, :2].shape).astype(np.float32)

        # Perturb speed magnitude
        speeds = np.sqrt(gt[:, 3] ** 2 + gt[:, 4] ** 2)
        speeds += self.rng.normal(0.0, self.speed_sigma, speeds.shape).astype(np.float32)
        speeds = np.clip(speeds, 0.0, 80.0)

        # Reconstruct vx, vy from perturbed speed and original heading
        headings = gt[:, 2]
        gt[:, 3] = speeds * np.cos(headings)
        gt[:, 4] = speeds * np.sin(headings)

        return gt
