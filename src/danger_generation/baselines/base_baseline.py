"""
Abstract base class for adversarial trajectory baselines.

Handles pkl data loading (mirrors matplotlib_visualizer.py load_sample logic)
and metric computation (danger_score + standard_metrics), so concrete baselines
only need to implement generate().
"""

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Path setup: allow imports from GNN_Trans root ─────────────────────────────
_GNN_ROOT = Path(__file__).resolve().parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

from utils.danger_score import compute_danger_score
from visualization.standard_metrics import (
    compute_collision,
    compute_min_distance,
    compute_ttc,
)


class BaseBaseline(ABC):
    """
    Subclass and implement generate(sample) -> np.ndarray (60, 5).
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ───────────────────────────────────────────────────────────

    def _list_pkl_files(self) -> List[Path]:
        files = sorted(self.data_dir.glob("danger_pair_features_*.pkl"))
        if not files:
            raise FileNotFoundError(f"No pkl files found in {self.data_dir}")
        return files

    def load_sample(self, pkl_file: Path) -> Optional[Dict]:
        """
        Load one pkl sample.  Returns None if required fields are missing.
        Output keys:
            ego_history          (40, 5)
            key_agent_history    (40, 5)
            ego_future_gt        (60, 5)
            key_agent_future     (60, 5) or None
            scenario_name        str
        """
        import pandas as pd

        try:
            data = pd.read_pickle(pkl_file)
        except Exception as e:
            logger.warning("Failed to load %s: %s", pkl_file.name, e)
            return None

        ego_history = data.get("ego_history")
        agent_history = data.get("key_agent_history")
        ego_future_gt = data.get("ego_future_gt")

        if ego_history is None or agent_history is None or ego_future_gt is None:
            logger.warning("Missing required fields in %s", pkl_file.name)
            return None

        if ego_future_gt.ndim == 3:
            ego_future_gt = ego_future_gt.squeeze(0)

        raw_agent_future = data.get("key_agent_future")
        agent_future = self._normalize_agent_future(
            raw_agent_future, np.asarray(agent_history, dtype=np.float32)
        )

        return {
            "ego_history": np.asarray(ego_history, dtype=np.float32),
            "key_agent_history": np.asarray(agent_history, dtype=np.float32),
            "ego_future_gt": np.asarray(ego_future_gt[:60], dtype=np.float32),
            "key_agent_future": agent_future,  # always (60, 5) after normalisation
            "scenario_name": data.get("scenario_name", pkl_file.stem),
        }

    @staticmethod
    def _normalize_agent_future(
        agent_future: Optional[np.ndarray], agent_history: np.ndarray
    ) -> np.ndarray:
        """
        Mirrors matplotlib_visualizer._normalize_agent_future.
        When agent_future is None, returns a frozen trajectory at the agent's
        last known state (constant position / velocity).
        """
        if agent_future is None:
            frozen = np.zeros((60, 5), dtype=np.float32)
            frozen[:, :2] = agent_history[-1, :2]
            frozen[:, 2] = agent_history[-1, 2]
            frozen[:, 3] = agent_history[-1, 3]
            frozen[:, 4] = agent_history[-1, 4]
            return frozen

        agent_future = np.asarray(agent_future, dtype=np.float32)
        if agent_future.ndim == 3:
            agent_future = agent_future.squeeze(0)

        num_frames = min(len(agent_future), 60)
        full = np.zeros((60, 5), dtype=np.float32)
        width = min(agent_future.shape[1], 5)
        full[:num_frames, :width] = agent_future[:num_frames, :width]
        if num_frames < 60:
            full[num_frames:, :2] = full[max(num_frames - 1, 0), :2]
        if agent_future.shape[1] < 3:
            full[:, 2] = agent_history[-1, 2]
        if agent_future.shape[1] < 5:
            full[:, 3] = agent_history[-1, 3]
            full[:, 4] = agent_history[-1, 4]
        return full

    # ── Metric computation ─────────────────────────────────────────────────────

    def compute_metrics(
        self,
        gen_traj: np.ndarray,
        gt_traj: np.ndarray,
        sample: Dict,
    ) -> Dict:
        """
        Returns danger scores and standard metrics for one sample.

        danger_score uses outcome_danger['total_score'].
        Standard metrics (collision/TTC/distance) are computed against agent_future
        when available; skipped otherwise.
        """
        ego_h = sample["ego_history"]
        agent_h = sample["key_agent_history"]
        agent_f = sample["key_agent_future"]

        # agent_future is always (60,5) after _normalize_agent_future in load_sample
        gt_result = compute_danger_score(ego_h, gt_traj, agent_h, agent_f)
        gen_result = compute_danger_score(ego_h, gen_traj, agent_h, agent_f)

        gt_danger = float(gt_result["outcome_danger"]["total_score"])
        gen_danger = float(gen_result["outcome_danger"]["total_score"])

        return {
            "gt_danger": gt_danger,
            "gen_danger": gen_danger,
            "danger_improvement": gen_danger - gt_danger,
            "margin_met": (gen_danger - gt_danger) > 0.5,
            "collision": compute_collision(gen_traj, agent_f),
            "min_distance": compute_min_distance(gen_traj, agent_f),
            "ttc": compute_ttc(gen_traj, agent_f),
        }

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def generate(self, sample: Dict) -> np.ndarray:
        """Return a generated ego trajectory (60, 5): [x, y, heading, vx, vy]."""

    # ── Evaluation loop ────────────────────────────────────────────────────────

    def evaluate(self, num_samples: int) -> Dict:
        """
        Run evaluation over up to num_samples pkl files.
        Returns aggregated metrics dict and saves results JSON.
        """
        pkl_files = self._list_pkl_files()[:num_samples]
        logger.info("Evaluating %d samples with %s", len(pkl_files), self.__class__.__name__)

        per_sample: List[Dict] = []
        skipped = 0

        for pkl_file in pkl_files:
            sample = self.load_sample(pkl_file)
            if sample is None:
                skipped += 1
                continue

            try:
                gen_traj = self.generate(sample)
            except Exception as e:
                logger.warning("generate() failed for %s: %s", pkl_file.name, e)
                skipped += 1
                continue

            metrics = self.compute_metrics(gen_traj, sample["ego_future_gt"], sample)
            metrics["scenario_name"] = sample["scenario_name"]
            per_sample.append(metrics)

        if not per_sample:
            raise RuntimeError("No samples evaluated successfully.")

        n = len(per_sample)
        valid_collisions = [m["collision"] for m in per_sample]
        valid_distances = [m["min_distance"] for m in per_sample]
        valid_ttcs = [m["ttc"] for m in per_sample if np.isfinite(m["ttc"])]

        summary = {
            "baseline": self.__class__.__name__,
            "n_evaluated": n,
            "n_skipped": skipped,
            "margin_met_rate": float(sum(m["margin_met"] for m in per_sample) / n),
            "mean_gt_danger": float(np.mean([m["gt_danger"] for m in per_sample])),
            "mean_gen_danger": float(np.mean([m["gen_danger"] for m in per_sample])),
            "mean_danger_improvement": float(np.mean([m["danger_improvement"] for m in per_sample])),
            "collision_rate": float(np.mean(valid_collisions)) if valid_collisions else float("nan"),
            "mean_min_distance": float(np.mean(valid_distances)) if valid_distances else float("nan"),
            "mean_ttc": float(np.mean(valid_ttcs)) if valid_ttcs else float("inf"),
            "near_miss_rate": float(len(valid_ttcs) / n),
            "per_sample": per_sample,
        }

        out_path = self.output_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Results saved to %s", out_path)

        return summary
