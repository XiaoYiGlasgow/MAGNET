"""
Classifier Guidance baseline (inspired by Safe-Sim, ECCV 2024).

Wraps the trained MatplotlibVisualizer (same checkpoint as main evaluation)
and applies danger-score gradient guidance during DDPM reverse sampling.

Key idea:
  - Same pretrained model as v3.10 (identical weights, identical condition encoding)
  - At each guided denoising step, shift x_0_pred toward higher danger
  - Tests whether training-time danger supervision is necessary vs. inference-time guidance
"""

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

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

from baselines.base_baseline import BaseBaseline
from baselines.classifier_guidance.guided_sampler import guided_generate
from visualization.trajectory_converter import recover_trajectory_from_increment
from utils.danger_score import compute_danger_score


class ClassifierGuidanceBaseline(BaseBaseline):
    """
    Test-time classifier guidance on the pretrained diffusion model.

    Reuses MatplotlibVisualizer for model loading and scene/condition encoding
    to support both simple and gnn_trans encoder modes without duplication.

    Args:
        model_path          : path to .pt checkpoint (v3.10 best_model.pt)
        data_dir            : danger_pair_scenarios/ directory
        processed_data_dir  : processed graph data for gnn_trans encoder
        output_dir          : where to save results.json
        guidance_scale      : gradient scale at each guided step (default 50.0)
        guide_every         : apply guidance every N denoising steps (default 10)
        device              : 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model_path: str,
        data_dir: str,
        output_dir: str,
        processed_data_dir: str = config.PROCESSED_DATA_DIR,
        guidance_scale: float = 50.0,
        guide_every: int = 10,
        device: str = "cpu",
    ):
        super().__init__(data_dir, output_dir)
        self.guidance_scale = guidance_scale
        self.guide_every    = guide_every

        # Reuse MatplotlibVisualizer for model loading (handles all sys.path / encoder logic)
        from visualization.matplotlib_visualizer import MatplotlibVisualizer
        self.viz = MatplotlibVisualizer(
            model_path        = model_path,
            data_dir          = data_dir,
            processed_data_dir= processed_data_dir,
            device            = device,
            min_gt_danger     = 0.0,   # no danger filter; we evaluate all samples
        )
        self.diffusion_model = self.viz.diffusion_model
        self.device = torch.device(device)

        logger.info(
            "ClassifierGuidanceBaseline ready: encoder=%s, traj_dim=%d, "
            "guidance_scale=%.1f, guide_every=%d",
            self.viz.scene_encoder_mode,
            self.diffusion_model.config.trajectory_dim,
            guidance_scale,
            guide_every,
        )

    # ── Override load_sample to include graph_data for gnn_trans ─────────────

    def load_sample(self, pkl_file: Path):
        """
        Load sample via MatplotlibVisualizer (handles graph_data for gnn_trans),
        then normalise agent_future using BaseBaseline logic.
        """
        sample = self.viz.load_sample(pkl_file)
        if sample is None:
            return None

        # MatplotlibVisualizer uses 'agent_history'; normalise to BaseBaseline key convention
        if "agent_history" in sample and "key_agent_history" not in sample:
            sample["key_agent_history"] = sample["agent_history"]

        # Normalise agent_future (frozen trajectory when None)
        agent_future = sample.get("key_agent_future")
        agent_history = np.asarray(sample["key_agent_history"], dtype=np.float32)
        sample["key_agent_future"] = self._normalize_agent_future(agent_future, agent_history)

        # Ensure float32
        for k in ("ego_history", "key_agent_history", "ego_future_gt"):
            if k in sample:
                sample[k] = np.asarray(sample[k], dtype=np.float32)

        return sample

    # ── Condition encoding ────────────────────────────────────────────────────

    def _get_condition(self, sample: Dict) -> torch.Tensor:
        """
        Encode scene + danger targets → condition vector (1, 256).
        Delegates to the visualizer's scene encoder (supports both simple/gnn_trans).
        """
        ego_h   = torch.tensor(sample["ego_history"],        dtype=torch.float32, device=self.device)
        agent_h = torch.tensor(sample["key_agent_history"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.viz.scene_encoder_mode == "simple":
                scene_feat = self.viz.scene_encoder(ego_h.unsqueeze(0), agent_h.unsqueeze(0))
            elif self.viz.scene_encoder_mode == "gnn_trans":
                graph_data = sample.get("graph_data")
                if graph_data is None:
                    raise RuntimeError("graph_data missing for gnn_trans encoder")
                scene_feat = self.viz.scene_encoder.get_encoder_features(graph_data)
                scene_feat = scene_feat.to(self.device)
            else:
                raise RuntimeError(f"Unknown encoder mode: {self.viz.scene_encoder_mode}")

            # Compute inherent danger for condition (mirrors generate_trajectory)
            gt_result = compute_danger_score(
                sample["ego_history"],
                sample["ego_future_gt"],
                sample["key_agent_history"],
                sample["key_agent_future"],
            )
            inherent_danger = float(gt_result["inherent_danger"]["total_score"])
            danger_target   = inherent_danger * 2.0

            svo_t      = torch.tensor([[sample["svo_angle"]]], dtype=torch.float32, device=self.device)
            inherent_t = torch.tensor([[inherent_danger]],               dtype=torch.float32, device=self.device)
            target_t   = torch.tensor([[danger_target]],                 dtype=torch.float32, device=self.device)

            condition = self.viz.condition_encoder(scene_feat, svo_t, inherent_t, target_t)

        return condition  # (1, 256)

    # ── BaseBaseline interface ────────────────────────────────────────────────

    def generate(self, sample: Dict) -> np.ndarray:
        """Run guided DDPM sampling and decode to (60, 5) trajectory."""
        # Starting state from last ego history frame
        sample["start_pos"]     = sample["ego_history"][-1, :2]
        sample["start_heading"] = float(sample["ego_history"][-1, 2])
        sample["start_vx"]      = float(sample["ego_history"][-1, 3])
        sample["start_vy"]      = float(sample["ego_history"][-1, 4])

        condition = self._get_condition(sample)

        ego_increments = guided_generate(
            model          = self.diffusion_model,
            condition      = condition,
            sample         = sample,
            guidance_scale = self.guidance_scale,
            guide_every    = self.guide_every,
            device         = self.device,
        )  # (120,) numpy: ego (dx, dy) increments

        traj_data = recover_trajectory_from_increment(
            ego_increments,
            sample["start_pos"],
            sample["start_heading"],
            sample["start_vx"],
            sample["start_vy"],
        )

        return np.stack([
            traj_data.positions[:, 0],
            traj_data.positions[:, 1],
            traj_data.headings,
            traj_data.velocities[:, 0],
            traj_data.velocities[:, 1],
        ], axis=1).astype(np.float32)  # (60, 5)
