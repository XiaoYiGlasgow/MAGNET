"""
Matplotlib Quick Validation Visualizer

Fast trajectory quality check without external dependencies.

Features:
- Trajectory comparison (GT vs Generated)
- Velocity/acceleration curves
- Danger score visualization
- Physical validity check
- Summary statistics

Usage:
    from visualization import MatplotlibVisualizer

    viz = MatplotlibVisualizer(model_path="best_model.pt")
    viz.evaluate_samples(num_samples=10, output_dir="eval_results")
"""

import os
import sys

# Avoid importing the local wheel-cache directory `nuplan-devkit/torch` as PyTorch.
sys.path = [
    p for p in sys.path
    if os.path.basename(os.path.normpath(p)) != "nuplan-devkit"
]

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)
_root_dir = os.path.dirname(os.path.dirname(_parent_dir))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import config

from visualization.trajectory_converter import (
    recover_trajectory_from_increment,
    kinematic_decode_np,
    compute_trajectory_metrics,
    check_trajectory_validity,
    check_kinematic_validity,
    future_gt_to_kinematic_np,
    agent_future_to_kinematic_np,
    gt_to_cumulative_kinematic_np,
    kinematic_cumulative_decode_np,
    gt_to_xy_displacement_np,
    xy_displacement_decode_np,
    TrajectoryData
)
from visualization.standard_metrics import (
    compute_standard_metrics,
    compute_collision,
    compute_min_distance,
    COLLISION_DIST,
)
from visualization import register_visualizer


@dataclass
class EvalSample:
    """Container for evaluation sample."""
    sample_idx: int
    scenario_name: str
    ego_history: np.ndarray
    agent_history: np.ndarray
    gt_trajectory: np.ndarray
    gen_trajectory: np.ndarray
    gt_danger: float
    gen_danger: float
    svo_angle: float
    validity: Dict
    agent_future: Optional[np.ndarray] = None
    gen_agent_future: Optional[np.ndarray] = None


@register_visualizer("matplotlib")
class MatplotlibVisualizer:
    """Quick matplotlib-based trajectory validator."""

    def __init__(
        self,
        model_path: str,
        data_dir: str = config.DANGER_PAIR_SCENARIOS_DIR,
        processed_data_dir: str = config.PROCESSED_DATA_DIR,
        device: str = "cuda",
        min_gt_danger: float = 0.5,
    ):
        """
        Initialize visualizer with trained model.

        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing danger_pair pkl files
            device: Device to run on
            min_gt_danger: Skip samples whose GT outcome danger is below this
                threshold. Under tent score (v4.0+), typical range is 0-8;
                0.5 filters near-zero trivial samples.
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        if not self.model_path.is_absolute():
            self.model_path = (Path(_parent_dir) / self.model_path).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (Path(_parent_dir) / self.data_dir).resolve()
        if not self.processed_data_dir.is_absolute():
            self.processed_data_dir = (Path(_parent_dir) / self.processed_data_dir).resolve()
        self.device = device
        self.min_gt_danger = min_gt_danger

        # Load model
        self._load_model()

        # Import danger score calculator
        sys.path.insert(0, os.path.join(_parent_dir, 'utils'))
        from danger_score import compute_danger_score, DangerScoreConfig, compute_pair_svo
        self.compute_danger = compute_danger_score
        self.compute_pair_svo = compute_pair_svo
        self.danger_config = DangerScoreConfig()

    def _load_model(self):
        """Load trained diffusion model."""
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Import model classes - need to add paths
        sys.path.insert(0, _parent_dir)
        sys.path.insert(0, os.path.join(_parent_dir, 'models'))
        sys.path.insert(0, os.path.join(_parent_dir, 'utils'))
        transformer_models_dir = os.path.abspath(os.path.join(_parent_dir, '..', 'models'))
        if transformer_models_dir not in sys.path:
            sys.path.insert(0, transformer_models_dir)
        from scene_condition_encoder import (
            SceneConditionEncoder, ConditionEncoderConfig,
            SimpleSceneEncoder, AgentPredictionHead,
        )
        from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
        from dataset import GraphData
        from transformer_predictor import TransformerPredictor

        # Recreate models
        self.scene_encoder = None
        self.scene_encoder_mode = None
        self.graph_data_cls = GraphData
        self.agent_pred_head = None  # loaded only if checkpoint contains the key (v4.0+)

        self.condition_encoder = SceneConditionEncoder(ConditionEncoderConfig()).to(self.device)
        self.diffusion_model = DangerSceneDiffusion(DiffusionConfig()).to(self.device)
        self.agent_pred_head = None  # loaded only if checkpoint contains the key (v4.0+)

        # Load weights - check for different checkpoint key names
        if 'scene_encoder' in checkpoint:
            self.scene_encoder = SimpleSceneEncoder(
                ego_history_dim=5,
                agent_history_dim=5,
                output_dim=384,
                history_length=40
            ).to(self.device)
            self.scene_encoder.load_state_dict(checkpoint['scene_encoder'])
            self.scene_encoder_mode = "simple"
        elif 'gnn_trans_encoder' in checkpoint:
            self.scene_encoder = TransformerPredictor().to(self.device)
            self.scene_encoder.load_state_dict(checkpoint['gnn_trans_encoder'])
            self.scene_encoder_mode = "gnn_trans"
        else:
            raise KeyError(f"No scene encoder found in checkpoint. Available keys: {list(checkpoint.keys())}")

        self.condition_encoder.load_state_dict(checkpoint['condition_encoder'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])

        # Load AgentPredictionHead if available (v4.0+)
        if 'agent_prediction_head' in checkpoint:
            self.agent_pred_head = AgentPredictionHead().to(self.device)
            self.agent_pred_head.load_state_dict(checkpoint['agent_prediction_head'])
            self.agent_pred_head.eval()
            print(f"[MatplotlibVisualizer] AgentPredictionHead loaded (pred-anchor SDEdit, v4.0+)")
        else:
            print(f"[MatplotlibVisualizer] No agent_prediction_head in checkpoint — SDEdit uses GT anchor")

        self.scene_encoder.eval()
        self.condition_encoder.eval()
        self.diffusion_model.eval()

        print(f"[MatplotlibVisualizer] Model loaded from {self.model_path}")

    def generate_trajectory(
        self,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor,
        svo_angle: float,
        inherent_danger: float,
        danger_target: float,
        graph_data=None
    ) -> np.ndarray:
        """
        Generate trajectory from diffusion model.

        Args:
            ego_history: (40, 5) ego history
            agent_history: (40, 5) agent history
            svo_angle: SVO angle in degrees
            inherent_danger: Inherent danger score
            danger_target: Target danger score

        Returns:
            Generated increments (120,)
        """
        with torch.no_grad():
            # Encode scene
            if self.scene_encoder_mode == "simple":
                scene_feat = self.scene_encoder(ego_history.unsqueeze(0), agent_history.unsqueeze(0))
            elif self.scene_encoder_mode == "gnn_trans":
                if graph_data is None:
                    raise ValueError("graph_data is required for gnn_trans checkpoints")
                scene_feat = self.scene_encoder.get_encoder_features(graph_data)
            else:
                raise RuntimeError(f"Unsupported scene encoder mode: {self.scene_encoder_mode}")

            # Encode condition
            svo_tensor = torch.tensor([[svo_angle]], dtype=torch.float32, device=self.device)
            inherent_tensor = torch.tensor([[inherent_danger]], dtype=torch.float32, device=self.device)
            target_tensor = torch.tensor([[danger_target]], dtype=torch.float32, device=self.device)
            agent_state_tensor = agent_history[-1, 2:5].unsqueeze(0).to(self.device)  # [1,3]

            condition = self.condition_encoder(scene_feat, svo_tensor, inherent_tensor, target_tensor,
                                               agent_state=agent_state_tensor)

            # Generate trajectory
            generated = self.diffusion_model.generate(condition)

        return generated.cpu().numpy().squeeze()

    def generate_trajectory_sdedit(
        self,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor,
        svo_angle: float,
        inherent_danger: float,
        danger_target: float,
        sample_data: dict,
        k: int = 40,
        graph_data=None,
    ) -> np.ndarray:
        """Generate trajectory using SDEdit (partial noise anchored to GT).

        Converts GT future trajectory to (Δx, Δy) displacement space, adds partial
        forward noise at level k, then denoises back to 0. Prevents degenerate
        outputs by anchoring to GT road-following behaviour.

        Args:
            ego_history: (40, 5) ego history tensor on device
            agent_history: (40, 5) agent history tensor on device
            svo_angle: SVO angle in degrees
            inherent_danger: inherent danger score from GT evaluation
            danger_target: target danger score (usually gt_danger + 2.0)
            sample_data: dict from load_sample(), must contain:
                ego_future_gt, key_agent_future, start_pos, agent_history (numpy)
            k: SDEdit noise level (1..99); lower → closer to GT
            graph_data: required for gnn_trans checkpoints

        Returns:
            (240,) numpy array of generated (Δx, Δy) increments
        """
        with torch.no_grad():
            # Encode scene
            if self.scene_encoder_mode == "simple":
                scene_feat = self.scene_encoder(
                    ego_history.unsqueeze(0), agent_history.unsqueeze(0)
                )
            elif self.scene_encoder_mode == "gnn_trans":
                if graph_data is None:
                    raise ValueError("graph_data is required for gnn_trans checkpoints")
                scene_feat = self.scene_encoder.get_encoder_features(graph_data)
            else:
                raise RuntimeError(f"Unsupported scene encoder mode: {self.scene_encoder_mode}")

            svo_tensor = torch.tensor([[svo_angle]], dtype=torch.float32, device=self.device)
            inherent_tensor = torch.tensor([[inherent_danger]], dtype=torch.float32, device=self.device)
            target_tensor = torch.tensor([[danger_target]], dtype=torch.float32, device=self.device)
            agent_state_tensor = agent_history[-1, 2:5].unsqueeze(0).to(self.device)  # [1,3]
            condition = self.condition_encoder(
                scene_feat, svo_tensor, inherent_tensor, target_tensor,
                agent_state=agent_state_tensor
            )

            # Build 240-dim SDEdit anchor in (Δx,Δy) space.
            # v4.0+: GNN+Trans ego pred + AgentPredictionHead → no GT needed at inference.
            # Fallback: use GT trajectory as anchor (pre-v4.0 checkpoints).
            if self.agent_pred_head is not None and self.scene_encoder_mode == "gnn_trans":
                # ego_pred from pre-trained GNN+Trans output_head
                fused    = self.scene_encoder.feature_fusion(scene_feat)      # [1, 128]
                ego_pred = self.scene_encoder.output_head(fused)              # [1, 120]
                # agent_pred from AgentPredictionHead
                agent_last_state = agent_history[-1:, :].to(self.device)     # [1, 5]
                agent_pred = self.agent_pred_head(scene_feat, agent_last_state)  # [1, 120]
                x_anchor_t = torch.cat([ego_pred, agent_pred], dim=-1)       # [1, 240]
            else:
                ego_xy = gt_to_xy_displacement_np(
                    sample_data['ego_future_gt'], sample_data['start_pos']
                )  # (120,)
                agent_last_pos = sample_data['agent_history'][-1, :2]
                agent_xy = gt_to_xy_displacement_np(
                    sample_data['key_agent_future'], agent_last_pos
                )  # (120,)
                x_anchor_t = torch.tensor(
                    np.concatenate([ego_xy, agent_xy]),
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # [1, 240]

            # SDEdit in (Δx,Δy) space
            generated = self.diffusion_model.sdedit_generate_dx_dy(
                x_anchor_t, condition, k=k
            )

        return generated.cpu().numpy().squeeze()

    def load_sample(self, pkl_file: Path) -> Dict:
        """Load a sample from pkl file."""
        import pandas as pd
        data = pd.read_pickle(pkl_file)

        # Extract required fields
        ego_history = data['ego_history']  # (40, 5)
        agent_history = data['key_agent_history']  # (40, 5)
        ego_future_gt = data['ego_future_gt']  # (60, 5)

        if ego_future_gt.ndim == 3:
            ego_future_gt = ego_future_gt.squeeze(0)

        agent_future = data.get('key_agent_future', None)
        agent_future = self._normalize_agent_future(agent_future, agent_history)
        graph_data = None
        if self.scene_encoder_mode == "gnn_trans":
            # Use the same graph structure as training: Node0=ego, Node1=key_agent,
            # Nodes2+=boundary agents.  This gives the GNN encoder the agent's
            # spatial position (x, y) as it was trained to receive.
            # _load_graph_data() used lane polylines instead — a train/inference
            # mismatch that prevented the model from locating the agent.
            from core.dataset.danger_pair_dataset import _build_graph_data as _build_train_graph
            boundary_agents = data.get('boundary_agents', [])
            graph_data = _build_train_graph(ego_history, agent_history, boundary_agents)

        # Get starting state
        start_pos = ego_history[-1, :2]
        start_heading = ego_history[-1, 2]
        start_vx = ego_history[-1, 3]
        start_vy = ego_history[-1, 4]

        # Fix frozen velocity fields: recompute vx,vy from position differences
        # PKL stores vx,vy frozen at last history value — corrupts TTC/danger scores
        ego_future_gt  = self._fix_velocity_fields(ego_future_gt[:60],  start_pos)
        agent_last_pos = agent_history[-1, :2]
        agent_future   = self._fix_velocity_fields(agent_future, agent_last_pos)

        # Compute SVO on-the-fly from trajectories — pkl has no svo_angle field.
        # Training uses the same compute_pair_svo call (run_full_training.py:309).
        svo_info = self.compute_pair_svo(
            ego_history, ego_future_gt, agent_history, agent_future, self.danger_config
        )

        return {
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future_gt': ego_future_gt[:60],
            'key_agent_future': agent_future,
            'graph_data': graph_data,
            'start_pos': start_pos,
            'start_heading': start_heading,
            'start_vx': start_vx,
            'start_vy': start_vy,
            'scenario_name': data.get('scenario_name', 'unknown'),
            'svo_angle': svo_info['ego_svo_angle_deg'],
        }

    def _load_graph_data(self, pkl_file: Path):
        """Load the processed graph input required by gnn_trans checkpoints."""
        import pandas as pd

        stem = pkl_file.stem
        if 'features_' in stem:
            scenario_name = stem.replace('danger_pair_features_', '').split('_agent')[0]
        else:
            scenario_name = stem.replace('danger_pair_', '').split('_agent')[0]
        processed_file = self.processed_data_dir / f"features_{scenario_name}.pkl"
        if not processed_file.exists():
            raise FileNotFoundError(f"Processed graph file not found: {processed_file}")

        processed_data = pd.read_pickle(processed_file)
        polyline_features = processed_data.iloc[0]['POLYLINE_FEATURES']
        traj_id_to_indices = processed_data.iloc[0]['TRAJ_ID_TO_INDICES']
        return self._build_graph_data(polyline_features, traj_id_to_indices)

    def _build_graph_data(self, polyline_features: np.ndarray, traj_id_to_indices: dict):
        """Build GraphData compatible with TransformerPredictor.get_encoder_features."""
        x = torch.tensor(polyline_features[:, :9], dtype=torch.float32)
        num_nodes = x.size(0)

        if num_nodes > 1:
            positions = polyline_features[:, :2]
            edges = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if np.linalg.norm(positions[i] - positions[j]) <= 30.0:
                        edges.append([i, j])
                        edges.append([j, i])
            if edges:
                gnn_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        if 0 in traj_id_to_indices:
            start, end = traj_id_to_indices[0]
            ego_polyline = polyline_features[start:end + 1]
            ego_history_positions = ego_polyline[:, :2]
            if len(ego_history_positions) >= 40:
                ego_history_tensor = torch.tensor(ego_history_positions[:40], dtype=torch.float32)
            else:
                padding = np.zeros((40 - len(ego_history_positions), 2), dtype=ego_history_positions.dtype)
                ego_history_tensor = torch.tensor(
                    np.concatenate([ego_history_positions, padding], axis=0),
                    dtype=torch.float32
                )
        else:
            ego_history_tensor = torch.zeros(40, 2, dtype=torch.float32)

        return self.graph_data_cls(
            x=x,
            gnn_x=x.clone(),
            gnn_edge_index=gnn_edge_index,
            ego_history=ego_history_tensor
        )

    @staticmethod
    def _fix_velocity_fields(traj: np.ndarray, last_pos: np.ndarray,
                              dt: float = 0.1) -> np.ndarray:
        """Convert trajectory to (T, 5) [x, y, heading, vx, vy] with correct velocity.

        New format (T, 2): ego-centric increments (Δx, Δy).
          → convert to absolute positions via cumsum + start_pos offset,
            compute heading from arctan2, velocity from increment/dt.
        Old format (T, 5): absolute ego-centric positions with frozen vx/vy.
          → keep positions/heading, recompute vx/vy from position differences.
        """
        T = len(traj)
        if traj.shape[1] < 5:
            from core.dataset.trajectory_utils import (
                increments_to_positions, compute_velocity_from_increments,
            )
            abs_pos  = increments_to_positions(traj)           # cumsum → (T, 2)
            abs_xy   = last_pos[:2] + abs_pos                  # shift to ego-centric absolute
            dx, dy   = traj[:, 0], traj[:, 1]
            headings = np.arctan2(dy, dx)
            vels     = compute_velocity_from_increments(traj, dt)  # (T, 2)
            return np.stack(
                [abs_xy[:, 0], abs_xy[:, 1], headings, vels[:, 0], vels[:, 1]],
                axis=1,
            ).astype(np.float32)
        else:
            fixed     = traj.copy()
            positions = fixed[:, :2]
            if T > 1:
                fixed[0, 3] = (positions[1, 0] - positions[0, 0]) / dt
                fixed[0, 4] = (positions[1, 1] - positions[0, 1]) / dt
            for t in range(1, T):
                fixed[t, 3] = (positions[t, 0] - positions[t - 1, 0]) / dt
                fixed[t, 4] = (positions[t, 1] - positions[t - 1, 1]) / dt
            return fixed

    @staticmethod
    def _normalize_agent_future(agent_future: Optional[np.ndarray], agent_history: np.ndarray) -> np.ndarray:
        """Convert agent future to a stable (60, 5) array for danger evaluation."""
        if agent_future is None:
            agent_future_full = np.zeros((60, 5), dtype=np.float32)
            agent_future_full[:, :2] = agent_history[-1, :2]
            agent_future_full[:, 2] = agent_history[-1, 2]
            agent_future_full[:, 3] = agent_history[-1, 3]
            agent_future_full[:, 4] = agent_history[-1, 4]
            return agent_future_full

        agent_future = np.asarray(agent_future)
        if agent_future.ndim == 3:
            agent_future = agent_future.squeeze(0)

        num_frames = min(len(agent_future), 60)
        agent_future_full = np.zeros((60, 5), dtype=agent_future.dtype)
        if num_frames > 0:
            width = min(agent_future.shape[1], 5)
            agent_future_full[:num_frames, :width] = agent_future[:num_frames, :width]

        if num_frames < 60:
            agent_future_full[num_frames:, :2] = agent_future_full[max(num_frames - 1, 0), :2]

        if agent_future.shape[1] < 3:
            agent_future_full[:, 2] = agent_history[-1, 2]
        if agent_future.shape[1] < 5:
            agent_future_full[:, 3] = agent_history[-1, 3]
            agent_future_full[:, 4] = agent_history[-1, 4]

        return agent_future_full

    def evaluate_samples(
        self,
        num_samples: int = 10,
        output_dir: str = config.DIFFUSION_VISUALIZATION_RESULTS_DIR,
        show_plots: bool = False,
        num_inference_samples: int = 1,
        use_sdedit: bool = False,
        sdedit_k: int = 50,
    ) -> Dict:
        """
        Evaluate multiple samples and generate visualization.

        Args:
            num_samples: Number of samples to evaluate
            output_dir: Directory to save results
            show_plots: Whether to display plots interactively
            num_inference_samples: Best-of-N — generate this many diffusion
                candidates per scene and keep the most dangerous one (default 1).

        Returns:
            Evaluation results dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Graph data is now built directly from pkl (ego/agent/boundary_agents),
        # so all pkls are eligible regardless of features file availability.
        import random as _random
        all_pkl_files = list(self.data_dir.glob('danger_pair_*.pkl'))
        pkl_files = _random.sample(all_pkl_files, min(num_samples, len(all_pkl_files)))
        print(f"[Eval] Sampling {len(pkl_files)} from {len(all_pkl_files)} pkls")
        print(f"[Eval] Evaluating {len(pkl_files)} samples...")

        if not pkl_files:
            raise FileNotFoundError(
                f"No files matching 'danger_pair_*.pkl' found in {self.data_dir}"
            )

        eval_samples: List[EvalSample] = []
        results_summary = {
            'total_samples': 0,
            'skipped_count': 0,
            'margin_met_count': 0,       # gen_danger > gt_danger+0.5, collision excluded
            'margin_met_raw_count': 0,   # same but collision included (for reference)
            'collision_count': 0,        # gen trajectory collides with agent (dist < 2m)
            'valid_count': 0,
            'nan_count': 0,
            'over40_count': 0,
            'over80_count': 0,
            'gt_dangers': [],
            'inherent_dangers': [],
            'gen_dangers': [],
            'gen_dangers_vs_gt_agent': [],  # danger(ego_gen, agent_GT) — second metric
            'min_distances': [],
            'metrics_gt': [],
            'metrics_gen': []
        }

        for idx, pkl_file in enumerate(pkl_files):
            print(f"\n[Sample {idx+1}/{len(pkl_files)}] {pkl_file.name}")

            # Load sample
            try:
                sample_data = self.load_sample(pkl_file)
            except FileNotFoundError as e:
                print(f"  [Skip] {e}")
                continue

            # Convert to tensors
            ego_history_t = torch.tensor(sample_data['ego_history'], dtype=torch.float32, device=self.device)
            agent_history_t = torch.tensor(sample_data['agent_history'], dtype=torch.float32, device=self.device)

            # Compute GT danger
            agent_future = sample_data['key_agent_future']

            gt_danger_result = self.compute_danger(
                sample_data['ego_history'],
                sample_data['ego_future_gt'],
                sample_data['agent_history'],
                agent_future,
                self.danger_config
            )
            gt_danger = gt_danger_result['outcome_danger']['total_score']
            inherent_danger = gt_danger_result['inherent_danger']['total_score']

            # Skip samples whose GT danger is too low — they are out-of-distribution
            # relative to training data (which had GT danger ~31). Conditioning on
            # near-zero danger produces unreliable generated trajectories.
            if gt_danger < self.min_gt_danger:
                print(f"  [Skip] gt_danger={gt_danger:.3f} < min_gt_danger={self.min_gt_danger:.1f}")
                results_summary['skipped_count'] += 1
                continue

            results_summary['total_samples'] += 1

            # Generate trajectory — Best-of-N: keep the most dangerous candidate
            danger_target = gt_danger + 2.0  # Target exceeds GT by 2
            _ego_v0 = np.sqrt(sample_data['start_vx'] ** 2 + sample_data['start_vy'] ** 2)
            _graph = sample_data['graph_data'].to(self.device) if sample_data['graph_data'] is not None else None
            best_gen_inc, best_score = None, -float('inf')
            for _trial in range(num_inference_samples):
                if use_sdedit:
                    _cand = self.generate_trajectory_sdedit(
                        ego_history_t, agent_history_t,
                        sample_data['svo_angle'], inherent_danger, danger_target,
                        sample_data=sample_data,
                        k=sdedit_k,
                        graph_data=_graph,
                    )
                else:
                    _cand = self.generate_trajectory(
                        ego_history_t, agent_history_t,
                        sample_data['svo_angle'], inherent_danger, danger_target,
                        graph_data=_graph
                    )
                _jm = len(_cand) == 240
                _et = xy_displacement_decode_np(
                    _cand[:120] if _jm else _cand,
                    sample_data['start_pos']
                )
                if _jm:
                    _ag = xy_displacement_decode_np(
                        _cand[120:], sample_data['agent_history'][-1, :2]
                    ).full_trajectory
                else:
                    _ag = agent_future
                _score = self.compute_danger(
                    sample_data['ego_history'], _et.full_trajectory,
                    sample_data['agent_history'], _ag, self.danger_config
                )['outcome_danger']['total_score']
                if _score > best_score:
                    best_score, best_gen_inc = _score, _cand
            gen_increments = best_gen_inc

            # Detect joint generation mode (240-dim) vs ego-only (120-dim)
            joint_mode = len(gen_increments) == 240
            ego_inc = gen_increments[:120] if joint_mode else gen_increments

            # Recover ego trajectory from (Δx,Δy) displacement decode
            gen_traj = xy_displacement_decode_np(
                ego_inc,
                sample_data['start_pos'],
            )

            # Determine agent trajectory used for danger evaluation
            if joint_mode:
                agent_inc = gen_increments[120:]
                agent_traj_data = xy_displacement_decode_np(agent_inc, sample_data['agent_history'][-1, :2])
                danger_agent = agent_traj_data.full_trajectory
            else:
                danger_agent = agent_future

            # Compute generated danger — Joint: danger(ego_gen, agent_gen)
            gen_danger_result = self.compute_danger(
                sample_data['ego_history'],
                gen_traj.full_trajectory,
                sample_data['agent_history'],
                danger_agent,
                self.danger_config
            )
            gen_danger = gen_danger_result['outcome_danger']['total_score']

            # Second metric: danger(ego_gen, agent_GT) — fixed agent, ego-only effect
            gen_danger_vs_gt_result = self.compute_danger(
                sample_data['ego_history'],
                gen_traj.full_trajectory,
                sample_data['agent_history'],
                agent_future,
                self.danger_config
            )
            gen_danger_vs_gt = gen_danger_vs_gt_result['outcome_danger']['total_score']

            # Check validity
            validity = check_kinematic_validity(gen_traj)

            # Compute metrics — GT uses the same cumsum+gradient pipeline as the
            # generated trajectory because ego_future_gt[:, :2] stores per-frame
            # increments (not absolute positions) and ego_future_gt[:, 3:5] stores
            # a constant copied velocity rather than frame-wise derivatives.
            gt_traj_data = recover_trajectory_from_increment(
                sample_data['ego_future_gt'][:, :2].flatten(),
                sample_data['start_pos'],
                sample_data['start_heading'],
                sample_data['start_vx'],
                sample_data['start_vy'],
            )
            metrics_gt = compute_trajectory_metrics(gt_traj_data)
            metrics_gen = compute_trajectory_metrics(gen_traj)

            # Create eval sample
            eval_sample = EvalSample(
                sample_idx=idx,
                scenario_name=sample_data['scenario_name'],
                ego_history=sample_data['ego_history'],
                agent_history=sample_data['agent_history'],
                gt_trajectory=gt_traj_data.full_trajectory,
                gen_trajectory=gen_traj.full_trajectory,
                gt_danger=gt_danger,
                gen_danger=gen_danger,
                svo_angle=sample_data['svo_angle'],
                validity=validity,
                agent_future=sample_data.get('key_agent_future'),
            )
            eval_samples.append(eval_sample)

            # Per-sample collision check (literature-aligned: min_distance < vehicle radius sum)
            _agent_for_collision = danger_agent if danger_agent is not None else agent_future
            _has_collision = compute_collision(gen_traj.full_trajectory, _agent_for_collision)
            _min_dist = compute_min_distance(gen_traj.full_trajectory, _agent_for_collision)

            # Update summary
            results_summary['gt_dangers'].append(gt_danger)
            results_summary['inherent_dangers'].append(inherent_danger)
            results_summary['gen_dangers'].append(gen_danger)
            results_summary['gen_dangers_vs_gt_agent'].append(gen_danger_vs_gt)
            results_summary['min_distances'].append(_min_dist)
            results_summary['metrics_gt'].append(metrics_gt)
            results_summary['metrics_gen'].append(metrics_gen)

            _beats_gt = gen_danger > gt_danger + 0.5
            if _beats_gt:
                results_summary['margin_met_raw_count'] += 1
            # Primary margin_met excludes collision trajectories (physically invalid,
            # not meaningful for AV testing — consistent with KING / AdvSim / CTG)
            if _beats_gt and not _has_collision:
                results_summary['margin_met_count'] += 1
            if _has_collision:
                results_summary['collision_count'] += 1
            if validity['overall_valid']:
                results_summary['valid_count'] += 1
            if validity.get('has_nan', False):
                results_summary['nan_count'] += 1
            if validity.get('has_speed_over_40', False):
                results_summary['over40_count'] += 1
            if validity.get('has_speed_over_80', False):
                results_summary['over80_count'] += 1

            # Plot individual sample
            self._plot_sample(eval_sample, output_path / f"sample_{idx}.png", show=show_plots)

        # Summary statistics
        n = results_summary['total_samples']
        if n == 0:
            print(f"\n[Warning] All {results_summary['skipped_count']} samples were skipped "
                  f"(gt_danger < {self.min_gt_danger}). Lower --min_gt_danger to evaluate more samples.")
            return results_summary
        results_summary['margin_met_rate'] = results_summary['margin_met_count'] / n
        results_summary['margin_met_raw_rate'] = results_summary['margin_met_raw_count'] / n
        results_summary['collision_rate'] = results_summary['collision_count'] / n
        results_summary['valid_rate'] = results_summary['valid_count'] / n
        results_summary['nan_rate'] = results_summary['nan_count'] / n
        results_summary['over40_rate'] = results_summary['over40_count'] / n
        results_summary['over80_rate'] = results_summary['over80_count'] / n
        results_summary['mean_gt_danger'] = float(np.mean(results_summary['gt_dangers']))
        results_summary['mean_inherent_danger'] = float(np.mean(results_summary['inherent_dangers']))
        results_summary['mean_gen_danger'] = float(np.mean(results_summary['gen_dangers']))
        results_summary['mean_gen_danger_vs_gt_agent'] = float(np.mean(results_summary['gen_dangers_vs_gt_agent']))
        results_summary['mean_min_distance'] = float(np.mean(results_summary['min_distances']))
        results_summary['danger_improvement'] = results_summary['mean_gen_danger'] - results_summary['mean_gt_danger']
        results_summary['danger_improvement_vs_gt_agent'] = results_summary['mean_gen_danger_vs_gt_agent'] - results_summary['mean_gt_danger']

        # Standard metrics (literature-aligned)
        std = compute_standard_metrics(
            eval_samples,
            results_summary['metrics_gen'],
            results_summary['metrics_gt'],
        )
        results_summary['standard_metrics'] = std

        # Plot summary
        self._plot_summary(results_summary, eval_samples, output_path / "summary.png", show=show_plots)

        # Save JSON report
        report_path = output_path / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n[Eval done] {n} samples evaluated, {results_summary['skipped_count']} skipped. "
              f"Report: {output_path}")

        return results_summary

    def _plot_sample(self, sample: EvalSample, save_path: Path, show: bool = False):
        """Plot single sample comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Trajectory comparison
        ax1 = axes[0, 0]
        ax1.plot(sample.ego_history[:, 0], sample.ego_history[:, 1], 'b-', label='Ego History', alpha=0.5)
        ax1.plot(sample.gt_trajectory[:, 0], sample.gt_trajectory[:, 1], 'g-', label='GT Future', linewidth=2)
        ax1.plot(sample.gen_trajectory[:, 0], sample.gen_trajectory[:, 1], 'r--', label='Generated', linewidth=2)
        ax1.plot(sample.agent_history[:, 0], sample.agent_history[:, 1], 'orange', label='Agent', alpha=0.5)
        ax1.scatter(sample.ego_history[-1, 0], sample.ego_history[-1, 1], c='blue', s=100, marker='o', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'Sample {sample.sample_idx}: Trajectory Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')

        # 2. Speed comparison
        ax2 = axes[0, 1]
        gt_speed = np.sqrt(sample.gt_trajectory[:, 3]**2 + sample.gt_trajectory[:, 4]**2)
        gen_speed = np.sqrt(sample.gen_trajectory[:, 3]**2 + sample.gen_trajectory[:, 4]**2)
        time = np.arange(60) * 0.1
        ax2.plot(time, gt_speed, 'g-', label='GT Speed', linewidth=2)
        ax2.plot(time, gen_speed, 'r--', label='Generated Speed', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed Profile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Danger scores
        ax3 = axes[1, 0]
        ax3.bar(['GT', 'Generated'], [sample.gt_danger, sample.gen_danger],
                color=['green', 'red'], alpha=0.7)
        ax3.axhline(y=sample.gt_danger + 0.5, color='black', linestyle='--', label='Target Margin')
        ax3.set_ylabel('Danger Score')
        ax3.set_title(f'Danger Comparison\nMargin Met: {sample.gen_danger > sample.gt_danger + 0.5}')
        ax3.legend()

        # 4. Validity info
        ax4 = axes[1, 1]
        metrics = sample.validity['metrics']
        text_lines = [
            f"Scenario: {sample.scenario_name}",
            f"SVO Angle: {sample.svo_angle:.1f}°",
            "",
            "Generated Trajectory Metrics:",
            f"  Max Speed: {metrics['max_speed']:.2f} m/s",
            f"  Max Accel: {metrics['max_accel']:.2f} m/s²",
            f"  Max Jerk: {metrics['max_jerk']:.2f} m/s³",
            "",
            "Validity Checks:",
            f"  Speed: {'✓' if sample.validity['speed_valid'] else '✗'}",
            f"  Accel: {'✓' if sample.validity['accel_valid'] else '✗'}",
            f"  Jerk: {'✓' if sample.validity['jerk_valid'] else '✗'}",
            f"  Overall: {'✓' if sample.validity['overall_valid'] else '✗'}"
        ]
        ax4.text(0.1, 0.5, '\n'.join(text_lines), transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
        ax4.axis('off')
        ax4.set_title('Sample Info')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()

    def _plot_summary(self, results: Dict, samples: List[EvalSample], save_path: Path, show: bool = False):
        """Plot summary statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Danger score distribution
        ax1 = axes[0, 0]
        ax1.scatter(results['gt_dangers'], results['gen_dangers'], alpha=0.6)
        max_danger = max(max(results['gt_dangers']), max(results['gen_dangers']))
        ax1.plot([0, max_danger], [0, max_danger], 'k--', label='Equal', alpha=0.5)
        ax1.plot([0, max_danger], [0.5, max_danger + 0.5], 'g--', label='Margin Met', alpha=0.5)
        ax1.set_xlabel('GT Danger')
        ax1.set_ylabel('Generated Danger')
        ax1.set_title(f'Danger Comparison\nMargin Met: {results["margin_met_rate"]:.1%}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Danger improvement histogram
        ax2 = axes[0, 1]
        improvements = [g - t for g, t in zip(results['gen_dangers'], results['gt_dangers'])]
        ax2.hist(improvements, bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0.5, color='green', linestyle='--', label='Target Margin')
        ax2.axvline(x=np.mean(improvements), color='red', linestyle='-', label='Mean Improvement')
        ax2.set_xlabel('Danger Improvement (Gen - GT)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Danger Improvement Distribution\nMean: {np.mean(improvements):+.2f}')
        ax2.legend()

        # 3. Speed comparison
        ax3 = axes[1, 0]
        gt_max_speeds = [m['max_speed'] for m in results['metrics_gt']]
        gen_max_speeds = [m['max_speed'] for m in results['metrics_gen']]
        ax3.boxplot([gt_max_speeds, gen_max_speeds], labels=['GT', 'Generated'])
        ax3.axhline(y=80, color='red', linestyle='--', label='Speed Limit (80 m/s)')
        ax3.set_ylabel('Max Speed (m/s)')
        ax3.set_title('Speed Distribution')
        ax3.legend()

        # 4. Validity summary
        ax4 = axes[1, 1]
        summary_text = [
            "=== Evaluation Summary ===",
            f"Total Samples: {results['total_samples']}",
            f"Margin Met: {results['margin_met_count']} ({results['margin_met_rate']:.1%})",
            f"Valid Trajectories: {results['valid_count']} ({results['valid_rate']:.1%})",
            "",
            f"Mean GT Danger: {results['mean_gt_danger']:.2f}",
            f"Mean Gen Danger: {results['mean_gen_danger']:.2f}",
            f"Danger Improvement: {results['danger_improvement']:+.2f}",
        ]
        ax4.text(0.1, 0.5, '\n'.join(summary_text), transform=ax4.transAxes,
                fontsize=12, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Summary Statistics')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close()


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Matplotlib Quick Validation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default=config.DANGER_PAIR_SCENARIOS_DIR)
    parser.add_argument('--processed_data_dir', type=str, default=config.PROCESSED_DATA_DIR)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default=config.DIFFUSION_VISUALIZATION_RESULTS_DIR)
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    viz = MatplotlibVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        device=args.device,
    )
    results = viz.evaluate_samples(args.num_samples, args.output_dir, args.show)
