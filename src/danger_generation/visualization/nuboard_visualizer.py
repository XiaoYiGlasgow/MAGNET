"""
NuBoard Detailed Display Visualizer

Convert Diffusion trajectories to NuPlan format and launch nuBoard.

Features:
- Map background visualization
- Time slider playback
- Agent interactions
- GT vs Generated comparison

Note: NuBoard requires NuPlan simulation data format.
This module creates compatible data from Diffusion outputs.

Usage:
    from visualization import NuBoardVisualizer

    viz = NuBoardVisualizer(model_path="experiments/danger_generation/training_runs/full_training/best_model.pt")
    viz.prepare_nuboard_data(num_samples=5, output_dir="experiments/danger_generation/visualizations/nuboard_data")
    viz.launch_nuboard()
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
import pickle
import lzma
import msgpack
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import subprocess
import json
import re

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
    TrajectoryData
)
from visualization import register_visualizer


@dataclass
class NuBoardScenarioData:
    """Container for NuBoard-compatible scenario data."""
    scenario_id: str
    log_name: str
    timestamp: int

    # Ego trajectory
    ego_gt_trajectory: np.ndarray      # (T, 5)
    ego_gen_trajectory: np.ndarray     # (T, 5)
    ego_history: np.ndarray            # (T_hist, 5)

    # Agent data
    agent_history: np.ndarray          # (T_hist, 5)
    agent_future: np.ndarray           # (T, 5)

    # Danger info
    gt_danger: float
    gen_danger: float
    svo_angle: float

    # Validity
    is_valid: bool


@dataclass
class NuBoardBundlePaths:
    """Exported real NuBoard bundle paths."""
    output_dir: Path
    simulations_dir: Path
    metrics_dir: Path
    aggregator_dir: Path
    nuboard_file: Path


class DiffusionMockScenario:
    """
    Lightweight, pickle-safe scenario used to display diffusion results in NuBoard.

    It wraps NuPlan's MockAbstractScenario but replaces the expert ego trajectory and
    tracked objects with the trajectories reconstructed from our diffusion samples.
    """

    def __init__(
        self,
        scenario_name: str,
        log_name: str,
        scenario_type: str,
        expert_states,
        tracked_detections,
        time_step: float = 0.1,
    ):
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario

        if len(expert_states) < 2:
            raise ValueError("expert_states must contain at least 2 states for NuBoard playback.")
        if len(expert_states) != len(tracked_detections):
            raise ValueError("expert_states and tracked_detections must have the same length.")

        initial_state = expert_states[0].rear_axle
        initial_velocity = expert_states[0].dynamic_car_state.rear_axle_velocity_2d
        mission_goal = expert_states[-1].rear_axle
        number_of_future_iterations = len(expert_states) - 1

        self._base = MockAbstractScenario(
            initial_time_us=TimePoint(0),
            time_step=time_step,
            number_of_future_iterations=number_of_future_iterations,
            number_of_past_iterations=0,
            initial_velocity=StateVector2D(x=initial_velocity.x, y=initial_velocity.y),
            fixed_acceleration=StateVector2D(x=0.0, y=0.0),
            number_of_detections=1,
            initial_ego_state=StateSE2(
                x=initial_state.x,
                y=initial_state.y,
                heading=initial_state.heading,
            ),
            mission_goal=StateSE2(
                x=mission_goal.x,
                y=mission_goal.y,
                heading=mission_goal.heading,
            ),
        )

        self._base._ego_states = expert_states
        self._base._tracked_objects = tracked_detections
        self._base._total_iterations = len(expert_states)
        self._base._number_of_future_iterations = len(expert_states) - 1
        self._base._current_iteration = 0

        self._scenario_name = scenario_name
        self._log_name = log_name
        self._scenario_type = scenario_type
        self._token = f"diffusion_{log_name}_{scenario_name}"

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __getstate__(self):
        return {
            "_base": self._base,
            "_scenario_name": self._scenario_name,
            "_log_name": self._log_name,
            "_scenario_type": self._scenario_type,
            "_token": self._token,
        }

    def __setstate__(self, state):
        self._base = state["_base"]
        self._scenario_name = state["_scenario_name"]
        self._log_name = state["_log_name"]
        self._scenario_type = state["_scenario_type"]
        self._token = state["_token"]

    @property
    def token(self) -> str:
        return self._token

    @property
    def log_name(self) -> str:
        return self._log_name

    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def scenario_type(self) -> str:
        return self._scenario_type


@register_visualizer("nuboard")
class NuBoardVisualizer:
    """NuBoard-based trajectory visualizer."""

    def __init__(
        self,
        model_path: str,
        data_dir: str = config.DANGER_PAIR_SCENARIOS_DIR,
        processed_data_dir: str = config.PROCESSED_DATA_DIR,
        nuplan_root: str = "/mnt/local/Xiao_Yi/nuplan-devkit",
        device: str = "cuda"
    ):
        """
        Initialize visualizer.

        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing danger_pair pkl files
            nuplan_root: Root directory of NuPlan devkit
            device: Device to run on
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.nuplan_root = Path(nuplan_root)
        if not self.model_path.is_absolute():
            self.model_path = (Path(_parent_dir) / self.model_path).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (Path(_parent_dir) / self.data_dir).resolve()
        if not self.processed_data_dir.is_absolute():
            self.processed_data_dir = (Path(_parent_dir) / self.processed_data_dir).resolve()
        if not self.nuplan_root.is_absolute():
            self.nuplan_root = (Path(_parent_dir) / self.nuplan_root).resolve()
        self.device = device

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

        # Import model classes
        sys.path.insert(0, os.path.join(_parent_dir, 'models'))
        sys.path.insert(0, os.path.join(_parent_dir, 'utils'))
        transformer_models_dir = os.path.abspath(os.path.join(_parent_dir, '..', 'models'))
        if transformer_models_dir not in sys.path:
            sys.path.insert(0, transformer_models_dir)
        from scene_condition_encoder import SceneConditionEncoder, ConditionEncoderConfig, SimpleSceneEncoder
        from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
        from dataset import GraphData
        from transformer_predictor import TransformerPredictor

        # Recreate models
        self.scene_encoder = None
        self.scene_encoder_mode = None
        self.graph_data_cls = GraphData

        self.condition_encoder = SceneConditionEncoder(ConditionEncoderConfig()).to(self.device)
        self.diffusion_model = DangerSceneDiffusion(DiffusionConfig()).to(self.device)

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

        self.scene_encoder.eval()
        self.condition_encoder.eval()
        self.diffusion_model.eval()

        print(f"[NuBoardVisualizer] Model loaded from {self.model_path}")

    def generate_trajectory(
        self,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor,
        svo_angle: float,
        inherent_danger: float,
        danger_target: float,
        graph_data=None
    ) -> np.ndarray:
        """Generate trajectory from diffusion model."""
        with torch.no_grad():
            if self.scene_encoder_mode == "simple":
                scene_feat = self.scene_encoder(ego_history.unsqueeze(0), agent_history.unsqueeze(0))
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

            condition = self.condition_encoder(scene_feat, svo_tensor, inherent_tensor, target_tensor,
                                               agent_state=agent_state_tensor)
            generated = self.diffusion_model.generate(condition)

        return generated.cpu().numpy().squeeze()

    def prepare_nuboard_data(
        self,
        num_samples: int = 5,
        output_dir: str = config.DIFFUSION_VISUALIZATION_RESULTS_DIR
    ) -> List[NuBoardScenarioData]:
        """
        Prepare NuBoard-compatible data from Diffusion outputs.

        Args:
            num_samples: Number of samples to prepare
            output_dir: Directory to save prepared data

        Returns:
            List of NuBoardScenarioData
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get pkl files
        pkl_files = list(self.data_dir.glob('danger_pair_features_*.pkl'))[:num_samples]
        print(f"[NuBoard] Preparing {len(pkl_files)} scenarios...")

        if not pkl_files:
            raise FileNotFoundError(
                f"No files matching 'danger_pair_features_*.pkl' found in {self.data_dir}"
            )

        scenarios: List[NuBoardScenarioData] = []

        for idx, pkl_file in enumerate(pkl_files):
            print(f"\n[Scenario {idx+1}/{len(pkl_files)}] {pkl_file.name}")

            # Load sample
            import pandas as pd
            data = pd.read_pickle(pkl_file)

            # Extract fields
            ego_history = data['ego_history']
            agent_history = data['key_agent_history']
            ego_future_gt = data['ego_future_gt']
            if ego_future_gt.ndim == 3:
                ego_future_gt = ego_future_gt.squeeze(0)

            scenario_name = data.get('scenario_name', f'scenario_{idx}')

            # Get starting state
            start_pos = ego_history[-1, :2]
            start_heading = ego_history[-1, 2]
            start_vx = ego_history[-1, 3]
            start_vy = ego_history[-1, 4]

            # Get agent future
            agent_future = self._normalize_agent_future(data.get('key_agent_future', None), agent_history)
            graph_data = self._load_graph_data(pkl_file) if self.scene_encoder_mode == "gnn_trans" else None

            # Compute SVO on-the-fly — pkl has no svo_angle field
            svo_angle = self.compute_pair_svo(
                ego_history, ego_future_gt[:60], agent_history, agent_future, self.danger_config
            )['ego_svo_angle_deg']

            # Compute GT danger
            gt_danger_result = self.compute_danger(
                ego_history, ego_future_gt[:60],
                agent_history, agent_future,
                self.danger_config
            )
            gt_danger = gt_danger_result['outcome_danger']['total_score']
            inherent_danger = gt_danger_result['inherent_danger']['total_score']

            # Generate trajectory
            ego_history_t = torch.tensor(ego_history, dtype=torch.float32, device=self.device)
            agent_history_t = torch.tensor(agent_history, dtype=torch.float32, device=self.device)

            danger_target = gt_danger + 2.0
            gen_increments = self.generate_trajectory(
                ego_history_t, agent_history_t,
                svo_angle, inherent_danger, danger_target,
                graph_data=graph_data.to(self.device) if graph_data is not None else None
            )

            # Recover trajectory — v3.9+ outputs 240-dim joint (ego 120 + agent 120)
            joint_mode = hasattr(gen_increments, '__len__') and len(gen_increments) == 240
            ego_inc = gen_increments[:120] if joint_mode else gen_increments
            start_speed = float(np.sqrt(start_vx**2 + start_vy**2))
            gen_traj = kinematic_decode_np(ego_inc, start_speed, start_heading, start_pos)

            # Compute generated danger
            gen_danger_result = self.compute_danger(
                ego_history, gen_traj.full_trajectory,
                agent_history, agent_future,
                self.danger_config
            )
            gen_danger = gen_danger_result['outcome_danger']['total_score']

            # Check validity
            validity = check_kinematic_validity(gen_traj)

            # Create scenario data
            scenario = NuBoardScenarioData(
                scenario_id=f"diffusion_{idx:03d}",
                log_name=scenario_name,
                timestamp=int(idx * 1000),
                ego_gt_trajectory=ego_future_gt[:60],
                ego_gen_trajectory=gen_traj.full_trajectory,
                ego_history=ego_history,
                agent_history=agent_history,
                agent_future=agent_future,
                gt_danger=gt_danger,
                gen_danger=gen_danger,
                svo_angle=svo_angle,
                is_valid=validity['overall_valid']
            )
            scenarios.append(scenario)

            # Save individual scenario
            scenario_file = output_path / f"scenario_{idx:03d}.pkl"
            with open(scenario_file, 'wb') as f:
                pickle.dump(scenario, f)

        # Save summary
        summary = {
            'num_scenarios': len(scenarios),
            'scenarios': [
                {
                    'id': s.scenario_id,
                    'log': s.log_name,
                    'gt_danger': float(s.gt_danger),
                    'gen_danger': float(s.gen_danger),
                    'margin_met': bool(s.gen_danger > s.gt_danger + 0.5),
                    'valid': bool(s.is_valid)
                }
                for s in scenarios
            ],
            'mean_gt_danger': float(np.mean([s.gt_danger for s in scenarios])),
            'mean_gen_danger': float(np.mean([s.gen_danger for s in scenarios])),
        }

        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[NuBoard] Data prepared in {output_path}")
        print(f"  Margin Met: {sum(1 for s in scenarios if s.gen_danger > s.gt_danger + 0.5)}/{len(scenarios)}")
        print(f"  Valid: {sum(1 for s in scenarios if s.is_valid)}/{len(scenarios)}")

        return scenarios

    def _load_graph_data(self, pkl_file: Path):
        """Load the processed graph input required by gnn_trans checkpoints."""
        import pandas as pd

        scenario_name = pkl_file.stem.replace('danger_pair_features_', '').split('_agent')[0]
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
    def _normalize_agent_future(agent_future: Optional[np.ndarray], agent_history: np.ndarray) -> np.ndarray:
        """Convert agent future to a stable (60, 5) array for danger evaluation and plotting."""
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

    @staticmethod
    def _sanitize_component_name(name: str, fallback: str) -> str:
        """Convert arbitrary names to filesystem and NuBoard friendly identifiers."""
        cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("._")
        return cleaned or fallback

    def _ensure_nuplan_imports(self) -> None:
        """Expose the local nuPlan devkit on sys.path after PyTorch is already imported."""
        nuplan_root = str(self.nuplan_root)
        if nuplan_root not in sys.path:
            sys.path.insert(0, nuplan_root)

    @staticmethod
    def _trajectory_to_ego_states(trajectory: np.ndarray):
        """Convert a `(T, 5)` trajectory to NuPlan EgoState objects."""
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

        vehicle_parameters = get_pacifica_parameters()
        ego_states = []

        for idx, state in enumerate(np.asarray(trajectory)):
            time_point = TimePoint(int(idx * 100000))
            ego_states.append(
                EgoState.build_from_rear_axle(
                    StateSE2(float(state[0]), float(state[1]), float(state[2])),
                    vehicle_parameters=vehicle_parameters,
                    rear_axle_velocity_2d=StateVector2D(x=float(state[3]), y=float(state[4])),
                    rear_axle_acceleration_2d=StateVector2D(x=0.0, y=0.0),
                    tire_steering_angle=0.0,
                    time_point=time_point,
                )
            )

        return ego_states

    @staticmethod
    def _trajectory_to_agent_detections(trajectory: np.ndarray):
        """Convert a `(T, 5)` agent trajectory to NuPlan tracked-object detections."""
        from nuplan.common.actor_state.agent import Agent
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.scene_object import SceneObjectMetadata
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

        detections = []
        track_token = "diffusion_key_agent"

        for idx, state in enumerate(np.asarray(trajectory)):
            timestamp_us = int(idx * 100000)
            agent = Agent(
                tracked_object_type=TrackedObjectType.VEHICLE,
                oriented_box=OrientedBox(
                    StateSE2(float(state[0]), float(state[1]), float(state[2])),
                    length=4.8,
                    width=2.0,
                    height=1.6,
                ),
                velocity=StateVector2D(x=float(state[3]), y=float(state[4])),
                metadata=SceneObjectMetadata(
                    timestamp_us=timestamp_us,
                    token=f"{track_token}_{idx}",
                    track_id=1,
                    track_token=track_token,
                    category_name="vehicle",
                ),
            )
            detections.append(DetectionsTracks(TrackedObjects([agent])))

        return detections

    def export_real_nuboard_bundle(
        self,
        scenarios: List[NuBoardScenarioData],
        output_dir: str = config.DIFFUSION_VISUALIZATION_RESULTS_DIR,
        planner_name: str = "DiffusionPlanner",
        scenario_type: str = "DiffusionGenerated",
    ) -> NuBoardBundlePaths:
        """
        Export prepared diffusion scenarios into a real NuBoard simulation bundle.

        The output can be opened directly by `launch_real_nuboard`.
        """
        if not scenarios:
            raise ValueError("No scenarios were provided for NuBoard export.")

        self._ensure_nuplan_imports()

        from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
        from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
        from nuplan.planning.simulation.simulation_log import SimulationLog
        from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        simulations_dir = output_path / "simulations"
        metrics_dir = output_path / "metrics"
        aggregator_dir = output_path / "aggregator_metric"
        simulations_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        aggregator_dir.mkdir(parents=True, exist_ok=True)

        safe_planner_name = self._sanitize_component_name(planner_name, "DiffusionPlanner")
        safe_scenario_type = self._sanitize_component_name(scenario_type, "DiffusionGenerated")

        exported = []

        for idx, scenario in enumerate(scenarios):
            safe_log_name = self._sanitize_component_name(scenario.log_name, f"log_{idx:03d}")
            safe_scenario_name = self._sanitize_component_name(scenario.scenario_id, f"scenario_{idx:03d}")

            gt_full = np.concatenate(
                [scenario.ego_history[-1:, :5], scenario.ego_gt_trajectory[:60, :5]],
                axis=0,
            )
            gen_full = np.concatenate(
                [scenario.ego_history[-1:, :5], scenario.ego_gen_trajectory[:60, :5]],
                axis=0,
            )
            agent_full = np.concatenate(
                [scenario.agent_history[-1:, :5], scenario.agent_future[:60, :5]],
                axis=0,
            )

            expert_states = self._trajectory_to_ego_states(gt_full)
            generated_states = self._trajectory_to_ego_states(gen_full)
            tracked_detections = self._trajectory_to_agent_detections(agent_full)

            scenario_object = DiffusionMockScenario(
                scenario_name=safe_scenario_name,
                log_name=safe_log_name,
                scenario_type=safe_scenario_type,
                expert_states=expert_states,
                tracked_detections=tracked_detections,
                time_step=0.1,
            )

            history = SimulationHistory(
                scenario_object.map_api,
                scenario_object.get_mission_goal(),
            )

            for step, ego_state in enumerate(generated_states):
                end_index = min(step + 20, len(generated_states))
                planned_states = generated_states[step:end_index]
                if len(planned_states) < 2:
                    planned_states = generated_states[max(step - 1, 0):step + 1]
                history.add_sample(
                    SimulationHistorySample(
                        iteration=SimulationIteration(time_point=ego_state.time_point, index=step),
                        ego_state=ego_state,
                        trajectory=InterpolatedTrajectory(planned_states),
                        observation=tracked_detections[min(step, len(tracked_detections) - 1)],
                        traffic_light_status=list(scenario_object.get_traffic_light_status_at_iteration(0)),
                    )
                )

            planner = SimplePlanner(horizon_seconds=2.0, sampling_time=0.1, acceleration=np.array([0.0, 0.0]))
            scenario_dir = simulations_dir / safe_planner_name / safe_scenario_type / safe_log_name / safe_scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            simulation_log_path = scenario_dir / f"{safe_scenario_name}.msgpack.xz"

            simulation_log = SimulationLog(
                file_path=simulation_log_path,
                scenario=scenario_object,
                planner=planner,
                simulation_history=history,
            )
            self._save_local_simulation_log(simulation_log)

            exported.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "log_name": scenario.log_name,
                    "simulation_log": str(simulation_log_path),
                    "gt_danger": float(scenario.gt_danger),
                    "gen_danger": float(scenario.gen_danger),
                    "valid": bool(scenario.is_valid),
                }
            )

        nuboard_file = output_path / "diffusion_generated.nuboard"
        self._save_local_nuboard_file(
            nuboard_file=nuboard_file,
            simulation_main_path=str(output_path),
            metric_main_path=str(output_path),
            metric_folder=metrics_dir.name,
            aggregator_metric_folder=aggregator_dir.name,
            simulation_folder=simulations_dir.name,
        )

        bundle_summary = {
            "nuboard_file": str(nuboard_file),
            "simulations_dir": str(simulations_dir),
            "planner_name": safe_planner_name,
            "scenario_type": safe_scenario_type,
            "num_scenarios": len(exported),
            "scenarios": exported,
        }
        with open(output_path / "nuboard_bundle_summary.json", "w") as f:
            json.dump(bundle_summary, f, indent=2)

        print(f"[NuBoard] Real NuBoard bundle exported to {output_path}")
        print(f"  nuboard file: {nuboard_file}")
        print(f"  simulation dir: {simulations_dir}")

        return NuBoardBundlePaths(
            output_dir=output_path,
            simulations_dir=simulations_dir,
            metrics_dir=metrics_dir,
            aggregator_dir=aggregator_dir,
            nuboard_file=nuboard_file,
        )

    @staticmethod
    def _save_local_simulation_log(simulation_log) -> None:
        """Serialize SimulationLog locally without relying on nuPlan's async file helpers."""
        simulation_log.file_path.parent.mkdir(parents=True, exist_ok=True)
        pickle_object = pickle.dumps(simulation_log, protocol=pickle.HIGHEST_PROTOCOL)
        msg_packed_bytes = msgpack.packb(pickle_object)
        with lzma.open(simulation_log.file_path, "wb", preset=0) as f:
            f.write(msg_packed_bytes)

    @staticmethod
    def _save_local_nuboard_file(
        nuboard_file: Path,
        simulation_main_path: str,
        metric_main_path: str,
        metric_folder: str,
        aggregator_metric_folder: str,
        simulation_folder: str,
    ) -> None:
        """Write a `.nuboard` file locally using the serialized dictionary format NuBoard expects."""
        payload = {
            "simulation_main_path": simulation_main_path,
            "metric_main_path": metric_main_path,
            "metric_folder": metric_folder,
            "aggregator_metric_folder": aggregator_metric_folder,
            "simulation_folder": simulation_folder,
        }
        with open(nuboard_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def create_bokeh_visualization(
        self,
        scenarios: List[NuBoardScenarioData],
        output_dir: str = config.DIFFUSION_VISUALIZATION_RESULTS_DIR
    ) -> Path:
        """
        Create standalone Bokeh visualization (lightweight nuBoard alternative).

        This creates an HTML visualization without requiring full NuPlan nuBoard setup.

        Args:
            scenarios: List of prepared scenario data
            output_dir: Directory to save visualization

        Returns:
            Path to HTML file
        """
        if not scenarios:
            raise ValueError("No scenarios provided for visualization.")

        try:
            from bokeh.plotting import figure, output_file, save, show
            from bokeh.models import ColumnDataSource, Slider, CustomJS, HoverTool
            from bokeh.layouts import column, row
            from bokeh.io import curdoc
            from bokeh.models.widgets import Tabs, Panel
        except ImportError:
            print("[Warning] Bokeh not installed. Creating simple HTML visualization.")
            return self._create_simple_html_visualization(scenarios, output_dir)

        output_path = Path(output_dir)
        html_file = output_path / "diffusion_visualization.html"
        output_file(str(html_file))

        # Create tabs for each scenario
        tabs_list = []

        for idx, scenario in enumerate(scenarios):
            # Create figure
            p = figure(
                title=f"Scenario {idx}: {scenario.log_name}",
                x_axis_label='X (m)',
                y_axis_label='Y (m)',
                width=800,
                height=600,
                match_aspect=True
            )

            # Plot ego history
            ego_hist_source = ColumnDataSource(data={
                'x': scenario.ego_history[:, 0],
                'y': scenario.ego_history[:, 1],
                't': np.arange(len(scenario.ego_history)) * 0.1
            })
            p.line('x', 'y', source=ego_hist_source, line_width=2, line_color='blue',
                   legend_label='Ego History', alpha=0.5)

            # Plot agent history
            agent_hist_source = ColumnDataSource(data={
                'x': scenario.agent_history[:, 0],
                'y': scenario.agent_history[:, 1],
                't': np.arange(len(scenario.agent_history)) * 0.1
            })
            p.line('x', 'y', source=agent_hist_source, line_width=2, line_color='orange',
                   legend_label='Agent History', alpha=0.5)

            # Plot GT future
            gt_source = ColumnDataSource(data={
                'x': scenario.ego_gt_trajectory[:, 0],
                'y': scenario.ego_gt_trajectory[:, 1],
                't': np.arange(60) * 0.1 + 4.0  # Start at 4s
            })
            p.line('x', 'y', source=gt_source, line_width=3, line_color='green',
                   legend_label='GT Future')

            # Plot Generated future
            gen_source = ColumnDataSource(data={
                'x': scenario.ego_gen_trajectory[:, 0],
                'y': scenario.ego_gen_trajectory[:, 1],
                't': np.arange(60) * 0.1 + 4.0
            })
            p.line('x', 'y', source=gen_source, line_width=3, line_color='red',
                   legend_label='Generated', line_dash='dashed')

            # Add hover tool
            hover = HoverTool(tooltips=[("Time", "@t{s}"), ("X", "@x"), ("Y", "@y")])
            p.add_tools(hover)

            p.legend.location = 'top_right'
            p.grid.visible = True

            # Create panel
            panel = Panel(child=p, title=f"Scenario {idx}")
            tabs_list.append(panel)

        tabs = Tabs(tabs=tabs_list)

        # Add summary plot
        summary_p = figure(
            title="Danger Score Comparison",
            x_axis_label='Scenario',
            y_axis_label='Danger Score',
            width=800,
            height=300
        )

        scenario_ids = [s.scenario_id for s in scenarios]
        gt_dangers = [s.gt_danger for s in scenarios]
        gen_dangers = [s.gen_danger for s in scenarios]

        summary_p.vbar(x=scenario_ids, top=gt_dangers, width=0.4, color='green',
                       legend_label='GT Danger', alpha=0.7)
        summary_p.vbar(x=scenario_ids, top=gen_dangers, width=0.4, color='red',
                       legend_label='Gen Danger', alpha=0.7)

        summary_p.legend.location = 'top_right'

        # Combine layout
        layout = column(tabs, summary_p)
        save(layout)

        print(f"[NuBoard] Bokeh visualization saved to {html_file}")
        return html_file

    def _create_simple_html_visualization(
        self,
        scenarios: List[NuBoardScenarioData],
        output_dir: str
    ) -> Path:
        """Create simple HTML visualization without Bokeh."""
        if not scenarios:
            raise ValueError("No scenarios provided for visualization.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        html_file = output_path / "diffusion_visualization_simple.html"

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Diffusion Trajectory Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .scenario { margin: 20px 0; padding: 10px; border: 1px solid #ddd; }
        .info { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        canvas { border: 1px solid #ccc; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Diffusion Dangerous Scene Generator - Trajectory Visualization</h1>
"""

        for idx, scenario in enumerate(scenarios):
            html_content += f"""
    <div class="scenario">
        <h2>Scenario {idx}: {scenario.log_name}</h2>
        <div class="info">
            <strong>Danger Scores:</strong> GT={scenario.gt_danger:.2f}, Generated={scenario.gen_danger:.2f}
            <br><strong>Margin Met:</strong> {scenario.gen_danger > scenario.gt_danger + 0.5}
            <br><strong>Valid:</strong> {scenario.is_valid}
            <br><strong>SVO Angle:</strong> {scenario.svo_angle:.1f}°
        </div>
        <canvas id="chart_{idx}" width="600" height="400"></canvas>
    </div>
"""

        html_content += """
    <script>
"""
        # Add chart.js code for each scenario
        for idx, scenario in enumerate(scenarios):
            gt_points = [
                {'x': float(x), 'y': float(y)}
                for x, y in zip(
                    list(scenario.ego_history[:, 0]) + list(scenario.ego_gt_trajectory[:, 0]),
                    list(scenario.ego_history[:, 1]) + list(scenario.ego_gt_trajectory[:, 1])
                )
            ]
            gen_points = [
                {'x': float(x), 'y': float(y)}
                for x, y in zip(
                    list(scenario.ego_history[:, 0]) + list(scenario.ego_gen_trajectory[:, 0]),
                    list(scenario.ego_history[:, 1]) + list(scenario.ego_gen_trajectory[:, 1])
                )
            ]
            agent_points = [
                {'x': float(x), 'y': float(y)}
                for x, y in zip(scenario.agent_history[:, 0], scenario.agent_history[:, 1])
            ]

            html_content += f"""
        const ctx_{idx} = document.getElementById('chart_{idx}').getContext('2d');
        new Chart(ctx_{idx}, {{
            type: 'scatter',
            data: {{
                datasets: [
                    {{
                        label: 'GT Trajectory',
                        data: {json.dumps(gt_points)},
                        borderColor: 'green',
                        showLine: true,
                        fill: false
                    }},
                    {{
                        label: 'Generated Trajectory',
                        data: {json.dumps(gen_points)},
                        borderColor: 'red',
                        showLine: true,
                        fill: false,
                        borderDash: [5, 5]
                    }},
                    {{
                        label: 'Agent',
                        data: {json.dumps(agent_points)},
                        borderColor: 'orange',
                        showLine: true,
                        fill: false
                    }}
                ]
            }},
            options: {{
                scales: {{
                    x: {{ title: {{ display: true, text: 'X (m)' }} }},
                    y: {{ title: {{ display: true, text: 'Y (m)' }} }}
                }}
            }}
        }});
"""

        html_content += """
    </script>
</body>
</html>
"""

        with open(html_file, 'w') as f:
            f.write(html_content)

        print(f"[NuBoard] Simple HTML visualization saved to {html_file}")
        return html_file

    def launch_real_nuboard(
        self,
        nuboard_file: str,
        port: int = 5006,
    ) -> None:
        """
        Launch a real NuBoard server from an exported `.nuboard` bundle.

        Args:
            nuboard_file: Path to the exported `.nuboard` file
            port: Port number for the NuBoard server
        """
        self._ensure_nuplan_imports()

        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.nuboard.nuboard import NuBoard
        from nuplan.planning.scenario_builder.test.mock_abstract_scenario_builder import MockAbstractScenarioBuilder

        nuboard_path = Path(nuboard_file).resolve()
        if not nuboard_path.exists():
            raise FileNotFoundError(f"NuBoard file not found: {nuboard_path}")

        os.environ.setdefault("NUPLAN_EXP_ROOT", str(nuboard_path.parent))

        print("[NuBoard] Launching real NuBoard server...")
        print(f"  nuboard file: {nuboard_path}")
        print(f"  URL: http://localhost:{port}")
        print("  Press Ctrl+C to stop the server")

        nuboard = NuBoard(
            profiler_path=nuboard_path.parent,
            nuboard_paths=[str(nuboard_path)],
            scenario_builder=MockAbstractScenarioBuilder(),
            vehicle_parameters=get_pacifica_parameters(),
            port_number=port,
        )
        nuboard.run()

    def export_and_launch_nuboard(
        self,
        num_samples: int = 5,
        output_dir: str = config.DIFFUSION_VISUALIZATION_RESULTS_DIR,
        port: int = 5006,
    ) -> NuBoardBundlePaths:
        """
        Prepare diffusion scenarios, convert them into NuBoard simulation data, and launch NuBoard.
        """
        scenarios = self.prepare_nuboard_data(num_samples=num_samples, output_dir=output_dir)
        bundle = self.export_real_nuboard_bundle(scenarios=scenarios, output_dir=output_dir)
        self.launch_real_nuboard(str(bundle.nuboard_file), port=port)
        return bundle


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NuBoard Trajectory Visualization')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default=config.DANGER_PAIR_SCENARIOS_DIR)
    parser.add_argument('--processed_data_dir', type=str, default=config.PROCESSED_DATA_DIR)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default=config.DIFFUSION_VISUALIZATION_RESULTS_DIR)
    parser.add_argument('--mode', type=str, default='prepare',
                        choices=['prepare', 'bokeh', 'export', 'launch'],
                        help='prepare: prepare data only; bokeh: create html visualization; export: export real NuBoard bundle; launch: export and launch real NuBoard')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nuplan_root', type=str, default='/mnt/local/Xiao_Yi/nuplan-devkit')
    parser.add_argument('--port', type=int, default=5006)

    args = parser.parse_args()

    viz = NuBoardVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        nuplan_root=args.nuplan_root,
        device=args.device,
    )

    if args.mode == 'prepare':
        scenarios = viz.prepare_nuboard_data(args.num_samples, args.output_dir)

    elif args.mode == 'bokeh':
        scenarios = viz.prepare_nuboard_data(args.num_samples, args.output_dir)
        viz.create_bokeh_visualization(scenarios, args.output_dir)

    elif args.mode == 'export':
        scenarios = viz.prepare_nuboard_data(args.num_samples, args.output_dir)
        viz.export_real_nuboard_bundle(scenarios, args.output_dir)

    elif args.mode == 'launch':
        viz.export_and_launch_nuboard(args.num_samples, args.output_dir, args.port)
