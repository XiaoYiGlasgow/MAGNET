"""
GNN_Trans Planner for nuBoard visualization
"""

import numpy as np
import torch
from typing import List, Type

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.planner_report import PlannerInput, PlannerReport
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

import sys
import os
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import process_scenario_for_inference


class GNNTransPlanner(AbstractPlanner):
    """
    GNN_Trans trajectory prediction planner for nuBoard visualization.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize GNN_Trans planner.

        Args:
            model_path: Path to trained model weights
        """
        super().__init__()

        # Load model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerPredictor().to(self.device)

        # Load trained weights
        if model_path is None:
            model_path = "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/GNN_Trans_params/best_model.pth"

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded GNN_Trans model from: {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")

        self.prediction_horizon = 3.0  # 3 seconds
        self.prediction_steps = 30     # 30 time steps

    def name(self) -> str:
        """Return planner name."""
        return "GNNTransPlanner"

    def observation_type(self) -> Type:
        """Return observation type."""
        from nuplan.planning.simulation.observation.observation_type import Observation
        return Observation

    def compute_planner_trajectory(self, planner_input: PlannerInput) -> InterpolatedTrajectory:
        """
        Compute trajectory using GNN_Trans model.

        Args:
            planner_input: Input containing current scenario state

        Returns:
            InterpolatedTrajectory: Predicted trajectory
        """
        try:
            # Extract scenario data
            current_input = planner_input.initialization
            history = planner_input.history

            # Process scenario data for model input
            model_input = self._process_scenario_data(current_input, history)

            # Run model prediction
            with torch.no_grad():
                prediction = self.model(model_input)

            # Convert prediction to trajectory
            trajectory = self._prediction_to_trajectory(prediction, current_input.rear_axle)

            return trajectory

        except Exception as e:
            print(f"Error in GNN_Trans prediction: {e}")
            # Return simple straight trajectory as fallback
            return self._get_fallback_trajectory(planner_input)

    def _process_scenario_data(self, current_input, history):
        """
        Process nuplan scenario data into GNN_Trans model input format.
        """
        # This is a simplified version - you may need to adapt based on your exact data format
        current_state = current_input.rear_axle

        # Create mock data structure similar to your training data
        # You'll need to adapt this based on your exact input format
        data = torch.zeros(1, 100, 4).to(self.device)  # Mock input

        return data

    def _prediction_to_trajectory(self, prediction, current_state):
        """
        Convert model prediction to nuplan trajectory format.
        """
        # Convert prediction tensor to trajectory points
        pred_array = prediction.cpu().numpy().reshape(-1, 2)

        # Create trajectory states
        trajectory_states = []
        current_x = current_state.x
        current_y = current_state.y
        current_heading = current_state.heading

        # Time step (assuming 0.1s intervals)
        dt = 0.1

        for i, (dx, dy) in enumerate(pred_array):
            # Accumulate incremental predictions
            current_x += dx
            current_y += dy

            # Create state
            state = StateSE2(current_x, current_y, current_heading)
            trajectory_states.append(state)

        # Create timestamps
        timestamps = [i * dt for i in range(len(trajectory_states))]

        return InterpolatedTrajectory(trajectory_states, timestamps)

    def _get_fallback_trajectory(self, planner_input):
        """
        Generate a simple fallback trajectory.
        """
        current_state = planner_input.initialization.rear_axle

        # Simple straight line trajectory
        trajectory_states = []
        dt = 0.1
        velocity = 5.0  # m/s

        for i in range(self.prediction_steps):
            time = i * dt
            x = current_state.x + velocity * time * np.cos(current_state.heading)
            y = current_state.y + velocity * time * np.sin(current_state.heading)

            state = StateSE2(x, y, current_state.heading)
            trajectory_states.append(state)

        timestamps = [i * dt for i in range(len(trajectory_states))]

        return InterpolatedTrajectory(trajectory_states, timestamps)
