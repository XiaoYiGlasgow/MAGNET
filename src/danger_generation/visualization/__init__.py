"""
Diffusion Trajectory Visualization Module

提供两种可视化方案:
1. Matplotlib quick validation - 快速检查生成轨迹质量
2. NuBoard detailed display - 包含地图背景的详细可视化

Usage:
    # Quick validation
    python visualization/run_evaluation.py --mode quick --model_path best_model.pt

    # Bokeh visualization
    python visualization/run_evaluation.py --mode bokeh --model_path best_model.pt

    # Full evaluation (both modes)
    python visualization/run_evaluation.py --mode full --model_path best_model.pt
"""

from pathlib import Path

VISUALIZATION_FACTORY: dict = {}


def register_visualizer(name: str):
    """Registry decorator for visualizers."""
    def decorator(cls):
        VISUALIZATION_FACTORY[name] = cls
        return cls
    return decorator


def VisualizerFactory(name: str):
    """Factory for creating visualizers."""
    visualizer = VISUALIZATION_FACTORY.get(name, None)
    if visualizer is None:
        raise ValueError(f"Visualizer '{name}' not found. Available: {list(VISUALIZATION_FACTORY.keys())}")
    return visualizer


# Import submodules to trigger registration
from .trajectory_converter import (
    recover_trajectory_from_increment,
    compute_trajectory_metrics,
    check_trajectory_validity,
    TrajectoryData
)
from .matplotlib_visualizer import MatplotlibVisualizer
from .nuboard_visualizer import NuBoardVisualizer


__all__ = [
    "register_visualizer",
    "VisualizerFactory",
    "VISUALIZATION_FACTORY",
    "MatplotlibVisualizer",
    "NuBoardVisualizer",
    "TrajectoryData",
    "recover_trajectory_from_increment",
    "compute_trajectory_metrics",
    "check_trajectory_validity",
]