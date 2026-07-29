"""
GNN_Trans Models Module

包含：
- scene_condition_encoder.py: 场景条件编码器
- danger_diffusion.py: 危险场景Diffusion生成器
"""

from .scene_condition_encoder import (
    SceneConditionEncoder,
    SimpleSceneEncoder,
    MultiAgentConditionEncoder,
    ConditionEncoderConfig
)

from .danger_diffusion import (
    DangerSceneDiffusion,
    SimpleDiffusionGenerator,
    DiffusionConfig,
    TrajectoryDenoiser,
    SinusoidalPosEmb,
    TimeStepMLP
)

__all__ = [
    'SceneConditionEncoder',
    'SimpleSceneEncoder',
    'MultiAgentConditionEncoder',
    'ConditionEncoderConfig',
    'DangerSceneDiffusion',
    'SimpleDiffusionGenerator',
    'DiffusionConfig',
    'TrajectoryDenoiser',
    'SinusoidalPosEmb',
    'TimeStepMLP'
]