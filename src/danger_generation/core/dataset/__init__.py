# 注意：旧 DangerPairDataset（_load 用旧 cumsum，对 (60,5) 绝对数据会展平成 300 维直接崩）
# 已于 v8 清理归档到 archive/，不再导出。训练用 train_diffusion_gnn_trans._load_single_sample，
# benchmark/eval 用 eval_v5_svo.SimpleEvalDataset。
from .trajectory_utils import (
    increments_to_positions,
    compute_velocity_from_increments,
    positions_to_increments,
    build_ego_future_5col,
    build_agent_future_5col,
    global_to_ego_centric,
)

__all__ = [
    "increments_to_positions",
    "compute_velocity_from_increments",
    "positions_to_increments",
    "build_ego_future_5col",
    "build_agent_future_5col",
    "global_to_ego_centric",
]
