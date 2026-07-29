"""
危险评分模块 - PyTorch版本

用于Diffusion训练中的danger loss计算。
将numpy版本的核心函数转换为PyTorch张量操作，支持GPU加速和自动梯度。

核心组件：
1. TTC计算（纵向和横向）
2. 距离危险评分
3. 碰撞检测
4. 危险评分总和

用于训练时的danger_exceed_loss计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# 配置参数（与numpy版本保持一致）
# ============================================================

@dataclass
class DangerScoreConfigTorch:
    """危险评分计算配置 - PyTorch版本"""

    # TTC阈值
    ttc_critical: float = 1.0   # 临界TTC（秒）
    ttc_warning: float = 3.0    # 警告TTC（秒）
    ttc_max: float = 10.0       # 最大考虑的TTC（秒）

    # 距离阈值
    min_safe_distance: float = 5.0      # 最小安全距离（米）
    collision_distance: float = 2.0     # 认定为碰撞的距离（米）
    near_miss_distance: float = 3.0     # 近距离阈值（米）

    # 车辆尺寸（用于碰撞检测）
    ego_length: float = 4.5
    ego_width: float = 2.0
    agent_length: float = 4.0
    agent_width: float = 2.0

    # 车道相关参数
    lane_width: float = 3.5
    same_lane_threshold: float = 1.75
    adjacent_lane_threshold: float = 5.25
    lane_change_vy_threshold: float = 0.5

    # 不同危险组件的权重
    weight_ttc_longitudinal: float = 1.0
    weight_ttc_lateral: float = 0.8
    weight_ttc_same_lane: float = 1.5
    weight_min_distance: float = 1.5
    weight_collision: float = 10.0
    weight_near_miss: float = 3.0
    weight_relative_velocity: float = 0.5
    weight_acceleration: float = 0.3
    weight_lane_change_conflict: float = 2.0

    # 物理约束
    max_acceleration: float = 3.0
    max_deceleration: float = 5.0
    max_lateral_accel: float = 2.0

    # SVO参数（简化，主要用于推理）
    svo_angle_min: float = -45.0
    svo_angle_max: float = 135.0
    svo_decel_threshold: float = 3.0
    svo_accel_threshold: float = 2.0
    svo_lateral_speed_threshold: float = 0.5
    svo_risk_distance_threshold: float = 8.0
    weight_svo_agent: float = 0.5
    weight_svo_pair: float = 0.8


def get_default_config() -> DangerScoreConfigTorch:
    """获取默认配置"""
    return DangerScoreConfigTorch()


# ============================================================
# 辅助函数
# ============================================================

def compute_ttc_1d_torch(
    distance: torch.Tensor,
    relative_velocity: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算一维TTC（碰撞时间） - PyTorch版本

    Args:
        distance: 车辆间距离（正值） [batch]
        relative_velocity: 相对速度（负值表示接近） [batch]
        epsilon: 避免除零的小值

    Returns:
        TTC值（如果不发生碰撞则返回inf）
    """
    # 正在远离，不会碰撞
    no_collision = relative_velocity >= 0

    # 已经重叠
    already_overlap = distance <= 0

    # 正常计算
    ttc = distance / (-relative_velocity + epsilon)

    # 处理特殊情况
    ttc = torch.where(no_collision, torch.tensor(float('inf'), device=ttc.device), ttc)
    ttc = torch.where(already_overlap, torch.tensor(0.0, device=ttc.device), ttc)

    return ttc


def compute_relative_state_torch(
    ego_state: torch.Tensor,
    agent_state: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    计算ego和agent在单个时间步的相对状态 - PyTorch版本

    Args:
        ego_state: [batch, 4] 或 [batch, 5] - [x, y, heading, vx, vy]
        agent_state: [batch, 4] 或 [batch, 5]

    Returns:
        包含相对状态信息的字典
    """
    # 提取位置
    ego_x = ego_state[:, 0]
    ego_y = ego_state[:, 1]
    agent_x = agent_state[:, 0]
    agent_y = agent_state[:, 1]

    # 提取速度（处理4D和5D两种格式）
    ego_vx = ego_state[:, -2]
    ego_vy = ego_state[:, -1]
    agent_vx = agent_state[:, -2]
    agent_vy = agent_state[:, -1]

    # 相对位置（ego坐标系）
    rel_x = agent_x - ego_x   # 正值 = agent在前方
    rel_y = agent_y - ego_y   # 正值 = agent在左侧

    # 相对速度
    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    # 距离计算（+eps 防止两车重叠时 sqrt 梯度为 inf）
    longitudinal_dist = torch.abs(rel_x)
    lateral_dist = torch.abs(rel_y)
    euclidean_dist = torch.sqrt(rel_x**2 + rel_y**2 + 1e-8)

    # 相对速率（标量，+eps 防止零向量 sqrt 梯度为 inf）
    relative_speed = torch.sqrt(rel_vx**2 + rel_vy**2 + 1e-8)

    return {
        'rel_x': rel_x,
        'rel_y': rel_y,
        'rel_vx': rel_vx,
        'rel_vy': rel_vy,
        'longitudinal_dist': longitudinal_dist,
        'lateral_dist': lateral_dist,
        'euclidean_dist': euclidean_dist,
        'relative_speed': relative_speed
    }


def compute_ttc_at_timestep_torch(
    ego_state: torch.Tensor,
    agent_state: torch.Tensor,
    config: DangerScoreConfigTorch
) -> Dict[str, torch.Tensor]:
    """
    计算单个时间步的TTC值 - PyTorch版本

    Args:
        ego_state: [batch, D] ego状态
        agent_state: [batch, D] agent状态
        config: 配置参数

    Returns:
        包含TTC值的字典
    """
    rel_state = compute_relative_state_torch(ego_state, agent_state)

    # 纵向TTC
    # Agent在前方：rel_x > 0，如果rel_vx < 0（agent更慢）则在接近
    agent_front = rel_state['rel_x'] > 0
    lon_ttc_front = compute_ttc_1d_torch(
        rel_state['rel_x'],
        rel_state['rel_vx']
    )
    lon_ttc_back = compute_ttc_1d_torch(
        torch.abs(rel_state['rel_x']),
        -rel_state['rel_vx']
    )
    lon_ttc = torch.where(agent_front, lon_ttc_front, lon_ttc_back)

    # 横向TTC
    agent_left = rel_state['rel_y'] > 0
    lat_ttc_left = compute_ttc_1d_torch(
        rel_state['rel_y'],
        rel_state['rel_vy']
    )
    lat_ttc_right = compute_ttc_1d_torch(
        torch.abs(rel_state['rel_y']),
        -rel_state['rel_vy']
    )
    lat_ttc = torch.where(agent_left, lat_ttc_left, lat_ttc_right)

    return {
        'lon_ttc': lon_ttc,
        'lat_ttc': lat_ttc,
        'euclidean_dist': rel_state['euclidean_dist'],
        'relative_speed': rel_state['relative_speed'],
        'rel_state': rel_state
    }


# ============================================================
# 危险评分组件 - PyTorch版本
# ============================================================

def ttc_danger_score_torch(
    lon_ttc: torch.Tensor,
    lat_ttc: torch.Tensor,
    config: DangerScoreConfigTorch
) -> torch.Tensor:
    """
    基于TTC值计算危险评分 - PyTorch版本

    Args:
        lon_ttc: [batch] 纵向TTC
        lat_ttc: [batch] 横向TTC
        config: 配置参数

    Returns:
        [batch] TTC相关的危险评分
    """
    score = torch.zeros_like(lon_ttc)

    # 纵向TTC危险
    critical_mask_lon = lon_ttc < config.ttc_critical
    warning_mask_lon = (lon_ttc >= config.ttc_critical) & (lon_ttc < config.ttc_warning)
    max_mask_lon = (lon_ttc >= config.ttc_warning) & (lon_ttc < config.ttc_max)

    score = score + torch.where(
        critical_mask_lon,
        config.weight_ttc_longitudinal * (config.ttc_critical - lon_ttc) * 2.0,
        torch.zeros_like(score)
    )
    score = score + torch.where(
        warning_mask_lon,
        config.weight_ttc_longitudinal * (config.ttc_warning - lon_ttc),
        torch.zeros_like(score)
    )
    score = score + torch.where(
        max_mask_lon,
        config.weight_ttc_longitudinal * (config.ttc_max - lon_ttc) * 0.1,
        torch.zeros_like(score)
    )

    # 横向TTC危险
    critical_mask_lat = lat_ttc < config.ttc_critical
    warning_mask_lat = (lat_ttc >= config.ttc_critical) & (lat_ttc < config.ttc_warning)
    max_mask_lat = (lat_ttc >= config.ttc_warning) & (lat_ttc < config.ttc_max)

    score = score + torch.where(
        critical_mask_lat,
        config.weight_ttc_lateral * (config.ttc_critical - lat_ttc) * 2.0,
        torch.zeros_like(score)
    )
    score = score + torch.where(
        warning_mask_lat,
        config.weight_ttc_lateral * (config.ttc_warning - lat_ttc),
        torch.zeros_like(score)
    )
    score = score + torch.where(
        max_mask_lat,
        config.weight_ttc_lateral * (config.ttc_max - lat_ttc) * 0.1,
        torch.zeros_like(score)
    )

    return score


def distance_danger_score_torch(
    min_distance: torch.Tensor,
    config: DangerScoreConfigTorch
) -> torch.Tensor:
    """
    基于最小距离计算危险评分 - PyTorch版本（near-miss范式）

    帐篷函数：峰值在 near_miss_distance（3m），碰撞区（<=2m）返回0。
    设计依据：STRIVE near-miss范式，碰撞不给奖励，near-miss才是训练目标。

    Args:
        min_distance: [batch] 车辆间最小距离
        config: 配置参数

    Returns:
        [batch] 距离相关的危险评分
    """
    near_miss_peak = config.weight_near_miss  # 3.0

    score = torch.zeros_like(min_distance)

    # 碰撞区：返回0（不给奖励）
    collision_mask = min_distance <= config.collision_distance

    # 上升侧：collision_distance(2m) → near_miss_distance(3m)，0 → peak
    near_miss_mask = (min_distance > config.collision_distance) & (min_distance <= config.near_miss_distance)
    near_miss_score = near_miss_peak * (min_distance - config.collision_distance) / \
                      (config.near_miss_distance - config.collision_distance)
    score = torch.where(near_miss_mask, near_miss_score, score)

    # 下降侧：near_miss_distance(3m) → min_safe_distance(5m)，peak → 0
    safe_mask = (min_distance > config.near_miss_distance) & (min_distance <= config.min_safe_distance)
    safe_score = near_miss_peak * (config.min_safe_distance - min_distance) / \
                 (config.min_safe_distance - config.near_miss_distance)
    score = torch.where(safe_mask, safe_score, score)

    # collision_mask 区域 score 已为0，无需额外处理
    _ = collision_mask  # 显式保留以便后续调试

    return score


def collision_detection_torch(
    ego_traj: torch.Tensor,
    agent_traj: torch.Tensor,
    config: DangerScoreConfigTorch
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    检测ego和agent轨迹之间的碰撞 - PyTorch版本

    Args:
        ego_traj: [batch, T, D] Ego轨迹
        agent_traj: [batch, T, D] Agent轨迹
        config: 配置参数

    Returns:
        (has_collision, min_distance, collision_timestep)
        has_collision: [batch] bool
        min_distance: [batch] float
        collision_timestep: [batch] int
    """
    batch_size = ego_traj.size(0)
    T = ego_traj.size(1)
    device = ego_traj.device

    # 计算所有时间步的距离 [batch, T]
    ego_x = ego_traj[:, :, 0]
    ego_y = ego_traj[:, :, 1]
    agent_x = agent_traj[:, :, 0]
    agent_y = agent_traj[:, :, 1]

    dist = torch.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2 + 1e-8)

    # 最小距离 [batch]
    min_distance = dist.min(dim=1)[0]

    # 碰撞阈值
    collision_threshold = (config.ego_length + config.agent_length) / 4 + \
                          (config.ego_width + config.agent_width) / 4

    # 碰撞检测 [batch, T]
    collision_at_t = dist < collision_threshold

    # 是否有碰撞 [batch]
    has_collision = collision_at_t.any(dim=1)

    # 碰撞时间步 [batch]（首次碰撞的时间步）
    # 找到每个batch中第一个True的位置
    collision_timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
    for b in range(batch_size):
        if collision_at_t[b].any():
            collision_timestep[b] = collision_at_t[b].nonzero()[0, 0]
        else:
            collision_timestep[b] = -1  # 无碰撞

    return has_collision, min_distance, collision_timestep


def relative_velocity_danger_score_torch(
    ego_traj: torch.Tensor,
    agent_traj: torch.Tensor,
    config: DangerScoreConfigTorch
) -> torch.Tensor:
    """
    基于相对速度计算危险评分 - PyTorch版本

    Args:
        ego_traj: [batch, T, D] Ego轨迹
        agent_traj: [batch, T, D] Agent轨迹
        config: 配置参数

    Returns:
        [batch] 速度相关的危险评分
    """
    batch_size = ego_traj.size(0)
    T = ego_traj.size(1)
    device = ego_traj.device

    # 计算相对状态
    ego_vx = ego_traj[:, :, -2]
    ego_vy = ego_traj[:, :, -1]
    agent_vx = agent_traj[:, :, -2]
    agent_vy = agent_traj[:, :, -1]

    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    ego_x = ego_traj[:, :, 0]
    ego_y = ego_traj[:, :, 1]
    agent_x = agent_traj[:, :, 0]
    agent_y = agent_traj[:, :, 1]
    euclidean_dist = torch.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2 + 1e-8)

    # 接近速度（closing speed）：相对速度在两车连线方向的分量
    # dir: 从 ego 指向 agent 的单位向量；closing_speed > 0 表示两车正在接近
    dir_x = (agent_x - ego_x) / euclidean_dist
    dir_y = (agent_y - ego_y) / euclidean_dist
    closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)   # 负号: agent 接近 ego 时为正
    closing_speed = torch.clamp(closing_speed, min=0.0)   # 背离时不计分

    # 近距离 + 正在接近 才危险
    near_mask = euclidean_dist < config.min_safe_distance
    very_near_mask = euclidean_dist < config.near_miss_distance

    speed_score = closing_speed * config.weight_relative_velocity
    speed_score = torch.where(very_near_mask, speed_score * 2.0, speed_score)
    speed_score = torch.where(near_mask, speed_score, torch.zeros_like(speed_score))

    # 取每个batch的最大值
    max_score = speed_score.max(dim=1)[0]

    return max_score


# ============================================================
# 主危险评分函数 - PyTorch版本
# ============================================================

def compute_single_trajectory_danger_torch(
    ego_traj: torch.Tensor,
    agent_traj: torch.Tensor,
    config: DangerScoreConfigTorch,
    traj_type: str = "future"
) -> Dict[str, torch.Tensor]:
    """
    计算单段轨迹（历史或未来）的危险指数 - PyTorch版本

    Args:
        ego_traj: [batch, T, D] Ego轨迹
        agent_traj: [batch, T, D] Agent轨迹
        config: 危险评分配置
        traj_type: "history" 或 "future"

    Returns:
        该段轨迹的危险指数字典
    """
    batch_size = ego_traj.size(0)
    T = ego_traj.size(1)
    device = ego_traj.device

    # TTC分析
    # 计算每个时间步的TTC [batch, T]
    ttc_scores = torch.zeros(batch_size, T, device=device)
    lon_ttc_seq = torch.zeros(batch_size, T, device=device)
    lat_ttc_seq = torch.zeros(batch_size, T, device=device)

    for t in range(T):
        ttc_result = compute_ttc_at_timestep_torch(ego_traj[:, t], agent_traj[:, t], config)
        lon_ttc_seq[:, t] = ttc_result['lon_ttc']
        lat_ttc_seq[:, t] = ttc_result['lat_ttc']
        ttc_scores[:, t] = ttc_danger_score_torch(
            ttc_result['lon_ttc'],
            ttc_result['lat_ttc'],
            config
        )

    # 平均TTC危险评分 [batch]
    ttc_score = ttc_scores.mean(dim=1)

    # 最小TTC [batch]
    min_lon_ttc = lon_ttc_seq.min(dim=1)[0]
    min_lat_ttc = lat_ttc_seq.min(dim=1)[0]

    # 碰撞检测
    has_collision, min_distance, collision_t = collision_detection_torch(ego_traj, agent_traj, config)

    # 碰撞评分 [batch]
    collision_score = torch.where(
        has_collision,
        config.weight_collision * 10.0,
        torch.zeros(batch_size, device=device)
    )

    # 近距离评分 [batch]
    near_miss_score = torch.where(
        (min_distance > config.collision_distance) & (min_distance <= config.near_miss_distance) & ~has_collision,
        config.weight_near_miss * (config.near_miss_distance - min_distance) / \
        (config.near_miss_distance - config.collision_distance),
        torch.zeros(batch_size, device=device)
    )

    # 距离评分 [batch]
    distance_score = distance_danger_score_torch(min_distance, config)

    # 速度评分 [batch]
    velocity_score = relative_velocity_danger_score_torch(ego_traj, agent_traj, config)

    # 计算总分 [batch]
    # collision_score / near_miss_score 仅保留在返回 dict 供监控，不计入 total（distance_score 已覆盖这两段）
    total_score = ttc_score + distance_score + velocity_score

    return {
        'traj_type': traj_type,
        'total_score': total_score,
        'ttc_score': ttc_score,
        'distance_score': distance_score,
        'collision_score': collision_score,
        'velocity_score': velocity_score,
        'min_lon_ttc': min_lon_ttc,
        'min_lat_ttc': min_lat_ttc,
        'min_distance': min_distance,
        'has_collision': has_collision,
        'collision_timestep': collision_t
    }


def compute_danger_score_torch(
    ego_history: torch.Tensor,
    ego_future: torch.Tensor,
    agent_history: torch.Tensor,
    agent_future: torch.Tensor,
    config: DangerScoreConfigTorch = None,
    dt: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    计算ego-agent轨迹对的综合性危险评分 - PyTorch版本

    用于Diffusion训练中的danger loss计算。

    Args:
        ego_history: [batch, T_hist, D] Ego历史轨迹
        ego_future: [batch, T_future, D] Ego未来轨迹
        agent_history: [batch, T_hist, D] Agent历史轨迹
        agent_future: [batch, T_future, D] Agent未来轨迹
        config: 危险评分配置
        dt: 时间步长（秒）

    Returns:
        字典包含：
        - inherent_danger: 历史轨迹的危险指数（场景固有风险）[batch]
        - outcome_danger: 未来轨迹的危险指数（结果风险）[batch]
        - avoidance_quality: 避让质量 [batch]
        - total_score: 总危险评分 [batch]
    """
    if config is None:
        config = get_default_config()

    # 1. 计算历史轨迹的危险指数（场景固有风险）
    inherent_result = compute_single_trajectory_danger_torch(
        ego_history, agent_history, config, traj_type="history"
    )
    inherent_danger = inherent_result['total_score']

    # 2. 计算未来轨迹的危险指数（结果风险）
    outcome_result = compute_single_trajectory_danger_torch(
        ego_future, agent_future, config, traj_type="future"
    )
    outcome_danger = outcome_result['total_score']

    # 3. 计算避让质量
    # 避让质量 = 1 - outcome_danger / inherent_danger（如果历史有风险）
    # 使用clamp防止除零
    inherent_safe = inherent_danger > 0
    avoidance_quality = torch.where(
        inherent_safe,
        1.0 - torch.clamp(outcome_danger / (inherent_danger + 1e-6), 0, 1),
        torch.where(
            outcome_danger > 0,
            -outcome_danger,  # 负值表示制造了风险
            torch.ones_like(outcome_danger)  # 无风险历史且无风险未来
        )
    )

    # 总危险评分
    total_score = inherent_danger + outcome_danger

    return {
        'inherent_danger': inherent_danger,
        'outcome_danger': outcome_danger,
        'avoidance_quality': avoidance_quality,
        'total_score': total_score,
        'inherent_min_distance': inherent_result['min_distance'],
        'outcome_min_distance': outcome_result['min_distance'],
        'outcome_has_collision': outcome_result['has_collision'],
        'outcome_min_ttc': outcome_result['min_lon_ttc']
    }


# ============================================================
# SVO辅助函数 - PyTorch版本
# ============================================================

def svo_angle_to_weights_torch(svo_angle_deg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 SVO 角度反向计算单位圆上的权重 - PyTorch版本

    Args:
        svo_angle_deg: [batch] SVO角度（度）

    Returns:
        (x_weight, y_weight): 单位圆上的权重
    """
    svo_angle_rad = torch.deg2rad(svo_angle_deg)
    x_weight = torch.cos(svo_angle_rad)
    y_weight = torch.sin(svo_angle_rad)
    return x_weight, y_weight


def compute_svo_reward_torch(
    self_payoff: torch.Tensor,
    other_payoff: torch.Tensor,
    svo_angle_deg: torch.Tensor
) -> torch.Tensor:
    """
    基于 SVO 角度计算 reward - PyTorch版本

    Args:
        self_payoff: [batch] 自己的利益
        other_payoff: [batch] 他人的利益
        svo_angle_deg: [batch] SVO角度（度）

    Returns:
        [batch] 加权后的reward
    """
    x, y = svo_angle_to_weights_torch(svo_angle_deg)
    reward = x * self_payoff + y * other_payoff
    return reward


# ============================================================
# Diffusion训练专用函数
# ============================================================

def recover_trajectory_from_increment(
    increments: torch.Tensor,
    last_pos: torch.Tensor,
    last_heading: torch.Tensor,
    last_vx: torch.Tensor,
    last_vy: torch.Tensor,
    dt: float = 0.1
) -> torch.Tensor:
    """
    从轨迹增量恢复完整轨迹 - 用于Diffusion输出后计算danger

    Args:
        increments: [batch, 120] 轨迹增量 (60步 × 2 dx, dy)
        last_pos: [batch, 2] 历史最后帧位置 (x, y)
        last_heading: [batch] 历史最后帧heading
        last_vx: [batch] 历史最后帧vx
        last_vy: [batch] 历史最后帧vy
        dt: 时间步长

    Returns:
        [batch, 60, 5] 恢复的轨迹 [x, y, heading, vx, vy]
    """
    batch_size = increments.size(0)
    device = increments.device

    # 重塑为 [batch, 60, 2]
    dx_dy = increments.view(batch_size, 60, 2)

    # 初始化轨迹
    trajectory = torch.zeros(batch_size, 60, 5, device=device)

    # 位置累积
    positions = torch.zeros(batch_size, 60, 2, device=device)
    positions[:, 0] = last_pos + dx_dy[:, 0]  # 第一步
    for t in range(1, 60):
        positions[:, t] = positions[:, t-1] + dx_dy[:, t]

    trajectory[:, :, 0:2] = positions

    # Heading估计（t=0 继承历史 heading，避免 atan2(0,0) 导致的方向突变）
    for t in range(60):
        if t > 0:
            delta_x = positions[:, t, 0] - positions[:, t-1, 0]
            delta_y = positions[:, t, 1] - positions[:, t-1, 1]
            trajectory[:, t, 2] = torch.atan2(delta_y, delta_x)
        else:
            # inc[0] may be (0,0) for ego; inherit last_heading to preserve continuity
            trajectory[:, 0, 2] = last_heading

    # 速度估计（t=0 继承历史 vx/vy，避免 dx=(0,0) 产生 speed=0 的假毛刺）
    for t in range(60):
        if t > 0:
            vx = (positions[:, t, 0] - positions[:, t-1, 0]) / dt
            vy = (positions[:, t, 1] - positions[:, t-1, 1]) / dt
            trajectory[:, t, 3] = vx
            trajectory[:, t, 4] = vy
        else:
            # Use history velocity at t=0 (inc[0] is always 0 for ego in new pkls)
            trajectory[:, 0, 3] = last_vx
            trajectory[:, 0, 4] = last_vy

    return trajectory


def kinematic_decode_torch(
    kin_params: torch.Tensor,
    v0: torch.Tensor,
    theta0: torch.Tensor,
    last_pos: torch.Tensor,
    dt: float = 0.1,
    max_dv: float = 2.0,
    max_dtheta: float = 0.25,
) -> torch.Tensor:
    """
    运动学解码：(Δv, Δθ) 参数 → [B, 60, 5] 轨迹

    Args:
        kin_params:  [B, 120] 交错排列 (Δv_0,Δθ_0, Δv_1,Δθ_1, ...)
        v0:          [B] 初始速度 (m/s)
        theta0:      [B] 初始航向角 (rad)
        last_pos:    [B, 2] 初始位置 (x, y)
        dt:          时间步长 (s)
        max_dv:      速度变化量上限 (m/s/步)，对应最大加速度 max_dv/dt
        max_dtheta:  航向角变化量上限 (rad/步)
    Returns:
        [B, 60, 5] 轨迹 [x, y, heading, vx, vy]
    """
    B = kin_params.size(0)
    device = kin_params.device
    params = kin_params.view(B, 60, 2)
    delta_v     = params[:, :, 0].clamp(-max_dv,     max_dv)
    delta_theta = params[:, :, 1].clamp(-max_dtheta, max_dtheta)

    speeds   = (v0.unsqueeze(1) + torch.cumsum(delta_v,     dim=1)).clamp(0, 40)
    headings =  theta0.unsqueeze(1) + torch.cumsum(delta_theta, dim=1)

    dx = speeds * torch.cos(headings) * dt
    dy = speeds * torch.sin(headings) * dt
    positions = last_pos.unsqueeze(1) + torch.cumsum(
        torch.stack([dx, dy], dim=-1), dim=1)

    vx = speeds * torch.cos(headings)
    vy = speeds * torch.sin(headings)
    traj = torch.cat([
        positions,
        headings.unsqueeze(-1),
        vx.unsqueeze(-1),
        vy.unsqueeze(-1),
    ], dim=-1)
    return traj


def kinematic_cumulative_decode_torch(
    cum_params: torch.Tensor,
    last_pos: torch.Tensor,
    dt: float = 0.1,
    v_scale: float = 20.0,
    hard_clamp: bool = False,
) -> torch.Tensor:
    """Decode cumulative kinematic (v_norm, θ_abs) parameters → [B, 60, 5] trajectory.

    Inverse of gt_to_cumulative_kinematic_np.  No v0/θ0 needed because each frame
    already stores absolute speed and heading.

    Args:
        cum_params: [B, 120] interleaved (v_norm_0, θ_0, v_norm_1, θ_1, ...)
        last_pos:   [B, 2] position at end of history (x, y)
        dt:         timestep (s)
        v_scale:    must match training (default 20.0; v_abs = v_norm * v_scale)
        hard_clamp: if True, apply hard clamp(0, 40) on speed (inference); if False,
                    use F.relu for a soft non-negative lower bound that preserves
                    gradients at all v_norm values (training).
    Returns:
        [B, 60, 5] trajectory [x, y, heading, vx, vy]
    """
    B = cum_params.size(0)
    device = cum_params.device
    params = cum_params.view(B, 60, 2)
    raw_speed = params[:, :, 0] * v_scale
    if hard_clamp:
        speeds = raw_speed.clamp(0.0, 40.0)
    else:
        # F.relu is differentiable everywhere (gradient = v_scale for v_norm > 0,
        # = 0 for v_norm < 0).  Avoids killing gradients at the hard upper bound
        # (which happens when high-noise x_0_pred amplifies v_norm above 2.0).
        speeds = F.relu(raw_speed)
    headings = params[:, :, 1]

    dx = speeds * torch.cos(headings) * dt
    dy = speeds * torch.sin(headings) * dt
    positions = last_pos.unsqueeze(1) + torch.cumsum(
        torch.stack([dx, dy], dim=-1), dim=1
    )
    vx = speeds * torch.cos(headings)
    vy = speeds * torch.sin(headings)
    return torch.cat(
        [positions, headings.unsqueeze(-1), vx.unsqueeze(-1), vy.unsqueeze(-1)],
        dim=-1,
    )


def xy_displacement_decode_torch(
    xy_params: torch.Tensor,
    last_pos: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """Decode (Δx, Δy) displacement [B, 120] → [B, 60, 5] trajectory.

    Inverse of gt_to_xy_displacement_np.

    Args:
        xy_params: [B, 120] interleaved [Δx₀, Δy₀, Δx₁, Δy₁, ...]
        last_pos:  [B, 2] position at end of history
        dt:        timestep (s)

    Returns:
        [B, 60, 5] trajectory [x, y, heading, vx, vy]
    """
    B = xy_params.size(0)
    params = xy_params.view(B, 60, 2)                             # [B, 60, 2]
    positions = last_pos.unsqueeze(1) + torch.cumsum(params, dim=1)  # [B, 60, 2]
    vx = params[:, :, 0] / dt                                    # [B, 60]
    vy = params[:, :, 1] / dt                                    # [B, 60]
    headings = torch.atan2(vy, vx)                               # [B, 60]
    return torch.cat(
        [positions, headings.unsqueeze(-1), vx.unsqueeze(-1), vy.unsqueeze(-1)],
        dim=-1,
    )


def bicycle_model_decode_torch(
    action_params: torch.Tensor,
    init_state: torch.Tensor,
    wheelbase: float = 2.8,
    dt: float = 0.1,
    a_max: float = 3.0,
    delta_max: float = 0.5,
) -> torch.Tensor:
    """Decode pre-activation bicycle actions [B,120] → [B,60,5] trajectory.

    action_params: [B,120] interleaved [raw_a₀, raw_δ₀, raw_a₁, raw_δ₁, ...]
    init_state:    [B,4]  [x0, y0, θ0, v0]

    tanh applied internally — physical constraints guaranteed:
      |a| ≤ a_max (3 m/s²), |δ| ≤ delta_max (0.5 rad), v ∈ [0, 40].
    No additional validity loss needed.
    """
    B = action_params.size(0)
    params = action_params.view(B, 60, 2)
    a     = a_max     * torch.tanh(params[:, :, 0])   # [B, 60]
    delta = delta_max * torch.tanh(params[:, :, 1])   # [B, 60]

    x     = init_state[:, 0:1]                         # [B, 1]
    y     = init_state[:, 1:2]
    theta = init_state[:, 2:3]
    v     = init_state[:, 3:4].clamp(0.0, 40.0)

    pos_list, hdg_list, vx_list, vy_list = [], [], [], []
    for t in range(60):
        v     = (v + a[:, t:t+1] * dt).clamp(0.0, 40.0)
        theta = theta + v * torch.tan(delta[:, t:t+1]) / wheelbase * dt
        x     = x + v * torch.cos(theta) * dt
        y     = y + v * torch.sin(theta) * dt
        pos_list.append(torch.cat([x, y], dim=1))
        hdg_list.append(theta)
        vx_list.append(v * torch.cos(theta))
        vy_list.append(v * torch.sin(theta))

    pos_t = torch.stack(pos_list, dim=1)   # [B, 60, 2]
    hdg_t = torch.stack(hdg_list, dim=1)   # [B, 60, 1]
    vx_t  = torch.stack(vx_list,  dim=1)   # [B, 60, 1]
    vy_t  = torch.stack(vy_list,  dim=1)   # [B, 60, 1]
    return torch.cat([pos_t, hdg_t, vx_t, vy_t], dim=-1)   # [B, 60, 5]


# ============================================================
# 车道偏离惩罚（lane deviation penalty）
# ============================================================

def lane_deviation_score_torch(
    traj_xy: torch.Tensor,
    lane_segs: torch.Tensor,
    half_width: float = 1.75,
) -> torch.Tensor:
    """Per-sample minimum distance from trajectory to nearest lane segment.

    Computes the mean excess distance (> half_width) over all trajectory timesteps,
    penalizing trajectories that leave the drivable lane region.

    Args:
        traj_xy:   [B, T, 2] trajectory (x, y) in ego-centric local frame
        lane_segs: [B, L, 4] padded lane segments (x_start, y_start, x_end, y_end)
                   zero-padded rows (all zeros) are treated as absent
        half_width: lane half-width threshold (m); deviation > this is penalized

    Returns:
        [B] mean per-step lane deviation loss (0 if within lane)
    """
    B, T, _ = traj_xy.shape
    L = lane_segs.shape[1]

    # Valid lane mask: a segment is real if its start != end (not all-zero padding)
    # shape [B, L]
    seg_len_sq = (lane_segs[:, :, 2] - lane_segs[:, :, 0]) ** 2 + \
                 (lane_segs[:, :, 3] - lane_segs[:, :, 1]) ** 2
    valid_mask = seg_len_sq > 1e-6   # [B, L]

    # If no valid lanes for a sample, return zero loss for that sample
    has_lane = valid_mask.any(dim=1)  # [B]

    # Expand for broadcasting: [B, T, L, 2]
    P  = traj_xy.unsqueeze(2).expand(B, T, L, 2)         # query points
    A  = lane_segs[:, :, :2].unsqueeze(1).expand(B, T, L, 2)   # segment starts
    B_ = lane_segs[:, :, 2:].unsqueeze(1).expand(B, T, L, 2)   # segment ends

    AB  = B_ - A                    # [B, T, L, 2]  segment vectors
    AP  = P  - A                    # [B, T, L, 2]  point - start
    ab2 = (AB * AB).sum(-1) + 1e-8  # [B, T, L]    |AB|^2

    # Projection parameter t ∈ [0, 1] → closest point on segment
    t = (AP * AB).sum(-1) / ab2     # [B, T, L]
    t = t.clamp(0.0, 1.0)

    # Closest point Q = A + t * AB
    Q    = A + t.unsqueeze(-1) * AB   # [B, T, L, 2]
    dist = (P - Q).pow(2).sum(-1).sqrt()  # [B, T, L]

    # Mask out padding segments: set their distance to +inf so min picks a real lane
    inf_mask = (~valid_mask).unsqueeze(1).expand(B, T, L)
    dist = dist.masked_fill(inf_mask, float('inf'))

    # Minimum distance to any lane segment per (batch, timestep)
    min_dist, _ = dist.min(dim=-1)   # [B, T]

    # If no valid lanes exist for a sample, min_dist will be inf — clamp to 0 to avoid NaN
    min_dist = torch.where(has_lane.unsqueeze(1).expand(B, T), min_dist, torch.zeros_like(min_dist))

    # Penalty: deviation beyond half_width, averaged over timesteps
    deviation = F.relu(min_dist - half_width)   # [B, T]
    return deviation.mean(dim=-1)               # [B]


# ============================================================
# Danger超越Loss（核心训练Loss）
# ============================================================

class DangerExceedLoss(nn.Module):
    """
    Danger超越Loss — v3.16：(Δx,Δy) 参数化 + 方向反转惩罚 + Jerk约束

    Loss组成：
    1. recon_loss:              MSE(pred_inc, gt_inc)
    2. danger_exceed_loss:      ReLU(gt_danger - pred_danger + margin)
    3. proximity_loss:          接近度
    4. direction_reversal_loss: 惩罚连续速度方向夹角 > 120°（封堵 zigzag 漏洞）
    5. jerk_loss:               惩罚 |jerk| > jerk_max（封堵高频振荡漏洞）
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        danger_weight: float = 2.0,
        danger_margin: float = 0.5,
        proximity_weight: float = 0.0,
        proximity_threshold: float = 8.0,
        direction_reversal_weight: float = 100.0,
        jerk_weight: float = 1e-5,
        cos_thresh: float = -0.5,
        jerk_max: float = 50.0,
        upper_bound_weight: float = 1.0,
        upper_bound_ratio: float = 2.0,
        lane_weight: float = 0.0,        # weight for lane deviation penalty
        lane_half_width: float = 1.75,   # lane half-width threshold (m)
        config: DangerScoreConfigTorch = None,
        use_kinematic_v6_0: bool = False,  # [v6.0] 六C: decode (Δv,Δθ) before danger computation
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.danger_weight = danger_weight
        self.use_kinematic_v6_0 = use_kinematic_v6_0  # [v6.0]
        self.danger_margin = danger_margin
        self.proximity_weight = proximity_weight
        self.proximity_threshold = proximity_threshold
        self.direction_reversal_weight = direction_reversal_weight
        self.jerk_weight = jerk_weight
        self.cos_thresh = cos_thresh
        self.jerk_max = jerk_max
        self.upper_bound_weight = upper_bound_weight
        self.upper_bound_ratio = upper_bound_ratio
        self.lane_weight = lane_weight
        self.lane_half_width = lane_half_width
        self.config = config if config is not None else get_default_config()

    def forward(
        self,
        pred_increments: torch.Tensor,
        gt_increments: torch.Tensor,
        ego_history: torch.Tensor,
        agent_history: torch.Tensor,
        ego_last_pos: torch.Tensor,
        ego_last_heading: torch.Tensor,
        ego_last_vx: torch.Tensor,
        ego_last_vy: torch.Tensor,
        gt_danger: torch.Tensor,
        svo_angle: Optional[torch.Tensor] = None,
        danger_target: Optional[torch.Tensor] = None,
        lane_polylines: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算Diffusion训练loss（(Δx,Δy) 参数化，v3.16）

        Args:
            pred_increments: [B, 240] 预测联合增量 (ego 120 + agent 120)
            gt_increments:   [B, 240] GT联合增量
            ego_history:     [B, T_hist, 5]
            agent_history:   [B, T_hist, 5]
            ego_last_pos:    [B, 2]
            ego_last_heading:[B]
            ego_last_vx:     [B]
            ego_last_vy:     [B]
            gt_danger:       [B]
            svo_angle:       [B] (可选)
            danger_target:   [B] (可选)
            lane_polylines:  [B, L, 4] padded lane segments (可选)
        """
        B = pred_increments.size(0)
        dt = 0.1

        # ── 1. 拆分 ego / agent 增量 ─────────────────────────────────────
        pred_ego_inc   = pred_increments[:, :120]    # [B, 120]
        pred_agent_inc = pred_increments[:, 120:]    # [B, 120]
        gt_ego_inc     = gt_increments[:, :120]
        gt_agent_inc   = gt_increments[:, 120:]

        # ── 2. 重建Loss（DDPM 标准 MSE）────────────────────────────────
        recon_loss = F.mse_loss(pred_ego_inc, gt_ego_inc) + \
                     F.mse_loss(pred_agent_inc, gt_agent_inc)

        # ── 3. 解码轨迹（与训练监控使用相同 decoder，t=0 继承历史状态）─────
        agent_last_pos_h     = agent_history[:, -1, :2]   # [B, 2]
        agent_last_heading_h = agent_history[:, -1, 2]    # [B]
        agent_last_vx_h      = agent_history[:, -1, 3]    # [B]
        agent_last_vy_h      = agent_history[:, -1, 4]    # [B]

        if self.use_kinematic_v6_0:
            # [v6.0] 六C: decode kinematic (Δv,Δθ) increments → trajectory
            ego_v0    = torch.sqrt(ego_last_vx ** 2 + ego_last_vy ** 2 + 1e-8)
            agent_v0  = torch.sqrt(agent_last_vx_h ** 2 + agent_last_vy_h ** 2 + 1e-8)
            ego_future_pred   = kinematic_decode_torch(
                pred_ego_inc, ego_v0, ego_last_heading, ego_last_pos, dt=dt)
            agent_future_pred = kinematic_decode_torch(
                pred_agent_inc, agent_v0, agent_last_heading_h, agent_last_pos_h, dt=dt)
        else:
            ego_future_pred   = recover_trajectory_from_increment(
                pred_ego_inc, ego_last_pos, ego_last_heading, ego_last_vx, ego_last_vy, dt=dt
            )  # [B, 60, 5]
            agent_future_pred = recover_trajectory_from_increment(
                pred_agent_inc, agent_last_pos_h, agent_last_heading_h,
                agent_last_vx_h, agent_last_vy_h, dt=dt
            )  # [B, 60, 5]

        # ── 4. 危险度计算 ────────────────────────────────────────────────
        pred_danger_result = compute_danger_score_torch(
            ego_history, ego_future_pred,
            agent_history, agent_future_pred,
            self.config
        )
        pred_danger = pred_danger_result['outcome_danger']   # [B]

        # ── 5. Danger 超越Loss（相对比例，归一化到 [0,1]）────────────────
        # danger_target 优先使用传入值，fallback 到 gt_danger + danger_margin
        if danger_target is not None:
            target = danger_target.squeeze(-1).clamp(min=1e-8)   # [B]
        else:
            target = (gt_danger + self.danger_margin).clamp(min=1e-8)

        # 下界：pred_danger 未达到 target 时有 loss，量级归一化到 [0,1]
        danger_exceed_loss = F.relu(1.0 - pred_danger / target).mean()

        # 上界：pred_danger 超过 target * upper_bound_ratio 时惩罚，防止虚高
        upper_bound_loss = F.relu(pred_danger / target - self.upper_bound_ratio).mean()

        # ── 6. 几何趋近Loss ──────────────────────────────────────────────
        ego_xy   = ego_future_pred[:, :, :2]
        agent_xy = agent_future_pred[:, :, :2]
        dist     = torch.sqrt(((ego_xy - agent_xy) ** 2).sum(dim=-1) + 1e-8)
        min_dist = dist.min(dim=-1)[0]
        proximity_loss = F.relu(min_dist - self.proximity_threshold).clamp(max=50.0).mean()

        # ── 7. 方向反转惩罚（防 zigzag）ego + agent 各自计算后取平均 ──────
        # cos_thresh = -0.5 → 允许最大 120° 转弯，180° (zigzag) 直接报警
        def _dir_reversal_loss(inc_flat: torch.Tensor) -> torch.Tensor:
            incr = inc_flat.view(B, 60, 2)
            vx_ = incr[:, :, 0] / dt
            vy_ = incr[:, :, 1] / dt
            sp0_ = torch.sqrt(vx_[:, :-1]**2 + vy_[:, :-1]**2 + 1e-8)
            sp1_ = torch.sqrt(vx_[:, 1:]**2  + vy_[:, 1:]**2  + 1e-8)
            cos_ = (vx_[:, :-1]*vx_[:, 1:] + vy_[:, :-1]*vy_[:, 1:]) / (sp0_*sp1_ + 1e-8)
            return F.relu(self.cos_thresh - cos_).pow(2).mean()

        direction_reversal_loss = (_dir_reversal_loss(pred_ego_inc) +
                                   _dir_reversal_loss(pred_agent_inc)) / 2

        # ── 8. Jerk 约束（防高频振荡）ego + agent 各自计算后取平均 ──────
        def _jerk_loss(inc_flat: torch.Tensor) -> torch.Tensor:
            incr = inc_flat.view(B, 60, 2)
            vx_ = incr[:, :, 0] / dt
            vy_ = incr[:, :, 1] / dt
            ax_ = (vx_[:, 1:] - vx_[:, :-1]) / dt   # [B, 59] m/s²
            ay_ = (vy_[:, 1:] - vy_[:, :-1]) / dt
            jx_ = (ax_[:, 1:] - ax_[:, :-1]) / dt   # [B, 58] m/s³
            jy_ = (ay_[:, 1:] - ay_[:, :-1]) / dt
            return F.relu(torch.sqrt(jx_**2 + jy_**2 + 1e-8) - self.jerk_max).pow(2).mean()

        jerk_loss = (_jerk_loss(pred_ego_inc) + _jerk_loss(pred_agent_inc)) / 2

        # ── 9. 车道偏离惩罚 ──────────────────────────────────────────────
        if self.lane_weight > 0.0 and lane_polylines is not None:
            ego_xy = ego_future_pred[:, :, :2]    # [B, T, 2]
            lane_deviation_loss = lane_deviation_score_torch(
                ego_xy, lane_polylines, self.lane_half_width
            ).mean()  # scalar
        else:
            lane_deviation_loss = torch.tensor(0.0, device=pred_increments.device)

        # ── 10. 总Loss ───────────────────────────────────────────────────
        total_loss = (
            self.recon_weight              * recon_loss +
            self.danger_weight             * danger_exceed_loss +
            self.upper_bound_weight        * upper_bound_loss +
            self.proximity_weight          * proximity_loss +
            self.direction_reversal_weight * direction_reversal_loss +
            self.jerk_weight               * jerk_loss +
            self.lane_weight               * lane_deviation_loss
        )

        if svo_angle is not None:
            svo_consistency_loss = torch.where(
                svo_angle < 0,
                F.relu(pred_danger - gt_danger),
                torch.zeros_like(pred_danger)
            ).mean()
            total_loss = total_loss - 0.1 * svo_consistency_loss

        loss_dict = {
            'total_loss':              total_loss,
            'recon_loss':              recon_loss,
            'danger_exceed_loss':      danger_exceed_loss,
            'upper_bound_loss':        upper_bound_loss,
            'direction_reversal_loss': direction_reversal_loss,
            'jerk_loss':               jerk_loss,
            'proximity_loss':          proximity_loss,
            'lane_deviation_loss':     lane_deviation_loss,
            'min_dist_mean':           min_dist.mean().item(),
            'pred_danger':             pred_danger.mean().item(),
            'gt_danger':               gt_danger.mean().item(),
            'danger_target_mean':      target.mean().item(),
            'danger_margin_met':       (pred_danger >= target).float().mean().item()
        }

        return total_loss, loss_dict


# ============================================================
# 测试函数
# ============================================================

def test_danger_score_torch():
    """测试PyTorch版本的danger score计算"""
    print("Testing danger_score_torch...")

    # 创建测试数据
    batch_size = 4
    T_hist = 20
    T_future = 30
    device = 'cpu'

    # Ego历史：直线行驶
    ego_history = torch.zeros(batch_size, T_hist, 5, device=device)
    for t in range(T_hist):
        ego_history[:, t, 0] = t * 0.5  # x
        ego_history[:, t, 1] = 0.0  # y (保持在车道中心)
        ego_history[:, t, 2] = 0.0  # heading
        ego_history[:, t, 3] = 5.0  # vx
        ego_history[:, t, 4] = 0.0  # vy

    # Agent历史：在ego前方
    agent_history = torch.zeros(batch_size, T_hist, 5, device=device)
    for t in range(T_hist):
        agent_history[:, t, 0] = t * 0.5 + 10.0  # x (前方10米)
        agent_history[:, t, 1] = 0.0  # y (同一车道)
        agent_history[:, t, 2] = 0.0  # heading
        agent_history[:, t, 3] = 3.0  # vx (比ego慢)
        agent_history[:, t, 4] = 0.0  # vy

    # Ego未来：继续行驶
    ego_future = torch.zeros(batch_size, T_future, 5, device=device)
    for t in range(T_future):
        ego_future[:, t, 0] = (T_hist + t) * 0.5  # x
        ego_future[:, t, 1] = 0.0  # y
        ego_future[:, t, 2] = 0.0  # heading
        ego_future[:, t, 3] = 5.0  # vx
        ego_future[:, t, 4] = 0.0  # vy

    # Agent未来：减速（危险）
    agent_future = torch.zeros(batch_size, T_future, 5, device=device)
    for t in range(T_future):
        agent_future[:, t, 0] = (T_hist + t) * 0.3 + 10.0  # x (减速)
        agent_future[:, t, 1] = 0.0  # y
        agent_future[:, t, 2] = 0.0  # heading
        agent_future[:, t, 3] = 3.0 - t * 0.05  # vx (逐渐减速)
        agent_future[:, t, 4] = 0.0  # vy

    # 计算danger
    config = get_default_config()
    result = compute_danger_score_torch(ego_history, ego_future, agent_history, agent_future, config)

    print(f"Inherent danger: {result['inherent_danger']}")
    print(f"Outcome danger: {result['outcome_danger']}")
    print(f"Total score: {result['total_score']}")
    print(f"Min TTC: {result['outcome_min_ttc']}")
    print(f"Min distance: {result['outcome_min_distance']}")

    print("Test passed!")


if __name__ == "__main__":
    test_danger_score_torch()