"""
IDM模拟模块 - 边界车辆伪交互

本模块实现了边界车辆的IDM (Intelligent Driver Model) 模拟，
用于在diffusion生成新的ego和key_agent轨迹后，
让周围边界车辆做出合理的响应。

IDM模型：
- 跟驰行为：保持安全距离，平滑加减速
- 变道响应：对相邻车道车辆的位置变化做出反应
- 速度调节：根据前车速度调整自身速度

主要功能：
1. 确定每个边界车辆的跟随对象
2. 计算IDM加速度
3. 更新边界车辆轨迹
4. 支持与diffusion生成轨迹的交互
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ============================================================
# IDM配置参数
# ============================================================

@dataclass
class IDMConfig:
    """IDM模型配置参数"""

    # IDM核心参数
    v_desired: float = 10.0           # 期望速度 (m/s)
    delta: float = 4.0                # 加速度指数
    T: float = 1.5                    # 安全时间头距 (s)
    s0: float = 2.0                   # 最小安全距离 (m)
    a_max: float = 2.0                # 最大加速度 (m/s²)
    b_comfort: float = 3.0            # 舒适减速度 (m/s²)

    # 车辆尺寸
    vehicle_length: float = 4.0       # 车辆长度 (m)
    vehicle_width: float = 2.0        # 车辆宽度 (m)

    # 跟驰判定阈值
    front_detection_range: float = 50.0   # 前方检测范围 (m)
    lateral_threshold: float = 3.5        # 横向阈值，判断是否同车道 (m)

    # 仿真参数
    dt: float = 0.1                   # 时间步长 (s)


# ============================================================
# 跟随对象确定
# ============================================================

@dataclass
class FollowTarget:
    """跟随目标信息"""
    target_id: int                    # 目标ID (-1表示无目标)
    target_type: str                  # 'ego', 'key_agent', 'boundary', 'none'
    distance: float                   # 相对距离 (m)
    relative_velocity: float          # 相对速度 (m/s)
    is_same_lane: bool               # 是否同车道


def determine_follow_target(
    agent_id: int,
    agent_state: np.ndarray,
    ego_state: np.ndarray,
    key_agent_state: np.ndarray,
    boundary_agents: List[Dict],
    config: IDMConfig
) -> FollowTarget:
    """
    确定边界车辆的跟随目标

    优先级：
    1. 同车道前方最近的车辆
    2. 如果ego或key_agent在前方，优先跟随
    3. 如果没有前车，则自由行驶

    Args:
        agent_id: 边界车辆ID
        agent_state: 边界车辆状态 [x, y, vx, vy]
        ego_state: Ego车辆状态
        key_agent_state: Key agent状态
        boundary_agents: 其他边界车辆列表
        config: IDM配置

    Returns:
        FollowTarget对象
    """
    agent_x, agent_y = agent_state[0], agent_state[1]
    agent_vx = agent_state[2] if len(agent_state) > 2 else 0.0

    candidates = []

    # 1. 检查ego是否在前方
    ego_x, ego_y = ego_state[0], ego_state[1]
    rel_x_ego = ego_x - agent_x
    rel_y_ego = ego_y - agent_y

    if 0 < rel_x_ego < config.front_detection_range:
        # ego在前方
        is_same_lane = abs(rel_y_ego) < config.lateral_threshold
        if is_same_lane:
            ego_vx = ego_state[2] if len(ego_state) > 2 else 0.0
            candidates.append({
                'target_id': 0,
                'target_type': 'ego',
                'distance': rel_x_ego,
                'relative_velocity': ego_vx - agent_vx,
                'is_same_lane': True,
                'priority': 1
            })

    # 2. 检查key_agent是否在前方
    key_x, key_y = key_agent_state[0], key_agent_state[1]
    rel_x_key = key_x - agent_x
    rel_y_key = key_y - agent_y

    if 0 < rel_x_key < config.front_detection_range:
        is_same_lane = abs(rel_y_key) < config.lateral_threshold
        if is_same_lane:
            key_vx = key_agent_state[2] if len(key_agent_state) > 2 else 0.0
            candidates.append({
                'target_id': -1,  # key agent的特殊ID
                'target_type': 'key_agent',
                'distance': rel_x_key,
                'relative_velocity': key_vx - agent_vx,
                'is_same_lane': True,
                'priority': 1
            })

    # 3. 检查其他边界车辆
    for other in boundary_agents:
        other_id = other['agent_id']
        if other_id == agent_id:
            continue

        other_state = other['agent_data'][-1]  # 使用最后一帧状态
        other_x, other_y = other_state[0], other_state[1]
        rel_x_other = other_x - agent_x
        rel_y_other = other_y - agent_y

        if 0 < rel_x_other < config.front_detection_range:
            is_same_lane = abs(rel_y_other) < config.lateral_threshold
            if is_same_lane:
                other_vx = other_state[2] if len(other_state) > 2 else 0.0
                candidates.append({
                    'target_id': other_id,
                    'target_type': 'boundary',
                    'distance': rel_x_other,
                    'relative_velocity': other_vx - agent_vx,
                    'is_same_lane': True,
                    'priority': 2
                })

    # 选择最近的同车道前车
    if candidates:
        # 按距离排序，选择最近的
        candidates.sort(key=lambda x: x['distance'])
        best = candidates[0]
        return FollowTarget(
            target_id=best['target_id'],
            target_type=best['target_type'],
            distance=best['distance'],
            relative_velocity=best['relative_velocity'],
            is_same_lane=best['is_same_lane']
        )

    # 无前车，自由行驶
    return FollowTarget(
        target_id=-1,
        target_type='none',
        distance=np.inf,
        relative_velocity=0.0,
        is_same_lane=False
    )


# ============================================================
# IDM加速度计算
# ============================================================

def compute_idm_acceleration(
    current_velocity: float,
    follow_target: FollowTarget,
    config: IDMConfig
) -> float:
    """
    计算IDM加速度

    IDM公式：
    a = a_max * [1 - (v/v_desired)^delta - (s*(v, Δv)/s)^2]

    其中：
    s*(v, Δv) = s0 + v*T + v*Δv/(2*sqrt(a_max*b))

    Args:
        current_velocity: 当前速度 (m/s)
        follow_target: 跟随目标信息
        config: IDM配置

    Returns:
        加速度 (m/s²)
    """
    # 自由行驶（无前车）
    if follow_target.target_type == 'none':
        # 逐渐加速到期望速度
        v_ratio = current_velocity / config.v_desired
        free_road_term = 1 - (v_ratio ** config.delta)

        acceleration = config.a_max * free_road_term
        return np.clip(acceleration, -config.b_comfort, config.a_max)

    # 有前车
    s = follow_target.distance - config.vehicle_length  # 净距离
    if s < 0:
        s = 0.1  # 避免除零

    delta_v = follow_target.relative_velocity  # 相对速度

    # 期望间距
    # s* = s0 + v*T + v*Δv/(2*sqrt(a_max*b))
    interaction_term = current_velocity * delta_v / (2 * np.sqrt(config.a_max * config.b_comfort))
    s_desired = config.s0 + current_velocity * config.T + max(0, interaction_term)

    # IDM加速度
    v_ratio = current_velocity / config.v_desired
    s_ratio = s_desired / s

    free_road_term = 1 - (v_ratio ** config.delta)
    interaction_term = s_ratio ** 2

    acceleration = config.a_max * (free_road_term - interaction_term)

    # 限制加速度范围
    return np.clip(acceleration, -config.b_comfort * 1.5, config.a_max)


def compute_lateral_adjustment(
    agent_state: np.ndarray,
    ego_state: np.ndarray,
    key_agent_state: np.ndarray,
    config: IDMConfig
) -> float:
    """
    计算横向调整（变道或避让）

    简化逻辑：
    - 如果前方有车辆且距离较近，考虑变道
    - 保持车道中心

    Args:
        agent_state: 边界车辆状态
        ego_state: Ego状态
        key_agent_state: Key agent状态
        config: IDM配置

    Returns:
        横向加速度 (m/s²)
    """
    agent_x, agent_y = agent_state[0], agent_state[1]
    ego_x, ego_y = ego_state[0], ego_state[1]
    key_x, key_y = key_agent_state[0], key_agent_state[1]

    lateral_accel = 0.0

    # 检查与ego的横向距离
    rel_y_ego = ego_y - agent_y
    rel_x_ego = ego_x - agent_x

    # 如果ego在侧前方且距离较近
    if 0 < rel_x_ego < 20 and abs(rel_y_ego) < config.lateral_threshold * 2:
        # 横向避让
        if rel_y_ego > 0:
            lateral_accel = -0.5  # 向右避让
        else:
            lateral_accel = 0.5   # 向左避让

    # 检查与key_agent的横向距离
    rel_y_key = key_y - agent_y
    rel_x_key = key_x - agent_x

    if 0 < rel_x_key < 20 and abs(rel_y_key) < config.lateral_threshold * 2:
        if rel_y_key > 0:
            lateral_accel = -0.5
        else:
            lateral_accel = 0.5

    return np.clip(lateral_accel, -config.a_max * 0.3, config.a_max * 0.3)


# ============================================================
# 轨迹更新
# ============================================================

def update_agent_state(
    agent_state: np.ndarray,
    acceleration: float,
    lateral_accel: float,
    dt: float,
    config: IDMConfig
) -> np.ndarray:
    """
    更新车辆状态

    使用运动学模型：
    v_new = v + a * dt
    x_new = x + v_new * dt  (使用更新后的速度)

    Args:
        agent_state: 当前状态 [x, y, vx, vy] 或 [x, y, heading, vx, vy]
        acceleration: 纵向加速度
        lateral_accel: 横向加速度
        dt: 时间步长
        config: IDM配置

    Returns:
        新状态
    """
    new_state = agent_state.copy()

    # 当前速度
    vx = agent_state[2] if len(agent_state) > 2 else 0.0
    vy = agent_state[3] if len(agent_state) > 3 else 0.0

    # 更新速度
    vx_new = vx + acceleration * dt
    vx_new = np.clip(vx_new, 0, config.v_desired * 1.5)

    vy_new = vy + lateral_accel * dt
    vy_new = np.clip(vy_new, -3.0, 3.0)

    # 更新位置（使用更新后的速度）
    new_state[0] = agent_state[0] + vx_new * dt  # x
    new_state[1] = agent_state[1] + vy_new * dt  # y

    # 更新速度
    if len(new_state) > 2:
        new_state[2] = vx_new
    if len(new_state) > 3:
        new_state[3] = vy_new

    # 更新heading（如果存在）
    if len(new_state) > 4:
        new_state[4] = np.arctan2(vy_new, vx_new)

    return new_state


# ============================================================
# 主模拟函数
# ============================================================

def simulate_boundary_agent(
    agent_data: np.ndarray,
    ego_future: np.ndarray,
    key_agent_future: np.ndarray,
    other_boundary_agents: List[Dict],
    config: IDMConfig,
    agent_id: int = -1
) -> np.ndarray:
    """
    模拟单个边界车辆的未来轨迹

    Args:
        agent_data: 边界车辆历史轨迹 (T_hist, D)
        ego_future: Ego未来轨迹 (T_future, D)
        key_agent_future: Key agent未来轨迹 (T_future, D)
        other_boundary_agents: 其他边界车辆信息
        config: IDM配置
        agent_id: 该边界车辆的ID

    Returns:
        模拟的未来轨迹 (T_future, D)
    """
    num_frames = len(ego_future)
    future_trajectory = np.zeros((num_frames, agent_data.shape[1]))

    # 初始状态
    current_state = agent_data[-1].copy()

    for t in range(num_frames):
        # 获取当前时刻的ego和key_agent状态
        ego_state = ego_future[t]
        key_agent_state = key_agent_future[t]

        # 确定跟随目标
        follow_target = determine_follow_target(
            agent_id, current_state,
            ego_state, key_agent_state,
            other_boundary_agents, config
        )

        # 计算当前速度
        current_vx = current_state[2] if len(current_state) > 2 else 0.0
        current_vy = current_state[3] if len(current_state) > 3 else 0.0
        current_speed = np.sqrt(current_vx**2 + current_vy**2)

        # 计算IDM加速度
        acceleration = compute_idm_acceleration(current_speed, follow_target, config)

        # 计算横向调整
        lateral_accel = compute_lateral_adjustment(
            current_state, ego_state, key_agent_state, config
        )

        # 更新状态
        current_state = update_agent_state(
            current_state, acceleration, lateral_accel, config.dt, config
        )

        # 保存轨迹
        future_trajectory[t] = current_state

    return future_trajectory


def simulate_all_boundary_agents(
    boundary_agents: List[Dict],
    ego_future: np.ndarray,
    key_agent_future: np.ndarray,
    config: IDMConfig
) -> List[Dict]:
    """
    模拟所有边界车辆的未来轨迹

    Args:
        boundary_agents: 边界车辆列表
        ego_future: Ego未来轨迹
        key_agent_future: Key agent未来轨迹
        config: IDM配置

    Returns:
        更新后的边界车辆列表，包含模拟的未来轨迹
    """
    simulated_agents = []

    for i, agent in enumerate(boundary_agents):
        agent_id = agent['agent_id']
        agent_data = agent['agent_data']

        # 其他边界车辆（用于跟驰判断）
        other_agents = [a for j, a in enumerate(boundary_agents) if j != i]

        # 模拟该边界车辆
        simulated_future = simulate_boundary_agent(
            agent_data,
            ego_future,
            key_agent_future,
            other_agents,
            config,
            agent_id
        )

        # 更新agent信息
        simulated_agent = {
            'agent_id': agent_id,
            'agent_history': agent_data,
            'agent_future_simulated': simulated_future,
            'agent_data_full': np.concatenate([agent_data, simulated_future], axis=0),
            'lon_ttc': agent.get('lon_ttc'),
            'lat_ttc': agent.get('lat_ttc'),
            'lon_risk': agent.get('lon_risk'),
            'lat_risk': agent.get('lat_risk')
        }

        simulated_agents.append(simulated_agent)

    return simulated_agents


# ============================================================
# 与Diffusion集成
# ============================================================

def update_boundaries_for_diffusion_trajectory(
    boundary_agents: List[Dict],
    ego_future_generated: np.ndarray,
    key_agent_future_generated: np.ndarray,
    config: IDMConfig = None
) -> List[Dict]:
    """
    根据diffusion生成的轨迹更新边界车辆轨迹

    这是主接口函数，在diffusion生成新的ego和key_agent轨迹后调用。

    Args:
        boundary_agents: 边界车辆信息列表
        ego_future_generated: Diffusion生成的ego未来轨迹
        key_agent_future_generated: Diffusion生成的key_agent未来轨迹
        config: IDM配置（None则使用默认值）

    Returns:
        更新后的边界车辆列表
    """
    if config is None:
        config = IDMConfig()

    return simulate_all_boundary_agents(
        boundary_agents,
        ego_future_generated,
        key_agent_future_generated,
        config
    )


# ============================================================
# 场景级模拟
# ============================================================

def simulate_complete_scenario(
    pair_data: Dict,
    ego_future_generated: np.ndarray = None,
    key_agent_future_generated: np.ndarray = None,
    config: IDMConfig = None
) -> Dict:
    """
    模拟完整场景

    输入：danger_pair数据 + diffusion生成的轨迹（可选）
    输出：完整场景的模拟结果

    Args:
        pair_data: 从danger_pair_*.pkl加载的数据
        ego_future_generated: Diffusion生成的ego轨迹（None则使用ground truth）
        key_agent_future_generated: Diffusion生成的key_agent轨迹
        config: IDM配置

    Returns:
        完整场景模拟结果
    """
    if config is None:
        config = IDMConfig()

    # 提取数据
    ego_history = pair_data['ego_history']
    key_agent_history = pair_data['key_agent_history']
    boundary_agents = pair_data['boundary_agents']

    # 使用生成的轨迹或ground truth
    if ego_future_generated is None:
        # 使用ground truth
        ego_future = pair_data.get('ego_future_gt')
        if ego_future is not None:
            ego_future = ego_future.squeeze(0)  # (1, 30, 2) -> (30, 2)
            # 需要补全速度信息
            ego_future = augment_trajectory_with_velocity(ego_future, ego_history)
        else:
            ego_future = extrapolate_trajectory(ego_history, 30)
    else:
        ego_future = ego_future_generated

    if key_agent_future_generated is None:
        # 使用历史轨迹外推
        key_agent_future = extrapolate_trajectory(key_agent_history, 30)
    else:
        key_agent_future = key_agent_future_generated

    # 模拟边界车辆
    simulated_boundaries = simulate_all_boundary_agents(
        boundary_agents, ego_future, key_agent_future, config
    )

    return {
        'ego_history': ego_history,
        'ego_future': ego_future,
        'key_agent_history': key_agent_history,
        'key_agent_future': key_agent_future,
        'boundary_agents': simulated_boundaries,
        'scenario_name': pair_data.get('scenario_name'),
        'config': config
    }


def extrapolate_trajectory(history: np.ndarray, num_future_frames: int) -> np.ndarray:
    """
    基于历史轨迹外推未来轨迹（简单线性外推）

    Args:
        history: 历史轨迹 (T_hist, D)
        num_future_frames: 未来帧数

    Returns:
        外推的未来轨迹 (num_future_frames, D)
    """
    if len(history) < 2:
        return np.zeros((num_future_frames, history.shape[1]))

    # 使用最后几帧的平均速度
    last_n = min(5, len(history))
    vx_mean = np.mean(history[-last_n:, 3]) if history.shape[1] > 3 else 0.0
    vy_mean = np.mean(history[-last_n:, 4]) if history.shape[1] > 4 else 0.0

    # 线性外推
    future = np.zeros((num_future_frames, history.shape[1]))
    future[:, 0] = history[-1, 0] + vx_mean * np.arange(1, num_future_frames + 1) * 0.1
    future[:, 1] = history[-1, 1] + vy_mean * np.arange(1, num_future_frames + 1) * 0.1
    if future.shape[1] > 3:
        future[:, 3] = vx_mean
    if future.shape[1] > 4:
        future[:, 4] = vy_mean

    return future


def augment_trajectory_with_velocity(
    trajectory: np.ndarray,
    history: np.ndarray
) -> np.ndarray:
    """
    为轨迹补充速度信息

    Args:
        trajectory: 位置轨迹 (T, 2) 或 (T, D)
        history: 历史轨迹（用于获取速度）

    Returns:
        完整轨迹 (T, D)
    """
    num_frames = len(trajectory)
    D = max(trajectory.shape[1], 5)
    full_trajectory = np.zeros((num_frames, D))

    # 复制位置
    full_trajectory[:, :2] = trajectory[:, :2]

    # 计算速度
    if num_frames > 1:
        dx = np.diff(trajectory[:, 0], prepend=trajectory[0, 0])
        dy = np.diff(trajectory[:, 1], prepend=trajectory[0, 1])
        full_trajectory[:, 3] = dx / 0.1  # 假设dt=0.1
        full_trajectory[:, 4] = dy / 0.1

    return full_trajectory


# ============================================================
# 可视化与调试
# ============================================================

def print_simulation_report(simulated_agents: List[Dict]):
    """打印模拟报告"""
    print('\n' + '='*60)
    print('IDM模拟报告')
    print('='*60)

    print(f"\n边界车辆数量: {len(simulated_agents)}")

    for agent in simulated_agents:
        print(f"\n--- Agent {agent['agent_id']} ---")

        # 跟随目标
        history = agent['agent_history']
        future = agent['agent_future_simulated']

        print(f"  初始位置: ({history[-1, 0]:.2f}, {history[-1, 1]:.2f})")
        print(f"  初始速度: ({history[-1, 3]:.2f}, {history[-1, 4]:.2f}) m/s")
        print(f"  终点位置: ({future[-1, 0]:.2f}, {future[-1, 1]:.2f})")
        print(f"  终点速度: ({future[-1, 3]:.2f}, {future[-1, 4]:.2f}) m/s")


# ============================================================
# 测试函数
# ============================================================

def test_idm_simulation():
    """测试IDM模拟模块"""
    print("测试IDM模拟模块...")

    config = IDMConfig()

    # 创建测试数据
    # Ego: 前进
    ego_future = np.zeros((30, 5))
    ego_future[:, 0] = np.linspace(0, 30, 30)
    ego_future[:, 3] = 10.0

    # Key agent: 前方，更慢
    key_agent_future = np.zeros((30, 5))
    key_agent_future[:, 0] = np.linspace(20, 40, 30)
    key_agent_future[:, 3] = 6.0

    # 边界车辆: 后方，速度适中
    boundary_agents = [
        {
            'agent_id': 1,
            'agent_data': np.zeros((21, 5)) + np.array([-10, 0, 0, 8, 0])
        },
        {
            'agent_id': 2,
            'agent_data': np.zeros((21, 5)) + np.array([-5, 3.5, 0, 9, 0])  # 左侧车道
        }
    ]

    print("\n=== 测试场景: 边界车辆跟随ego ===")

    simulated = simulate_all_boundary_agents(
        boundary_agents, ego_future, key_agent_future, config
    )

    print_simulation_report(simulated)

    print("\n=== 测试IDM加速度计算 ===")

    # 测试自由行驶
    follow_target = FollowTarget(
        target_id=-1, target_type='none',
        distance=np.inf, relative_velocity=0, is_same_lane=False
    )
    accel = compute_idm_acceleration(8.0, follow_target, config)
    print(f"自由行驶 (v=8m/s): 加速度 = {accel:.2f} m/s²")

    # 测试跟驰
    follow_target = FollowTarget(
        target_id=0, target_type='ego',
        distance=20.0, relative_velocity=-2.0, is_same_lane=True
    )
    accel = compute_idm_acceleration(10.0, follow_target, config)
    print(f"跟驰 (s=20m, Δv=-2m/s): 加速度 = {accel:.2f} m/s²")

    # 测试紧急制动
    follow_target = FollowTarget(
        target_id=0, target_type='ego',
        distance=5.0, relative_velocity=-5.0, is_same_lane=True
    )
    accel = compute_idm_acceleration(12.0, follow_target, config)
    print(f"紧急制动 (s=5m, Δv=-5m/s): 加速度 = {accel:.2f} m/s²")

    print("\n所有测试完成！")


if __name__ == "__main__":
    test_idm_simulation()