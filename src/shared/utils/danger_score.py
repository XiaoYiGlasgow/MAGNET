"""
危险评分模块 - 用于轨迹生成

本模块定义了计算轨迹对（ego车辆和目标agent）危险评分的函数。
危险评分用于引导diffusion模型生成更危险的轨迹。

危险评分组件：
1. TTC危险：纵向和横向的碰撞时间
2. 距离危险：车辆间最小距离
3. 碰撞指示：二值碰撞检测
4. 近距指示：未碰撞但接近
5. 速度危险：高相对速度场景
6. 车道感知：考虑车道关系的危险评估
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


# ============================================================
# 配置参数
# ============================================================

@dataclass
class DangerScoreConfig:
    """危险评分计算配置"""

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
    lane_width: float = 3.5              # 标准车道宽度（米）
    same_lane_threshold: float = 1.75    # 同车道判定阈值（米），约为车道宽度的一半
    adjacent_lane_threshold: float = 5.25  # 相邻车道判定阈值（米），约1.5个车道宽度
    lane_change_vy_threshold: float = 0.5  # 变道横向速度阈值（米/秒）

    # 不同危险组件的权重
    weight_ttc_longitudinal: float = 1.0
    weight_ttc_lateral: float = 0.8
    weight_ttc_same_lane: float = 1.5    # 同车道TTC权重加成
    weight_min_distance: float = 1.5
    weight_collision: float = 10.0
    weight_near_miss: float = 3.0
    weight_relative_velocity: float = 0.5
    weight_acceleration: float = 0.3
    weight_lane_change_conflict: float = 2.0  # 变道冲突权重

    # 物理约束
    max_acceleration: float = 3.0        # 最大加速度（米/秒²）
    max_deceleration: float = 5.0        # 最大减速度（米/秒²）
    max_lateral_accel: float = 2.0       # 最大横向加速度（米/秒²）

    # SVO (Social Value Orientation) 相关参数 - 单位圆累加模型
    # SVO 角度范围
    svo_angle_min: float = -45.0             # 竞争型极限角度
    svo_angle_max: float = 135.0             # 利他型极限角度

    # SVO 权重增量参数（对个人权重 x 和集体权重 y 的影响）
    # 让行：减少个人权重，增加集体权重
    svo_yield_delta_x: float = -0.1          # 让行使个人权重下降
    svo_yield_delta_y: float = +0.15         # 让行使集体权重上升

    # 抢行：增加个人权重，减少集体权重
    svo_aggressive_delta_x: float = +0.15    # 抢行使个人权重上升
    svo_aggressive_delta_y: float = -0.1     # 抢行使集体权重下降

    # 避让：略微减少个人权重，增加集体权重
    svo_avoidance_delta_x: float = -0.08     # 避让使个人权重下降
    svo_avoidance_delta_y: float = +0.12     # 避让使集体权重上升

    # 靠近：略微增加个人权重
    svo_approach_delta_x: float = +0.05      # 靠近使个人权重上升
    svo_approach_delta_y: float = -0.03      # 靠近使集体权重下降

    # 社会力：增加集体权重
    svo_social_force_delta_y: float = +0.2   # 社会力使集体权重上升

    # SVO 行为检测阈值
    svo_decel_threshold: float = 3.0         # 让行减速度阈值 (m/s²)
    svo_accel_threshold: float = 2.0         # 抢行加速度阈值 (m/s²)
    svo_lateral_speed_threshold: float = 0.5 # 避让横向速度阈值 (m/s²)
    svo_risk_distance_threshold: float = 8.0 # 碰撞风险距离阈值 (m)

    # SVO 权重（用于危险评分）
    weight_svo_agent: float = 0.5            # Agent SVO 权重
    weight_svo_pair: float = 0.8             # Pair SVO 权重


# ============================================================
# 车道感知函数
# ============================================================

def get_lane_index(y_position: float, lane_width: float = 3.5) -> int:
    """
    根据Y坐标推断车道索引

    假设ego在Y=0车道（索引为0）
    Y > 0: 左侧车道（索引 > 0）
    Y < 0: 右侧车道（索引 < 0）

    Args:
        y_position: Y坐标（米）
        lane_width: 车道宽度（米）

    Returns:
        车道索引（整数）
    """
    return int(round(y_position / lane_width))


def get_lane_relationship(
    ego_y: float,
    agent_y: float,
    config: DangerScoreConfig
) -> str:
    """
    判断agent相对于ego的车道关系

    Args:
        ego_y: Ego的Y坐标
        agent_y: Agent的Y坐标
        config: 配置参数

    Returns:
        车道关系：'same_lane', 'adjacent_left', 'adjacent_right', 'far_left', 'far_right'
    """
    # 计算相对Y坐标
    rel_y = agent_y - ego_y
    abs_rel_y = abs(rel_y)

    if abs_rel_y < config.same_lane_threshold:
        return 'same_lane'
    elif abs_rel_y < config.adjacent_lane_threshold:
        if rel_y > 0:
            return 'adjacent_left'
        else:
            return 'adjacent_right'
    else:
        if rel_y > 0:
            return 'far_left'
        else:
            return 'far_right'


def detect_lane_change_intent(
    agent_vy: float,
    config: DangerScoreConfig
) -> Tuple[bool, str]:
    """
    检测agent是否有变道意图

    Args:
        agent_vy: Agent的横向速度
        config: 配置参数

    Returns:
        (是否有变道意图, 变道方向)
    """
    abs_vy = abs(agent_vy)

    if abs_vy < config.lane_change_vy_threshold:
        return False, 'none'

    if agent_vy > 0:
        return True, 'left'
    else:
        return True, 'right'


def compute_lane_aware_ttc_danger(
    lon_ttc: float,
    lat_ttc: float,
    lane_relationship: str,
    lane_change_intent: Tuple[bool, str],
    config: DangerScoreConfig
) -> float:
    """
    考虑车道关系的TTC危险评分

    同车道：纵向TTC是主要危险
    相邻车道：只有当有变道意图时才危险
    远处车道：危险较低

    Args:
        lon_ttc: 纵向TTC
        lat_ttc: 横向TTC
        lane_relationship: 车道关系
        lane_change_intent: (是否有变道意图, 方向)
        config: 配置参数

    Returns:
        车道感知的TTC危险评分
    """
    score = 0.0
    has_intent, intent_dir = lane_change_intent

    if lane_relationship == 'same_lane':
        # 同车道：主要关注纵向TTC，权重加成
        if lon_ttc < config.ttc_critical:
            score += config.weight_ttc_same_lane * (config.ttc_critical - lon_ttc) * 2.0
        elif lon_ttc < config.ttc_warning:
            score += config.weight_ttc_same_lane * (config.ttc_warning - lon_ttc)
        elif lon_ttc < config.ttc_max:
            score += config.weight_ttc_same_lane * (config.ttc_max - lon_ttc) * 0.1

    elif lane_relationship in ['adjacent_left', 'adjacent_right']:
        # 相邻车道：检查变道意图
        if has_intent:
            # 有变道意图：增加危险评分
            # 纵向TTC仍然重要，因为变道后可能追尾
            if lon_ttc < config.ttc_critical:
                score += config.weight_ttc_longitudinal * (config.ttc_critical - lon_ttc) * 1.5
            elif lon_ttc < config.ttc_warning:
                score += config.weight_ttc_longitudinal * (config.ttc_warning - lon_ttc) * 0.8

            # 横向TTC对于变道冲突更重要
            if lat_ttc < config.ttc_warning:
                score += config.weight_lane_change_conflict * (config.ttc_warning - lat_ttc)

    else:
        # 远处车道：危险较低
        if lon_ttc < config.ttc_critical:
            score += config.weight_ttc_longitudinal * (config.ttc_critical - lon_ttc) * 0.3

    return score


# ============================================================
# SVO (Social Value Orientation) 计算 - 单位圆累加模型
# ============================================================

def compute_agent_svo(
    agent_history: np.ndarray,
    agent_future: np.ndarray = None,
    neighbor_histories: Dict[int, np.ndarray] = None,
    config: DangerScoreConfig = None
) -> Dict:
    """
    计算单个 agent 的 SVO 倾向（单位圆累加模型）

    在单位圆上：
    - x 轴（个人权重）：对自己利益的关注程度
    - y 轴（集体权重）：对他人/集体利益的关注程度
    - SVO 角度 θ = arctan2(y, x)

    行为对权重的影响：
    - 让行：减少个人权重 x，增加集体权重 y（合作倾向）
    - 抢行：增加个人权重 x，减少集体权重 y（竞争倾向）
    - 避让：略微减少 x，增加 y（合作倾向）
    - 靠近：略微增加 x，减少 y（竞争倾向）
    - 社会力：增加集体权重 y

    Args:
        agent_history: Agent 历史轨迹 (T_hist, 5) [x, y, heading, vx, vy]
        agent_future: Agent 未来轨迹 (T_future, 5)，可选
        neighbor_histories: 邻居车辆历史轨迹字典 {neighbor_id: trajectory}
        config: 危险评分配置

    Returns:
        {
            'svo_angle_deg': float,   # SVO 角度
            'svo_type': str,          # 离散类型
            'x_weight': float,        # 个人权重
            'y_weight': float,        # 集体权重
            'yield_count': int,       # 让行次数
            'aggressive_count': int,  # 抢行次数
            'avoidance_count': int,   # 避让次数
            'approach_count': int,    # 靠近次数
            'behavior_intensity': float,  # 行为强度总和
            'social_force_magnitude': float  # 社会力大小
        }
    """
    if config is None:
        config = DangerScoreConfig()

    # ========== 合并轨迹 ==========
    full_trajectory = agent_history
    if agent_future is not None:
        full_trajectory = np.concatenate([agent_history, agent_future], axis=0)

    dt = 0.1  # 时间步长，10Hz

    # ========== 第一部分：纵向行为检测（让行/抢行）==========

    yield_count = 0
    aggressive_count = 0

    if full_trajectory.shape[0] >= 2:
        # 计算速度标量
        speeds = np.sqrt(full_trajectory[:, 3]**2 + full_trajectory[:, 4]**2)

        # 计算加速度
        accel = np.diff(speeds) / dt

        # 让行：减速超过阈值
        yield_count = int(np.sum(accel < -config.svo_decel_threshold))

        # 抢行：加速超过阈值
        aggressive_count = int(np.sum(accel > config.svo_accel_threshold))

    # ========== 第二部分：横向行为检测（避让/靠近）==========

    avoidance_count = 0
    approach_count = 0

    if full_trajectory.shape[0] >= 1 and neighbor_histories is not None and len(neighbor_histories) > 0:
        # 计算横向速度
        vx = full_trajectory[:, 3]
        vy = full_trajectory[:, 4]
        heading = full_trajectory[:, 2]

        lateral_speed = -vx * np.sin(heading) + vy * np.cos(heading)

        for neighbor_id, neighbor_traj in neighbor_histories.items():
            min_len = min(len(full_trajectory), len(neighbor_traj))

            for t in range(min_len):
                dist = np.sqrt(np.sum((full_trajectory[t, :2] - neighbor_traj[t, :2])**2))

                if dist < config.svo_risk_distance_threshold and dist > 0.1:
                    rel_pos = neighbor_traj[t, :2] - full_trajectory[t, :2]
                    neighbor_lateral_dir = np.sign(rel_pos[1])

                    if abs(lateral_speed[t]) > config.svo_lateral_speed_threshold:
                        if lateral_speed[t] * neighbor_lateral_dir < 0:
                            avoidance_count += 1
                        else:
                            approach_count += 1

    avoidance_count = int(avoidance_count)
    approach_count = int(approach_count)

    # ========== 第三部分：社会力计算（使用平均距离）==========

    social_force_magnitude = 0.0

    if neighbor_histories is not None and len(neighbor_histories) > 0:
        for neighbor_id, neighbor_traj in neighbor_histories.items():
            min_len = min(len(full_trajectory), len(neighbor_traj))

            distances = []
            for t in range(min_len):
                dist = np.sqrt(np.sum((full_trajectory[t, :2] - neighbor_traj[t, :2])**2))
                distances.append(dist)

            avg_dist = np.mean(distances)

            if avg_dist < config.svo_risk_distance_threshold and avg_dist > 0.1:
                force = (config.svo_risk_distance_threshold - avg_dist) / config.svo_risk_distance_threshold
                social_force_magnitude += force

    # ========== 第四部分：SVO 权重累加 ==========

    # 初始点：(1.0, 0.0) - 个人主义，θ = 0°
    x_raw = 1.0
    y_raw = 0.0

    # 让行行为影响
    x_raw += config.svo_yield_delta_x * yield_count
    y_raw += config.svo_yield_delta_y * yield_count

    # 抢行行为影响
    x_raw += config.svo_aggressive_delta_x * aggressive_count
    y_raw += config.svo_aggressive_delta_y * aggressive_count

    # 避让行为影响
    x_raw += config.svo_avoidance_delta_x * avoidance_count
    y_raw += config.svo_avoidance_delta_y * avoidance_count

    # 靠近行为影响
    x_raw += config.svo_approach_delta_x * approach_count
    y_raw += config.svo_approach_delta_y * approach_count

    # 社会力影响（只影响集体权重）
    y_raw += config.svo_social_force_delta_y * social_force_magnitude

    # 保证 x 有最小正值（人总是会关注自己的利益）
    x_raw = max(x_raw, 0.1)

    # ========== 第五部分：归一化到单位圆 ==========

    # 归一化：让 (x, y) 在单位圆上，满足 ||(x, y)|| = 1
    # 这样可以保证 x = cos(θ), y = sin(θ)，便于 diffusion 中反向计算
    norm = np.sqrt(x_raw**2 + y_raw**2)
    if norm > 1e-6:
        x_unit = x_raw / norm
        y_unit = y_raw / norm
    else:
        x_unit = 1.0
        y_unit = 0.0

    # 计算 SVO 角度
    # 在单位圆上，θ = arctan2(y_unit, x_unit)
    # 且 x_unit = cos(θ), y_unit = sin(θ)
    svo_angle_rad = np.arctan2(y_unit, x_unit)
    svo_angle_deg = np.rad2deg(svo_angle_rad)

    # 限制角度范围
    svo_angle_deg = np.clip(svo_angle_deg, config.svo_angle_min, config.svo_angle_max)

    # ========== 第六部分：离散类型分类 ==========

    if svo_angle_deg < -15:
        svo_type = 'competitive'
    elif svo_angle_deg < 22.5:
        svo_type = 'individualistic'
    elif svo_angle_deg < 57.5:
        svo_type = 'prosocial'
    else:
        svo_type = 'altruistic'

    behavior_intensity = float(yield_count + aggressive_count + avoidance_count + approach_count)

    return {
        # SVO 角度（用于 diffusion 和 reward 函数）
        'svo_angle_deg': float(svo_angle_deg),
        'svo_angle_rad': float(svo_angle_rad),
        'svo_type': svo_type,

        # 单位圆上的权重（用于 reward 函数：x = cos(θ), y = sin(θ)）
        'x_weight': float(x_unit),  # = cos(θ)
        'y_weight': float(y_unit),  # = sin(θ)

        # 原始累加值（保留强度信息，用于分析）
        'x_raw': float(x_raw),
        'y_raw': float(y_raw),
        'norm': float(norm),  # 累加向量的长度（代表行为强度）

        # 行为统计
        'yield_count': yield_count,
        'aggressive_count': aggressive_count,
        'avoidance_count': avoidance_count,
        'approach_count': approach_count,
        'behavior_intensity': behavior_intensity,
        'social_force_magnitude': float(social_force_magnitude)
    }


def compute_pair_svo(
    ego_history: np.ndarray,
    ego_future: np.ndarray,
    agent_history: np.ndarray,
    agent_future: np.ndarray,
    config: DangerScoreConfig = None
) -> Dict:
    """
    计算 ego-agent pair 的交互 SVO（简化版）

    基于 SVO 角度直接计算交互指标，修正了交互类型判断逻辑。

    Args:
        ego_history: Ego 历史轨迹 (T_hist, 5)
        ego_future: Ego 未来轨迹 (T_future, 5)
        agent_history: Agent 历史轨迹 (T_hist, 5)
        agent_future: Agent 未来轨迹 (T_future, 5)
        config: 危险评分配置

    Returns:
        {
            'ego_svo_angle_deg': float,
            'agent_svo_angle_deg': float,
            'svo_angle_diff': float,
            'cooperation_score': float,      # 归一化合作分数
            'conflict_tendency': float,      # 冲突倾向
            'conflict_intensity': float,
            'interaction_type': str,         # 交互类型
            'ego_svo_type': str,
            'agent_svo_type': str,
            'ego_svo': Dict,
            'agent_svo': Dict,
            'min_distance': float,
            'approach_speed': float
        }
    """
    if config is None:
        config = DangerScoreConfig()

    # ========== 计算各自的 SVO ==========
    # 传入对方轨迹用于避让检测
    ego_svo = compute_agent_svo(ego_history, ego_future, {'agent': agent_history}, config)
    agent_svo = compute_agent_svo(agent_history, agent_future, {'ego': ego_history}, config)

    # ========== 连续变量计算 ==========

    ego_angle = ego_svo['svo_angle_deg']
    agent_angle = agent_svo['svo_angle_deg']

    # SVO 角度差
    svo_angle_diff = abs(ego_angle - agent_angle)

    # 合作分数：基于双方 SVO 角度
    # 角度越高越合作，归一化到 [-1, 1]
    # 使用 67.5° 作为中性点（合作型和利他型的中间）
    cooperation_score = (ego_angle + agent_angle) / 2 / 67.5
    cooperation_score = np.clip(cooperation_score, -1, 1)

    # 冲突倾向：基于双方竞争程度
    # 当双方角度都为负（竞争型）时，冲突倾向高
    conflict_tendency = 0.0
    if ego_angle < 0 and agent_angle < 0:
        # 双方都是竞争型
        conflict_tendency = abs(ego_angle) + abs(agent_angle) / 90.0
    elif ego_angle < 0 or agent_angle < 0:
        # 一方竞争型
        conflict_tendency = abs(min(ego_angle, agent_angle)) / 90.0
    conflict_tendency = np.clip(conflict_tendency, 0, 1)

    # ========== 冲突烈度计算 ==========

    # 合并轨迹
    ego_full = np.concatenate([ego_history, ego_future], axis=0)
    agent_full = np.concatenate([agent_history, agent_future], axis=0)

    min_len = min(len(ego_full), len(agent_full))

    min_distance = float('inf')
    approach_speed = 0.0

    for t in range(min_len):
        ego_pos = ego_full[t, :2]
        agent_pos = agent_full[t, :2]

        dist = np.sqrt(np.sum((ego_pos - agent_pos) ** 2))
        min_distance = min(min_distance, dist)

        # 计算接近速度
        if t > 0:
            ego_vel = ego_full[t, 3:5]
            agent_vel = agent_full[t, 3:5]

            rel_vel = agent_vel - ego_vel
            rel_pos = agent_pos - ego_pos

            if np.linalg.norm(rel_pos) > 0.1:
                approach = np.dot(rel_vel, rel_pos) / np.linalg.norm(rel_pos)
                if approach > 0:
                    approach_speed = max(approach_speed, approach)

    # 归一化冲突烈度
    distance_factor = np.clip(1 - min_distance / config.svo_risk_distance_threshold, 0, 1)
    speed_factor = np.clip(approach_speed / 10.0, 0, 1)

    conflict_intensity = (distance_factor + speed_factor) / 2

    # 如果双方都是竞争型，增加冲突烈度
    if ego_angle < 0 and agent_angle < 0:
        conflict_intensity *= 1.5

    # ========== 交互类型分类（修正版）==========

    avg_angle = (ego_angle + agent_angle) / 2

    if avg_angle >= 22.5:
        # 平均角度达到合作型阈值
        interaction_type = 'cooperative'
    elif ego_angle < -15 and agent_angle < -15:
        # 双方都是竞争型
        interaction_type = 'conflict'
    elif svo_angle_diff > 45:
        # 角度差异大，一方合作一方竞争
        interaction_type = 'avoidance'
    else:
        interaction_type = 'mixed'

    # 离散 SVO 类型
    ego_svo_type = ego_svo['svo_type']
    agent_svo_type = agent_svo['svo_type']

    return {
        # 连续变量
        'ego_svo_angle_deg': ego_angle,
        'agent_svo_angle_deg': agent_angle,
        'svo_angle_diff': svo_angle_diff,
        'cooperation_score': float(cooperation_score),
        'conflict_tendency': float(conflict_tendency),
        'conflict_intensity': float(conflict_intensity),

        # 离散类型
        'interaction_type': interaction_type,
        'ego_svo_type': ego_svo_type,
        'agent_svo_type': agent_svo_type,

        # 详细信息
        'ego_svo': ego_svo,
        'agent_svo': agent_svo,
        'min_distance': float(min_distance),
        'approach_speed': float(approach_speed)
    }


# ============================================================
# 核心TTC计算
# ============================================================

def compute_ttc_1d(
    distance: float,
    relative_velocity: float,
    epsilon: float = 1e-6
) -> float:
    """
    计算一维TTC（碰撞时间）

    Args:
        distance: 车辆间距离（正值）
        relative_velocity: 相对速度（负值表示接近）
        epsilon: 避免除零的小值

    Returns:
        TTC值（如果不发生碰撞则返回inf）
    """
    if relative_velocity >= 0:
        # 正在远离，不会碰撞
        return np.inf

    if distance <= 0:
        # 已经重叠
        return 0.0

    return distance / (-relative_velocity + epsilon)


def compute_relative_state(
    ego_state: np.ndarray,
    agent_state: np.ndarray
) -> Dict[str, float]:
    """
    计算ego和agent在单个时间步的相对状态

    Args:
        ego_state: [x, y, vx, vy] 或 [x, y, heading, vx, vy]
        agent_state: 与ego_state相同格式

    Returns:
        包含相对状态信息的字典
    """
    # 提取位置
    ego_x, ego_y = ego_state[0], ego_state[1]
    agent_x, agent_y = agent_state[0], agent_state[1]

    # 提取速度（处理4D和5D两种格式）
    if len(ego_state) >= 4:
        ego_vx, ego_vy = ego_state[-2], ego_state[-1]
        agent_vx, agent_vy = agent_state[-2], agent_state[-1]
    else:
        ego_vx, ego_vy = 0.0, 0.0
        agent_vx, agent_vy = 0.0, 0.0

    # 相对位置（ego坐标系）
    rel_x = agent_x - ego_x   # 正值 = agent在前方
    rel_y = agent_y - ego_y   # 正值 = agent在左侧

    # 相对速度
    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    # 距离计算
    longitudinal_dist = abs(rel_x)
    lateral_dist = abs(rel_y)
    euclidean_dist = np.sqrt(rel_x**2 + rel_y**2)

    # 接近速度（closing speed）：相对速度在两车连线方向的分量，接近时为正
    if euclidean_dist > 1e-6:
        dir_x = rel_x / euclidean_dist
        dir_y = rel_y / euclidean_dist
        closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)
        closing_speed = max(closing_speed, 0.0)
    else:
        closing_speed = 0.0

    return {
        'rel_x': rel_x,
        'rel_y': rel_y,
        'rel_vx': rel_vx,
        'rel_vy': rel_vy,
        'longitudinal_dist': longitudinal_dist,
        'lateral_dist': lateral_dist,
        'euclidean_dist': euclidean_dist,
        'relative_speed': closing_speed   # 保留 key 名称，语义改为 closing_speed
    }


def compute_ttc_at_timestep(
    ego_state: np.ndarray,
    agent_state: np.ndarray,
    config: DangerScoreConfig
) -> Dict[str, float]:
    """
    计算单个时间步的TTC值

    Args:
        ego_state: 该时间步的ego状态
        agent_state: 该时间步的agent状态
        config: 危险评分配置

    Returns:
        包含TTC值的字典
    """
    rel_state = compute_relative_state(ego_state, agent_state)

    # 纵向TTC
    # Agent在前方：rel_x > 0，如果rel_vx < 0（agent更慢）则在接近
    # Agent在后方：rel_x < 0，如果rel_vx > 0（agent更快）则在接近
    if rel_state['rel_x'] > 0:
        # Agent在前方 - ego正在接近
        lon_ttc = compute_ttc_1d(
            rel_state['rel_x'],
            rel_state['rel_vx']  # 负值表示ego更快 = 正在接近
        )
    else:
        # Agent在后方 - 检查agent是否正在接近ego
        lon_ttc = compute_ttc_1d(
            abs(rel_state['rel_x']),
            -rel_state['rel_vx']  # 符号取反
        )

    # 横向TTC
    if rel_state['rel_y'] > 0:
        # Agent在左侧
        lat_ttc = compute_ttc_1d(
            rel_state['rel_y'],
            rel_state['rel_vy']
        )
    else:
        # Agent在右侧
        lat_ttc = compute_ttc_1d(
            abs(rel_state['rel_y']),
            -rel_state['rel_vy']
        )

    # 车道关系
    lane_rel = get_lane_relationship(ego_state[1], agent_state[1], config)

    # 变道意图
    agent_vy = agent_state[-1] if len(agent_state) >= 4 else 0.0
    lane_change = detect_lane_change_intent(agent_vy, config)

    return {
        'lon_ttc': lon_ttc,
        'lat_ttc': lat_ttc,
        'euclidean_dist': rel_state['euclidean_dist'],
        'relative_speed': rel_state['relative_speed'],
        'rel_state': rel_state,
        'lane_relationship': lane_rel,
        'lane_change_intent': lane_change
    }


# ============================================================
# 危险评分组件
# ============================================================

def ttc_danger_score(
    lon_ttc: float,
    lat_ttc: float,
    config: DangerScoreConfig
) -> float:
    """
    基于TTC值计算危险评分

    TTC越低 = 危险越高

    Args:
        lon_ttc: 纵向TTC
        lat_ttc: 横向TTC
        config: 配置参数

    Returns:
        TTC相关的危险评分
    """
    score = 0.0

    # 纵向TTC危险
    if lon_ttc < config.ttc_critical:
        score += config.weight_ttc_longitudinal * (config.ttc_critical - lon_ttc) * 2.0
    elif lon_ttc < config.ttc_warning:
        score += config.weight_ttc_longitudinal * (config.ttc_warning - lon_ttc)
    elif lon_ttc < config.ttc_max:
        score += config.weight_ttc_longitudinal * (config.ttc_max - lon_ttc) * 0.1

    # 横向TTC危险
    if lat_ttc < config.ttc_critical:
        score += config.weight_ttc_lateral * (config.ttc_critical - lat_ttc) * 2.0
    elif lat_ttc < config.ttc_warning:
        score += config.weight_ttc_lateral * (config.ttc_warning - lat_ttc)
    elif lat_ttc < config.ttc_max:
        score += config.weight_ttc_lateral * (config.ttc_max - lat_ttc) * 0.1

    return score


def distance_danger_score(
    min_distance: float,
    config: DangerScoreConfig
) -> float:
    """
    基于最小距离计算危险评分（near-miss范式）

    使用帐篷函数（tent function）：峰值在 near_miss_distance（3m），
    向碰撞方向和安全方向均平滑衰减至0。
    碰撞（dist <= collision_distance）返回0，不给予奖励。

    设计依据：STRIVE (NVIDIA, CVPR 2022) —— "effective safety-critical scenario =
    inducing small TTC WITHOUT causing collision; minSTTC < 3s = near-miss."
    碰撞是无用的结果（terminal state，轨迹不真实），near-miss才是训练目标。

    Args:
        min_distance: 车辆间最小距离
        config: 配置参数

    Returns:
        距离相关的危险评分
    """
    near_miss_peak = config.weight_near_miss  # 3.0，峰值在 near_miss_distance

    # 碰撞区：不给奖励，避免模型以碰撞为目标
    if min_distance <= config.collision_distance:
        return 0.0

    # 上升侧：collision_distance(2m) → near_miss_distance(3m)，从0升到peak
    if min_distance <= config.near_miss_distance:
        return near_miss_peak * (min_distance - config.collision_distance) / \
               (config.near_miss_distance - config.collision_distance)

    # 下降侧：near_miss_distance(3m) → min_safe_distance(5m)，从peak降到0
    if min_distance <= config.min_safe_distance:
        return near_miss_peak * (config.min_safe_distance - min_distance) / \
               (config.min_safe_distance - config.near_miss_distance)

    return 0.0


def collision_detection(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    config: DangerScoreConfig
) -> Tuple[bool, float, int]:
    """
    检测ego和agent轨迹之间的碰撞

    使用简单的AABB距离检测（可改进为OBB）

    Args:
        ego_traj: Ego轨迹 (T, 3) 或 (T, 5) - [x, y, heading, ...]
        agent_traj: Agent轨迹 (T, 3) 或 (T, 5)
        config: 配置参数

    Returns:
        (是否碰撞, 最小距离, 碰撞时间步)
    """
    has_collision = False
    min_distance = np.inf
    collision_timestep = -1

    for t in range(len(ego_traj)):
        ego_x, ego_y = ego_traj[t, 0], ego_traj[t, 1]
        agent_x, agent_y = agent_traj[t, 0], agent_traj[t, 1]

        # 简单欧氏距离（效率优先）
        dist = np.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2)

        if dist < min_distance:
            min_distance = dist

        # 使用简单距离检测碰撞（可用OBB改进）
        collision_threshold = (config.ego_length + config.agent_length) / 4 + \
                             (config.ego_width + config.agent_width) / 4

        if dist < collision_threshold and not has_collision:
            has_collision = True
            collision_timestep = t

    return has_collision, min_distance, collision_timestep


def relative_velocity_danger_score(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    config: DangerScoreConfig
) -> float:
    """
    基于相对速度计算危险评分

    近距离时高相对速度更危险

    Args:
        ego_traj: Ego轨迹
        agent_traj: Agent轨迹
        config: 配置参数

    Returns:
        速度相关的危险评分
    """
    max_score = 0.0

    for t in range(len(ego_traj)):
        ttc_result = compute_ttc_at_timestep(ego_traj[t], agent_traj[t], config)

        # 近距离高相对速度很危险
        if ttc_result['euclidean_dist'] < config.min_safe_distance:
            speed_score = ttc_result['relative_speed'] * config.weight_relative_velocity
            # 非常近时加倍
            if ttc_result['euclidean_dist'] < config.near_miss_distance:
                speed_score *= 2.0
            max_score = max(max_score, speed_score)

    return max_score


def acceleration_danger_score(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    dt: float,
    config: DangerScoreConfig
) -> float:
    """
    基于加速度模式计算危险评分

    碰撞附近的急刹车或急转向很危险

    Args:
        ego_traj: Ego轨迹
        agent_traj: Agent轨迹
        dt: 时间步长
        config: 配置参数

    Returns:
        加速度相关的危险评分
    """
    score = 0.0

    if len(ego_traj) < 3:
        return 0.0

    for t in range(1, len(ego_traj) - 1):
        # 检查该时间步的距离
        dist = np.sqrt(
            (ego_traj[t, 0] - agent_traj[t, 0])**2 +
            (ego_traj[t, 1] - agent_traj[t, 1])**2
        )

        if dist > config.min_safe_distance:
            continue

        # 计算ego加速度
        if ego_traj.shape[1] >= 4:
            ego_ax = (ego_traj[t+1, -2] - ego_traj[t-1, -2]) / (2 * dt)
            ego_ay = (ego_traj[t+1, -1] - ego_traj[t-1, -1]) / (2 * dt)
            ego_accel = np.sqrt(ego_ax**2 + ego_ay**2)

            # 碰撞附近的高加速度很危险
            if ego_accel > config.max_acceleration:
                score += config.weight_acceleration * (ego_accel - config.max_acceleration)

    return score


def lane_change_conflict_score(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    config: DangerScoreConfig
) -> float:
    """
    计算变道冲突危险评分

    检测两车是否有向同一车道变道的意图

    Args:
        ego_traj: Ego轨迹
        agent_traj: Agent轨迹

    Returns:
        变道冲突危险评分
    """
    score = 0.0

    for t in range(len(ego_traj)):
        ego_y = ego_traj[t, 1]
        agent_y = agent_traj[t, 1]

        ego_vy = ego_traj[t, -1] if ego_traj.shape[1] >= 4 else 0.0
        agent_vy = agent_traj[t, -1] if agent_traj.shape[1] >= 4 else 0.0

        # 检测变道意图
        ego_intent, ego_dir = detect_lane_change_intent(ego_vy, config)
        agent_intent, agent_dir = detect_lane_change_intent(agent_vy, config)

        # 如果两车都有变道意图
        if ego_intent and agent_intent:
            # 检查是否向同一车道变道（方向相反）
            if ego_dir != agent_dir:
                # 计算距离
                dist = abs(ego_y - agent_y)
                if dist < config.adjacent_lane_threshold:
                    # 在相邻车道，向同一车道变道很危险
                    score += config.weight_lane_change_conflict

    return score


# ============================================================
# 辅助函数：计算单段轨迹的危险指数
# ============================================================

def compute_single_trajectory_danger(
    ego_traj: np.ndarray,
    agent_traj: np.ndarray,
    config: DangerScoreConfig,
    traj_type: str = "future"
) -> Dict:
    """
    计算单段轨迹（历史或未来）的危险指数

    Args:
        ego_traj: Ego轨迹 (T, D)
        agent_traj: Agent轨迹 (T, D)
        config: 危险评分配置
        traj_type: "history" 或 "future"，用于标识

    Returns:
        该段轨迹的危险指数字典
    """
    # 初始化组件评分（与 torch 版 canonical 5 分量对齐）
    component_scores = {
        'ttc_score': 0.0,
        'distance_score': 0.0,
        'collision_score': 0.0,    # 仅监控，不计入 total（distance_score 已覆盖）
        'near_miss_score': 0.0,    # 仅监控，不计入 total
        'velocity_score': 0.0,
    }

    # TTC分析
    ttc_info = {
        'min_lon_ttc': np.inf,
        'min_lat_ttc': np.inf,
        'lon_ttc_sequence': [],
        'lat_ttc_sequence': [],
        'critical_timestep': -1
    }

    lane_info = {
        'lane_relationships': [],
        'lane_change_intents': [],
        'dominant_lane_rel': None
    }
    lane_rel_counts = {}

    for t in range(len(ego_traj)):
        ttc_result = compute_ttc_at_timestep(ego_traj[t], agent_traj[t], config)

        ttc_info['lon_ttc_sequence'].append(ttc_result['lon_ttc'])
        ttc_info['lat_ttc_sequence'].append(ttc_result['lat_ttc'])

        if ttc_result['lon_ttc'] < ttc_info['min_lon_ttc']:
            ttc_info['min_lon_ttc'] = ttc_result['lon_ttc']
            ttc_info['critical_timestep'] = t

        if ttc_result['lat_ttc'] < ttc_info['min_lat_ttc']:
            ttc_info['min_lat_ttc'] = ttc_result['lat_ttc']

        # 累积TTC危险评分
        component_scores['ttc_score'] += ttc_danger_score(
            ttc_result['lon_ttc'],
            ttc_result['lat_ttc'],
            config
        ) / len(ego_traj)

        lane_rel = ttc_result['lane_relationship']
        lane_info['lane_relationships'].append(lane_rel)
        lane_rel_counts[lane_rel] = lane_rel_counts.get(lane_rel, 0) + 1
        lane_info['lane_change_intents'].append(ttc_result['lane_change_intent'])

    if lane_rel_counts:
        lane_info['dominant_lane_rel'] = max(lane_rel_counts, key=lane_rel_counts.get)

    # 碰撞检测
    has_collision, min_distance, collision_t = collision_detection(
        ego_traj, agent_traj, config
    )

    collision_info = {
        'has_collision': has_collision,
        'min_distance': min_distance,
        'collision_timestep': collision_t
    }

    if has_collision:
        component_scores['collision_score'] = config.weight_collision * 10.0
    elif min_distance < config.near_miss_distance:
        component_scores['near_miss_score'] = config.weight_near_miss * \
            (config.near_miss_distance - min_distance) / \
            (config.near_miss_distance - config.collision_distance)

    component_scores['distance_score'] = distance_danger_score(min_distance, config)

    # 速度危险
    component_scores['velocity_score'] = relative_velocity_danger_score(
        ego_traj, agent_traj, config
    )

    # 计算总分（与 torch 版一致：ttc + distance + velocity，collision/near_miss 仅监控）
    total_score = (component_scores['ttc_score'] +
                   component_scores['distance_score'] +
                   component_scores['velocity_score'])

    return {
        'traj_type': traj_type,
        'total_score': total_score,
        'component_scores': component_scores,
        'ttc_info': ttc_info,
        'collision_info': collision_info,
        'lane_info': lane_info,
        'danger_level': classify_danger_level(total_score, config)
    }


# ============================================================
# 主危险评分函数
# ============================================================

def compute_danger_score(
    ego_history: np.ndarray,
    ego_future: np.ndarray,
    agent_history: np.ndarray,
    agent_future: np.ndarray,
    config: DangerScoreConfig = None,
    dt: float = 0.1,
    verbose: bool = False
) -> Dict:
    """
    计算ego-agent轨迹对的综合性危险评分，分离历史和未来两个指标

    这是diffusion训练和推理时的主函数。

    Args:
        ego_history: Ego历史轨迹 (T_hist, D)
        ego_future: 待评估的Ego未来轨迹 (T_future, D)
        agent_history: Agent历史轨迹 (T_hist, D)
        agent_future: 待评估的Agent未来轨迹 (T_future, D)
        config: 危险评分配置
        dt: 时间步长（秒）
        verbose: 是否打印详细评分

    Returns:
        字典包含：
        - inherent_danger: 历史轨迹的危险指数（场景固有风险）
        - outcome_danger: 未来轨迹的危险指数（结果风险）
        - svo_info: SVO交互分析
        - avoidance_quality: 避让质量
    """
    if config is None:
        config = DangerScoreConfig()

    # 1. 计算历史轨迹的危险指数（场景固有风险）
    inherent_danger = compute_single_trajectory_danger(
        ego_history, agent_history, config, traj_type="history"
    )

    # 2. 计算未来轨迹的危险指数（结果风险）
    outcome_danger = compute_single_trajectory_danger(
        ego_future, agent_future, config, traj_type="future"
    )

    # 3. 计算避让质量
    # 避让质量 = 1 - outcome_danger / inherent_danger（如果历史有风险）
    if inherent_danger['total_score'] > 0:
        avoidance_quality = 1.0 - np.clip(
            outcome_danger['total_score'] / inherent_danger['total_score'],
            0, 1
        )
    else:
        # 历史无风险，如果未来也无风险则质量为1，否则为负
        if outcome_danger['total_score'] > 0:
            avoidance_quality = -outcome_danger['total_score']  # 负值表示制造了风险
        else:
            avoidance_quality = 1.0

    # 4. SVO 交互分析（基于历史+未来）
    svo_info = compute_pair_svo(
        ego_history, ego_future,
        agent_history, agent_future,
        config
    )

    # 准备结果
    result = {
        # 分离的危险指数
        'inherent_danger': inherent_danger,
        'outcome_danger': outcome_danger,
        'avoidance_quality': avoidance_quality,

        # SVO 信息
        'svo_info': svo_info,

        # 简化的总分（可选，用于快速比较）
        'total_score': inherent_danger['total_score'] + outcome_danger['total_score'],

        # 危险等级基于历史风险（场景本身的危险程度）
        'danger_level': inherent_danger['danger_level'],

        # 用于兼容旧代码的合并信息
        'ttc_info': outcome_danger['ttc_info'],
        'collision_info': outcome_danger['collision_info'],
        'lane_info': outcome_danger['lane_info'],
        'component_scores': {
            'inherent_ttc': inherent_danger['component_scores']['ttc_score'],
            'outcome_ttc': outcome_danger['component_scores']['ttc_score'],
            'inherent_distance': inherent_danger['component_scores']['distance_score'],
            'outcome_distance': outcome_danger['component_scores']['distance_score'],
        }
    }

    if verbose:
        print_danger_score_report(result)

    return result


def classify_danger_level(score: float, config: DangerScoreConfig) -> str:
    """
    将危险评分分类为离散等级

    Args:
        score: 总危险评分
        config: 配置参数

    Returns:
        危险等级字符串
    """
    # 阈值基于 near-miss 范式下的实际得分范围（max ~10）
    if score >= 6.0:
        return 'critical'
    elif score >= 3.0:
        return 'high'
    elif score >= 1.0:
        return 'medium'
    elif score > 0:
        return 'low'
    else:
        return 'safe'


def print_danger_score_report(result: Dict):
    """打印详细危险评分报告（分离历史和未来）"""
    print('\n' + '='*60)
    print('危险评分报告')
    print('='*60)

    # 显示分离的危险指数
    print('\n=== 危险指数分解 ===')
    inherent = result['inherent_danger']
    outcome = result['outcome_danger']

    print(f'\n[历史轨迹危险指数 (inherent_danger)]')
    print(f'  总分: {inherent["total_score"]:.2f}')
    print(f'  危险等级: {inherent["danger_level"].upper()}')
    print(f'  最小距离: {inherent["collision_info"]["min_distance"]:.2f}m')
    print(f'  TTC分析:')
    print(f'    - 最小纵向TTC: {inherent["ttc_info"]["min_lon_ttc"]:.2f}s')
    print(f'    - 最小横向TTC: {inherent["ttc_info"]["min_lat_ttc"]:.2f}s')

    print(f'\n[未来轨迹危险指数 (outcome_danger)]')
    print(f'  总分: {outcome["total_score"]:.2f}')
    print(f'  危险等级: {outcome["danger_level"].upper()}')
    print(f'  最小距离: {outcome["collision_info"]["min_distance"]:.2f}m')
    print(f'  TTC分析:')
    print(f'    - 最小纵向TTC: {outcome["ttc_info"]["min_lon_ttc"]:.2f}s')
    print(f'    - 最小横向TTC: {outcome["ttc_info"]["min_lat_ttc"]:.2f}s')
    if outcome["collision_info"]["has_collision"]:
        print(f'    - 碰撞时间步: {outcome["collision_info"]["collision_timestep"]}')

    print(f'\n[避让质量]')
    print(f'  avoidance_quality: {result["avoidance_quality"]:.3f}')
    print(f'  (1.0 = 完美避让, 0.0 = 未改善, <0 = 制造风险)')

    # 组件评分
    print('\n--- 组件评分分解 ---')
    print('\n历史组件:')
    for name, score in inherent['component_scores'].items():
        if score > 0:
            print(f'  {name}: {score:.2f}')
    print('\n未来组件:')
    for name, score in outcome['component_scores'].items():
        if score > 0:
            print(f'  {name}: {score:.2f}')

    # SVO 分析
    if 'svo_info' in result:
        print('\n--- SVO 分析（单位圆累加模型）---')
        svo = result['svo_info']

        # SVO 角度和类型
        print(f'  Ego SVO 角度: {svo["ego_svo_angle_deg"]:.1f}° ({svo["ego_svo_type"]})')
        print(f'  Agent SVO 角度: {svo["agent_svo_angle_deg"]:.1f}° ({svo["agent_svo_type"]})')
        print(f'  SVO 角度差: {svo["svo_angle_diff"]:.1f}°')

        # 行为统计
        ego_svo = svo['ego_svo']
        agent_svo = svo['agent_svo']

        print(f'\n  Ego 行为统计:')
        print(f'    - 单位圆权重: x={ego_svo["x_weight"]:.3f}, y={ego_svo["y_weight"]:.3f}')
        print(f'    - 原始累加值: x_raw={ego_svo["x_raw"]:.2f}, y_raw={ego_svo["y_raw"]:.2f}')
        print(f'    - 行为强度(norm): {ego_svo["norm"]:.2f}')
        print(f'    - 让行次数: {ego_svo["yield_count"]}')
        print(f'    - 抢行次数: {ego_svo["aggressive_count"]}')
        print(f'    - 避让次数: {ego_svo["avoidance_count"]}')
        print(f'    - 社会力: {ego_svo["social_force_magnitude"]:.2f}')

        print(f'\n  Agent 行为统计:')
        print(f'    - 单位圆权重: x={agent_svo["x_weight"]:.3f}, y={agent_svo["y_weight"]:.3f}')
        print(f'    - 原始累加值: x_raw={agent_svo["x_raw"]:.2f}, y_raw={agent_svo["y_raw"]:.2f}')
        print(f'    - 行为强度(norm): {agent_svo["norm"]:.2f}')
        print(f'    - 让行次数: {agent_svo["yield_count"]}')
        print(f'    - 抢行次数: {agent_svo["aggressive_count"]}')
        print(f'    - 避让次数: {agent_svo["avoidance_count"]}')
        print(f'    - 社会力: {agent_svo["social_force_magnitude"]:.2f}')

        # 交互指标
        print(f'\n  交互指标:')
        print(f'    - 合作分数: {svo["cooperation_score"]:.3f}')
        print(f'    - 冲突倾向: {svo["conflict_tendency"]:.3f}')
        print(f'    - 冲突烈度: {svo["conflict_intensity"]:.3f}')
        print(f'    - 交互类型: {svo["interaction_type"]}')


# ============================================================
# Diffusion引导工具函数
# ============================================================

def svo_angle_to_weights(svo_angle_deg: float) -> Tuple[float, float]:
    """
    从 SVO 角度反向计算单位圆上的权重

    在单位圆上，天然满足：
    - x = cos(θ)
    - y = sin(θ)

    这个函数用于 diffusion 中：
    1. 调整角度 θ_new = θ + Δθ
    2. 用本函数得到新的权重 (x_new, y_new)
    3. 用权重计算 reward

    Args:
        svo_angle_deg: SVO 角度（度）

    Returns:
        (x_weight, y_weight): 单位圆上的权重
    """
    svo_angle_rad = np.deg2rad(svo_angle_deg)
    x_weight = np.cos(svo_angle_rad)
    y_weight = np.sin(svo_angle_rad)
    return x_weight, y_weight


def compute_svo_reward(
    self_payoff: float,
    other_payoff: float,
    svo_angle_deg: float
) -> float:
    """
    基于 SVO 角度计算 reward

    Reward = x × self_payoff + y × other_payoff
    其中 x = cos(θ), y = sin(θ)

    Args:
        self_payoff: 自己的利益（如：到达目标的速度、舒适度）
        other_payoff: 他人的利益（如：他人的安全、他人的舒适）
        svo_angle_deg: SVO 角度（度）

    Returns:
        加权后的 reward
    """
    x, y = svo_angle_to_weights(svo_angle_deg)
    reward = x * self_payoff + y * other_payoff
    return reward


def compute_danger_gradient(
    ego_future: np.ndarray,
    agent_future: np.ndarray,
    ego_history: np.ndarray,
    agent_history: np.ndarray,
    config: DangerScoreConfig = None,
    epsilon: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算危险评分对轨迹位置的梯度

    用于diffusion模型的分类器引导。

    Args:
        ego_future: Ego未来轨迹 (T, D)
        agent_future: Agent未来轨迹 (T, D)
        ego_history: Ego历史轨迹（用于上下文）
        agent_history: Agent历史轨迹（用于上下文）
        config: 配置参数
        epsilon: 有限差分步长

    Returns:
        (ego梯度, agent梯度)
    """
    if config is None:
        config = DangerScoreConfig()

    ego_gradient = np.zeros_like(ego_future[:, :2])  # 仅x, y
    agent_gradient = np.zeros_like(agent_future[:, :2])

    # 计算基础评分
    base_result = compute_danger_score(
        ego_history, ego_future, agent_history, agent_future, config
    )
    base_score = base_result['total_score']

    # 计算ego轨迹梯度
    for t in range(len(ego_future)):
        for dim in range(2):  # x, y
            # 扰动位置
            ego_perturbed = ego_future.copy()
            ego_perturbed[t, dim] += epsilon

            # 计算新评分
            new_result = compute_danger_score(
                ego_history, ego_perturbed, agent_history, agent_future, config
            )

            # 梯度 = (新评分 - 基础评分) / epsilon
            ego_gradient[t, dim] = (new_result['total_score'] - base_score) / epsilon

    # 计算agent轨迹梯度
    for t in range(len(agent_future)):
        for dim in range(2):  # x, y
            agent_perturbed = agent_future.copy()
            agent_perturbed[t, dim] += epsilon

            new_result = compute_danger_score(
                ego_history, ego_future, agent_history, agent_perturbed, config
            )

            agent_gradient[t, dim] = (new_result['total_score'] - base_score) / epsilon

    return ego_gradient, agent_gradient


def batch_compute_danger_scores(
    ego_futures: np.ndarray,
    agent_futures: np.ndarray,
    ego_history: np.ndarray,
    agent_history: np.ndarray,
    config: DangerScoreConfig = None
) -> np.ndarray:
    """
    批量计算轨迹对的危险评分

    用于奖励加权回归或筛选。

    Args:
        ego_futures: Ego未来轨迹批次 (B, T, D)
        agent_futures: Agent未来轨迹批次 (B, T, D)
        ego_history: 单个Ego历史轨迹 (T_hist, D)
        agent_history: 单个Agent历史轨迹 (T_hist, D)
        config: 配置参数

    Returns:
        危险评分数组 (B,)
    """
    if config is None:
        config = DangerScoreConfig()

    batch_size = len(ego_futures)
    scores = np.zeros(batch_size)

    for i in range(batch_size):
        result = compute_danger_score(
            ego_history, ego_futures[i],
            agent_history, agent_futures[i],
            config
        )
        scores[i] = result['total_score']

    return scores


# ============================================================
# 测试函数
# ============================================================

def test_danger_score():
    """测试危险评分计算"""
    print("测试危险评分模块...")

    config = DangerScoreConfig()

    # 创建测试轨迹
    # Ego: 以恒定速度前进
    ego_hist = np.zeros((20, 5))
    ego_hist[:, 0] = np.linspace(-20, 0, 20)  # x: -20 到 0
    ego_hist[:, 3] = 10.0  # vx = 10 m/s

    ego_fut = np.zeros((30, 5))
    ego_fut[:, 0] = np.linspace(0, 30, 30)  # x: 0 到 30
    ego_fut[:, 3] = 10.0  # vx = 10 m/s

    # Agent: 前方，速度更慢（同车道）
    agent_hist = np.zeros((20, 5))
    agent_hist[:, 0] = np.linspace(0, 20, 20)  # x: 0 到 20
    agent_hist[:, 3] = 8.0  # vx = 8 m/s

    agent_fut = np.zeros((30, 5))
    agent_fut[:, 0] = np.linspace(20, 44, 30)  # x: 20 到 44
    agent_fut[:, 3] = 8.0  # vx = 8 m/s

    print("\n=== 测试场景1: Ego接近前方较慢的agent（同车道）===")
    result = compute_danger_score(
        ego_hist, ego_fut, agent_hist, agent_fut, config, verbose=True
    )

    # 测试碰撞场景
    print("\n=== 测试场景2: 碰撞轨迹 ===")
    agent_fut_collision = np.zeros((30, 5))
    agent_fut_collision[:, 0] = np.linspace(20, 25, 30)  # x: 20 到 25（更慢）
    agent_fut_collision[:, 1] = np.linspace(0, 0, 30)    # y: 0（同车道）

    result = compute_danger_score(
        ego_hist, ego_fut, agent_hist, agent_fut_collision, config, verbose=True
    )

    # 测试相邻车道场景
    print("\n=== 测试场景3: 相邻车道（无变道意图）===")
    agent_fut_adjacent = np.zeros((30, 5))
    agent_fut_adjacent[:, 0] = np.linspace(20, 50, 30)
    agent_fut_adjacent[:, 1] = np.linspace(3.5, 3.5, 30)  # y: 3.5m偏移（相邻车道）
    agent_fut_adjacent[:, 3] = 10.0  # vx = 10 m/s
    agent_fut_adjacent[:, 4] = 0.0   # vy = 0（无横向速度）

    result = compute_danger_score(
        ego_hist, ego_fut, agent_hist, agent_fut_adjacent, config, verbose=True
    )

    # 测试变道冲突场景
    print("\n=== 测试场景4: 变道冲突 ===")
    agent_fut_lanechange = np.zeros((30, 5))
    agent_fut_lanechange[:, 0] = np.linspace(20, 50, 30)
    agent_fut_lanechange[:, 1] = np.linspace(3.5, -1.0, 30)  # y: 从左车道变到ego车道
    agent_fut_lanechange[:, 3] = 10.0
    agent_fut_lanechange[:, 4] = -0.8  # vy: 向右变道

    result = compute_danger_score(
        ego_hist, ego_fut, agent_hist, agent_fut_lanechange, config, verbose=True
    )

    print("\n=== 测试梯度计算 ===")
    ego_grad, agent_grad = compute_danger_gradient(
        ego_fut, agent_fut_collision, ego_hist, agent_hist, config
    )
    print(f"Ego梯度形状: {ego_grad.shape}")
    print(f"Ego梯度最大值: {np.abs(ego_grad).max():.2f}")
    print(f"Agent梯度形状: {agent_grad.shape}")
    print(f"Agent梯度最大值: {np.abs(agent_grad).max():.2f}")

    print("\n所有测试完成！")


if __name__ == "__main__":
    test_danger_score()