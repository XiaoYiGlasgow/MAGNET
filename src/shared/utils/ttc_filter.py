"""
TTC (Time To Collision) 筛选模块

用于筛选存在碰撞风险的周围车辆。
计算横向TTC和纵向TTC，根据筛选逻辑过滤旁车。

数据格式说明（基于12D特征）：
- pkl文件中的DataFrame包含:
  - POLYLINE_FEATURES: shape=(N, 12) 所有节点的特征
    - [0:2] x_start, y_start - segment起点
    - [2:4] x_end, y_end - segment终点
    - [4] heading_start
    - [5] heading_end
    - [6] normalized_timestamp
    - [7] v_start - 起点速度
    - [8] v_end - 终点速度
    - [9] subgraph_ID
    - [10] start_time
    - [11] end_time
  - TRAJ_ID_TO_INDICES: 字典，轨迹ID到节点索引的映射 (ID=0是ego，ID>=1是agents)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

_THIS_FILE = Path(os.path.abspath(__file__))
_GNN_ROOT = next(
    (parent for parent in _THIS_FILE.parents if (parent / "config.py").exists()),
    _THIS_FILE.parent,
)

import sys
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

import config

# Danger score import — used for gt_future_danger filtering at dataset build time
try:
    from utils.danger_score import compute_danger_score as _compute_danger_score
    DANGER_SCORE_AVAILABLE = True
except ImportError:
    try:
        from GNN_Trans.utils.danger_score import compute_danger_score as _compute_danger_score
        DANGER_SCORE_AVAILABLE = True
    except ImportError:
        _compute_danger_score = None
        DANGER_SCORE_AVAILABLE = False

# NuPlan imports for agent future extraction
try:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    NUPLAN_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] NuPlan scenario builder not available: {e}")
    print("[INFO] Agent futures will not be extracted from NuPlan. Using None instead.")
    NUPLAN_AVAILABLE = False


# ============================================================
# 可配置参数
# ============================================================

class TTCConfig:
    """TTC筛选配置参数"""

    # 纵向TTC阈值（秒）
    # 超过此值认为纵向距离足够远，无碰撞风险
    LONGITUDINAL_TTC_THRESHOLD: float = 5.0

    # 横向TTC阈值（秒）
    # 超过此值认为横向距离足够远，无碰撞风险
    LATERAL_TTC_THRESHOLD: float = 2.0

    # 是否考虑车辆尺寸
    # True: 使用车辆边界框计算
    # False: 使用质心距离计算
    CONSIDER_VEHICLE_SIZE: bool = False

    # 车辆尺寸缓冲区（米）
    # 在计算TTC时增加的安全边距
    VEHICLE_SIZE_BUFFER: float = 0.5


class TimeWindowConfig:
    """时间窗口配置参数"""

    # 帧率 (Hz)
    DISCRETE_SIZE: int = 10

    # 历史时间窗口 (秒)
    PAST_TIME_HORIZON: float = 4.0

    # 未来时间窗口 (秒)
    FUTURE_TIME_HORIZON: float = 6.0

    # 历史帧数
    NUM_PAST_POSES: int = 40  # 4s * 10Hz

    # 未来帧数
    NUM_FUTURE_POSES: int = 60  # 6s * 10Hz


# ============================================================
# 场景选择配置 - 在这里修改要处理的场景
# ============================================================

class ScenarioConfig:
    """场景选择配置"""

    # 数据目录 - 指向正式 processed_data
    DATA_DIR: str = config.PROCESSED_DATA_DIR

    # 单个场景文件名（设为None则处理整个目录）
    # 修改这里来切换不同场景
    SINGLE_FILE: Optional[str] = None

    # 输出目录（保存txt结果文件）
    OUTPUT_DIR: str = config.TTC_FILTER_RESULTS_DIR

    # 选中pair数据输出目录（保存用于diffusion训练的pkl文件）
    SELECTED_PAIR_DIR: str = config.DANGER_PAIR_SCENARIOS_DIR

    # 最大处理文件数（None表示全部处理）
    MAX_FILES: Optional[int] = None


# ============================================================
# 数据载入
# ============================================================

def load_feature_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """加载单个特征文件"""
    return pd.read_pickle(file_path)


def get_available_scenarios(data_dir: str) -> List[str]:
    """获取数据目录中所有可用的场景文件"""
    data_dir = Path(data_dir)
    pkl_files = sorted([f.name for f in data_dir.glob("*.pkl")])
    return pkl_files


def extract_trajectory_from_polyline(
    polyline_features: np.ndarray,
    start_idx: int,
    end_idx: int
) -> np.ndarray:
    """
    从polyline特征中提取轨迹点

    Args:
        polyline_features: shape=(N, 12) 的特征数组
        start_idx: 起始索引
        end_idx: 结束索引

    Returns:
        trajectory: shape=(num_points, 2) 的轨迹数组
    """
    nodes = polyline_features[start_idx:end_idx + 1]
    # 每个节点是segment，需要重建轨迹点
    x_coords = np.concatenate([nodes[:, 0], [nodes[-1, 2]]])
    y_coords = np.concatenate([nodes[:, 1], [nodes[-1, 3]]])
    return np.stack([x_coords, y_coords], axis=1)


def extract_speed_from_polyline(
    polyline_features: np.ndarray,
    start_idx: int,
    end_idx: int,
    trajectory: np.ndarray
) -> np.ndarray:
    """
    从polyline特征中提取速度并转换为速度向量

    Args:
        polyline_features: shape=(N, 12) 的特征数组
        start_idx: 起始索引
        end_idx: 结束索引
        trajectory: 对应的轨迹点数组

    Returns:
        velocities: shape=(num_points, 2) 的速度向量数组
    """
    nodes = polyline_features[start_idx:end_idx + 1]
    v_start = nodes[:, 7]
    v_end = nodes[:, 8]
    speeds = np.concatenate([[v_start[0]], v_end])

    # 计算速度方向
    if len(trajectory) > 1:
        dx = np.diff(trajectory[:, 0], prepend=trajectory[0, 0])
        dy = np.diff(trajectory[:, 1], prepend=trajectory[0, 1])
        direction = np.arctan2(dy, dx)
        vx = speeds * np.cos(direction)
        vy = speeds * np.sin(direction)
    else:
        vx = np.zeros(len(speeds))
        vy = np.zeros(len(speeds))

    return np.stack([vx, vy], axis=1)


def load_and_prepare_data(file_path: str) -> Dict:
    """
    加载数据并准备TTC计算所需的格式

    Args:
        file_path: pkl文件路径

    Returns:
        包含ego_data, agents_data, num_frames等信息的字典
    """
    df = load_feature_file(file_path)
    polyline_features = df.iloc[0]['POLYLINE_FEATURES']
    traj_indices = df.iloc[0]['TRAJ_ID_TO_INDICES']

    # 提取ego轨迹 (ID=0)
    ego_start, ego_end = traj_indices[0]
    ego_traj = extract_trajectory_from_polyline(polyline_features, ego_start, ego_end)
    ego_vel = extract_speed_from_polyline(polyline_features, ego_start, ego_end, ego_traj)
    num_frames = len(ego_traj)

    # 计算ego heading
    dx = np.diff(ego_traj[:, 0], prepend=ego_traj[0, 0])
    dy = np.diff(ego_traj[:, 1], prepend=ego_traj[0, 1])
    ego_heading = np.arctan2(dy, dx)

    ego_data = np.stack([
        ego_traj[:, 0], ego_traj[:, 1],
        ego_heading,
        ego_vel[:, 0], ego_vel[:, 1]
    ], axis=1)

    # 提取agents轨迹 (ID >= 1)
    agent_ids = sorted([k for k in traj_indices.keys() if k > 0])
    agents_data = []
    agent_valid_ids = []

    for aid in agent_ids:
        start, end = traj_indices[aid]
        traj = extract_trajectory_from_polyline(polyline_features, start, end)

        if len(traj) != num_frames:
            continue

        vel = extract_speed_from_polyline(polyline_features, start, end, traj)

        dx = np.diff(traj[:, 0], prepend=traj[0, 0])
        dy = np.diff(traj[:, 1], prepend=traj[0, 1])
        heading = np.arctan2(dy, dx)

        agent_data = np.stack([
            traj[:, 0], traj[:, 1],
            heading,
            vel[:, 0], vel[:, 1]
        ], axis=1)

        agents_data.append(agent_data)
        agent_valid_ids.append(aid)

    return {
        'ego_data': ego_data,
        'agents_data': agents_data,
        'agent_ids': agent_valid_ids,
        'num_frames': num_frames,
        'file_name': Path(file_path).name
    }


# ============================================================
# TTC 计算
# ============================================================

def compute_ttc_1d(distance: float, relative_velocity: float) -> float:
    """
    计算一维TTC

    Args:
        distance: 距离
        relative_velocity: 相对速度（远离为正，接近为负）

    Returns:
        TTC值，inf表示不会碰撞
    """
    if relative_velocity >= 0:
        return np.inf
    return abs(distance / relative_velocity)


def compute_longitudinal_lateral_ttc(
    agent_x: float,
    agent_y: float,
    agent_vx: float,
    agent_vy: float,
    ego_vx: float = 0.0,
    ego_vy: float = 0.0,
    consider_size: bool = False,
    agent_length: float = 4.0,
    agent_width: float = 2.0,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    buffer: float = 0.0
) -> Tuple[float, float]:
    """
    计算单帧的纵向TTC和横向TTC

    坐标系：ego坐标系，X轴指向前方，Y轴指向左侧
    """
    rel_vx = agent_vx - ego_vx
    rel_vy = agent_vy - ego_vy

    if consider_size:
        ego_front = ego_length / 2 + buffer
        ego_rear = -ego_length / 2 - buffer
        ego_left = ego_width / 2 + buffer
        ego_right = -ego_width / 2 - buffer

        agent_front = agent_x + agent_length / 2
        agent_rear = agent_x - agent_length / 2
        agent_left = agent_y + agent_width / 2
        agent_right = agent_y - agent_width / 2

        if agent_rear > ego_front:
            longitudinal_dist = agent_rear - ego_front
        elif agent_front < ego_rear:
            longitudinal_dist = ego_rear - agent_front
        else:
            longitudinal_dist = 0.0

        if agent_right > ego_left:
            lateral_dist = agent_right - ego_left
        elif agent_left < ego_right:
            lateral_dist = ego_right - agent_left
        else:
            lateral_dist = 0.0
    else:
        longitudinal_dist = agent_x
        lateral_dist = agent_y

    # 纵向TTC
    if longitudinal_dist > 0:
        longitudinal_ttc = compute_ttc_1d(longitudinal_dist, rel_vx)
    elif longitudinal_dist < 0:
        longitudinal_ttc = compute_ttc_1d(abs(longitudinal_dist), -rel_vx)
    else:
        longitudinal_ttc = 0.0

    # 横向TTC
    if lateral_dist > 0:
        lateral_ttc = compute_ttc_1d(lateral_dist, rel_vy)
    elif lateral_dist < 0:
        lateral_ttc = compute_ttc_1d(abs(lateral_dist), -rel_vy)
    else:
        lateral_ttc = 0.0

    return longitudinal_ttc, lateral_ttc


def compute_agent_ttc_sequence(
    agent_data: np.ndarray,
    ego_data: np.ndarray,
    config: TTCConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单个agent在所有帧的TTC序列

    Args:
        agent_data: shape=(num_frames, 5) [x, y, heading, vx, vy]
        ego_data: shape=(num_frames, 5)
        config: TTC配置

    Returns:
        (longitudinal_ttc_seq, lateral_ttc_seq)
    """
    num_frames = agent_data.shape[0]
    lon_ttc_seq = np.zeros(num_frames)
    lat_ttc_seq = np.zeros(num_frames)

    for t in range(num_frames):
        lon_ttc, lat_ttc = compute_longitudinal_lateral_ttc(
            agent_data[t, 0], agent_data[t, 1],
            agent_data[t, 3], agent_data[t, 4],
            ego_data[t, 3], ego_data[t, 4],
            consider_size=config.CONSIDER_VEHICLE_SIZE,
            buffer=config.VEHICLE_SIZE_BUFFER
        )
        lon_ttc_seq[t] = lon_ttc
        lat_ttc_seq[t] = lat_ttc

    return lon_ttc_seq, lat_ttc_seq


# ============================================================
# 筛选逻辑
# ============================================================

def is_monotonic_increasing(arr: np.ndarray) -> bool:
    """判断数组是否非严格单调递增"""
    if len(arr) <= 1:
        return True
    if np.all(np.isinf(arr)):
        return True
    return np.all(np.diff(arr) >= 0)


def check_dimension_risk(ttc_seq: np.ndarray, threshold: float) -> bool:
    """
    检查某个维度的TTC是否存在风险

    筛选逻辑：
    - 逻辑一：所有TTC > 阈值 → 无风险
    - 逻辑二：TTC非严格单调递增 → 无风险

    Returns:
        True表示存在风险，False表示无风险
    """
    all_above_threshold = np.all(ttc_seq > threshold)
    monotonic_increasing = is_monotonic_increasing(ttc_seq)
    no_risk = all_above_threshold or monotonic_increasing
    return not no_risk


# ============================================================
# 关键Agent选择逻辑
# ============================================================

class PairSelectionConfig:
    """关键agent pair选择配置"""

    # 静止车辆速度阈值 (m/s)
    # 低于此速度认为是静止车辆，不作为研究对象
    STATIC_SPEED_THRESHOLD: float = 0.5

    # Ego静止速度阈值 (m/s)
    # 低于此速度认为ego在我们的时间窗口内接近静止，跳过该场景
    # 原因：TTC筛选只检查agent速度，不检查ego速度；ego静止时danger_scale
    # 条件编码器输出与训练分布严重不匹配，生成效果差
    EGO_STATIC_SPEED_THRESHOLD: float = 0.5

    # GT未来轨迹危险度阈值（tent score 标定）
    # 低于此值说明ego在GT未来已成功规避危险，属于安全样本，不应加入训练集
    # 原因：TTC筛选基于历史接近，不保证GT未来也危险；大量样本GT danger接近0
    #
    # ⚠️ tent score 说明（v4.0+）：
    #   - 旧 score：碰撞=100，碰撞场景得分极高 → 阈值需设 15.0
    #   - tent score：碰撞=0，near-miss(3m)=3.0，最高约8 → 阈值应为 0.5
    #   使用旧阈值 15.0 会导致 tent score 下所有样本被拒绝。
    MIN_GT_FUTURE_DANGER: float = 0.5

    # 前方区域定义 (m)
    # X > 此值认为是前方agent
    FRONT_REGION_THRESHOLD: float = 0.0

    # TTC计算时使用的帧范围
    # 选择最后N帧的最小TTC作为紧急程度指标
    TTC_EVAL_FRAMES: int = 10

    # 对向车道过滤：heading angle threshold (deg) and minimum front distance (m)
    # In ego-centric frame ego heading≈0; opposite-lane oncoming agents have heading≈π.
    # Agents with |rel_heading| > ONCOMING_HEADING_THRESHOLD_DEG AND x > ONCOMING_MIN_FRONT_DIST
    # are excluded — they are normal oncoming traffic, not same-lane danger scenarios.
    ONCOMING_HEADING_THRESHOLD_DEG: float = 150.0
    ONCOMING_MIN_FRONT_DIST: float = 3.0


def compute_agent_metrics(result: Dict, ego_data: np.ndarray, config: PairSelectionConfig) -> Dict:
    """
    计算用于agent选择的各项指标

    Args:
        result: 单个agent的TTC计算结果
        ego_data: ego车辆数据
        config: pair选择配置

    Returns:
        包含各项指标的字典
    """
    agent_data = result['agent_data']
    lon_ttc = result['lon_ttc']
    lat_ttc = result['lat_ttc']

    # 计算agent平均速度
    vx_mean = np.mean(np.abs(agent_data[:, 3]))
    vy_mean = np.mean(np.abs(agent_data[:, 4]))
    total_speed = np.sqrt(vx_mean**2 + vy_mean**2)
    is_static = total_speed < config.STATIC_SPEED_THRESHOLD

    # 计算agent位置（最后一帧）
    final_x = agent_data[-1, 0]
    final_y = agent_data[-1, 1]
    is_front = final_x > config.FRONT_REGION_THRESHOLD

    # 对向车道检测：ego-centric frame中ego heading≈0，对向来车heading≈π
    agent_heading_last = float(agent_data[-1, 2])
    rel_heading_abs = abs(agent_heading_last) % (2 * np.pi)
    if rel_heading_abs > np.pi:
        rel_heading_abs = 2 * np.pi - rel_heading_abs  # normalize to [0, π]
    is_oncoming_opposite = (
        rel_heading_abs > np.deg2rad(config.ONCOMING_HEADING_THRESHOLD_DEG)
        and final_x > config.ONCOMING_MIN_FRONT_DIST
    )

    # 计算最小TTC（在评估帧范围内）
    eval_start = max(0, len(lon_ttc) - config.TTC_EVAL_FRAMES)
    lon_ttc_eval = lon_ttc[eval_start:]
    lat_ttc_eval = lat_ttc[eval_start:]

    lon_ttc_min = np.min(lon_ttc_eval[np.isfinite(lon_ttc_eval)]) if np.any(np.isfinite(lon_ttc_eval)) else np.inf
    lat_ttc_min = np.min(lat_ttc_eval[np.isfinite(lat_ttc_eval)]) if np.any(np.isfinite(lat_ttc_eval)) else np.inf

    # 综合TTC指标（取两个维度的最小值）
    combined_ttc = min(lon_ttc_min, lat_ttc_min)

    # 双重风险标志
    dual_risk = result['lon_risk'] and result['lat_risk']

    return {
        'agent_id': result['agent_id'],
        'is_static': is_static,
        'is_front': is_front,
        'is_oncoming_opposite': is_oncoming_opposite,
        'lon_ttc_min': lon_ttc_min,
        'lat_ttc_min': lat_ttc_min,
        'combined_ttc': combined_ttc,
        'dual_risk': dual_risk,
        'total_speed': total_speed,
        'final_x': final_x,
        'final_y': final_y
    }


def _compute_gt_future_danger(
    scenario_result: Dict,
    original_df: pd.DataFrame,
    selected_agent_data: np.ndarray,
    time_window_config: 'TimeWindowConfig' = None,
    dt: float = 0.1
) -> float:
    """
    计算 ego GT未来轨迹对选中agent的危险分数，用于数据集构建时过滤低质量样本。

    本函数读取的原始 GROUND_TRUTH 是逐帧位移增量 (dx, dy)，下面自行 cumsum 还原绝对坐标。
    （注意区分：save_selected_pair_data 落盘的 danger_pair['ego_future_gt'] 已是**绝对位置** (T,5)，
     增量仅是这里的源数据形态。）
    agent_future 无法从已有数据获取，改用恒速外推作为近似。

    ⚠️ 已知近似误差：agent 实际会制动或转向，恒速外推会高估危险（即预测碰撞但
    GT 实际是 near-miss）。为此增加备选判断：若恒速外推导致碰撞（min_dist < 2m），
    改用 min_dist ∈ [1.0, 6.0m] 作为通过条件，而非直接拒绝。

    Returns:
        outcome_danger['total_score']，计算失败时返回 0.0
    """
    if not DANGER_SCORE_AVAILABLE:
        return 0.0

    if time_window_config is None:
        time_window_config = TimeWindowConfig()

    try:
        ground_truth = original_df.iloc[0].get('GROUND_TRUTH') if 'GROUND_TRUTH' in original_df.columns else None
        if ground_truth is None:
            return 0.0

        gt_2d = ground_truth[0] if ground_truth.ndim == 3 else ground_truth
        n_future = time_window_config.NUM_FUTURE_POSES
        if len(gt_2d) < n_future:
            pad = np.zeros((n_future - len(gt_2d), 2))
            gt_2d = np.concatenate([gt_2d, pad], axis=0)
        else:
            gt_2d = gt_2d[:n_future]

        # ego_history 取最后 NUM_PAST_POSES 帧
        ego_data = scenario_result['ego_data']
        n_past = time_window_config.NUM_PAST_POSES
        ego_hist = ego_data[-n_past:] if len(ego_data) >= n_past else ego_data

        # 还原 ego 绝对位置（增量累加）
        start_pos = ego_hist[-1, :2]
        ego_future_abs = np.zeros((n_future, 5))
        ego_future_abs[0, :2] = start_pos + gt_2d[0]
        for t in range(1, n_future):
            ego_future_abs[t, :2] = ego_future_abs[t - 1, :2] + gt_2d[t]

        dx = np.diff(ego_future_abs[:, 0], prepend=ego_hist[-1, 0])
        dy = np.diff(ego_future_abs[:, 1], prepend=ego_hist[-1, 1])
        ego_future_abs[:, 2] = np.arctan2(dy, dx)
        ego_future_abs[:, 3] = dx / dt
        ego_future_abs[:, 4] = dy / dt

        # agent 历史取最后 NUM_PAST_POSES 帧
        agent_hist = selected_agent_data[-n_past:] if len(selected_agent_data) >= n_past else selected_agent_data

        # 恒速外推 agent 未来轨迹
        last = agent_hist[-1]  # [x, y, heading, vx, vy]
        agent_future = np.zeros((n_future, 5))
        for t in range(n_future):
            agent_future[t, 0] = last[0] + last[3] * (t + 1) * dt
            agent_future[t, 1] = last[1] + last[4] * (t + 1) * dt
            agent_future[t, 2] = last[2]
            agent_future[t, 3] = last[3]
            agent_future[t, 4] = last[4]

        # 计算 min_dist（用于备选判断）
        ego_xy   = ego_future_abs[:, :2]
        agent_xy = agent_future[:, :2]
        min_dist = float(np.linalg.norm(ego_xy - agent_xy, axis=1).min())

        # 若恒速外推导致碰撞（min_dist < 2m），agent 实际大概率会制动/转向，
        # GT 结果更可能是 near-miss 而非真实碰撞。直接接受此类样本（返回 1.0 通过阈值）。
        # 极端穿透（< 0.5m）说明两车轨迹完全交叉，大概率真实碰撞，予以拒绝。
        if min_dist < 2.0:
            return 1.0 if min_dist >= 0.5 else 0.0

        result = _compute_danger_score(
            ego_history=ego_hist,
            ego_future=ego_future_abs,
            agent_history=agent_hist,
            agent_future=agent_future,
            dt=dt
        )
        return float(result['outcome_danger']['total_score'])

    except Exception:
        return 0.0


def select_key_agent_pair(
    scenario_result: Dict,
    ttc_config: TTCConfig,
    pair_config: PairSelectionConfig = None
) -> Dict:
    """
    从危险场景中选择关键的研究对象pair (ego + one agent)

    选择优先级：
    1. 双重风险（纵向+横向）的非静止前方agent，按combined_ttc排序
    2. 如果没有双重风险，选择前方agent中纵向TTC最小的非静止agent
    3. 如果前方没有合适的agent，选择后方agent中TTC最小的非静止agent
    4. 如果全部是静止agent，返回None

    Args:
        scenario_result: 场景处理结果
        ttc_config: TTC配置
        pair_config: pair选择配置

    Returns:
        选择结果字典，包含选中的agent信息和选择原因
    """
    if pair_config is None:
        pair_config = PairSelectionConfig()

    ego_data = scenario_result['ego_data']
    agent_results = scenario_result['agent_results']

    # 跳过 ego 近似静止的场景（在评估窗口最后 TTC_EVAL_FRAMES 帧内）
    eval_start = max(0, len(ego_data) - pair_config.TTC_EVAL_FRAMES)
    ego_speed = float(np.mean(np.linalg.norm(ego_data[eval_start:, 3:5], axis=1)))
    if ego_speed < pair_config.EGO_STATIC_SPEED_THRESHOLD:
        return {
            'selected': False,
            'reason': f'Ego nearly stationary (mean_speed={ego_speed:.3f} m/s < {pair_config.EGO_STATIC_SPEED_THRESHOLD} m/s)'
        }

    # 只考虑被保留的危险agent
    kept_results = [r for r in agent_results if r['keep']]

    if len(kept_results) == 0:
        return {
            'selected': False,
            'reason': 'No risky agents in scenario'
        }

    # 计算每个agent的指标
    metrics_list = []
    for result in kept_results:
        metrics = compute_agent_metrics(result, ego_data, pair_config)
        metrics['original_result'] = result
        metrics_list.append(metrics)

    # 排除静止agent
    non_static_metrics = [m for m in metrics_list if not m['is_static']]

    if len(non_static_metrics) == 0:
        return {
            'selected': False,
            'reason': 'All risky agents are static (no dynamic interaction)'
        }

    # 排除对向车道来车：这类情况下 ego 和 agent 背离行驶，不属于训练目标的危险交互
    same_lane_metrics = [m for m in non_static_metrics if not m['is_oncoming_opposite']]
    if len(same_lane_metrics) > 0:
        non_static_metrics = same_lane_metrics
    # else: 如果全部都是对向来车，fallback 到不过滤（保留覆盖率）

    # 按优先级选择
    # 优先级1：双重风险 + 前方 + 非静止
    dual_risk_front = [m for m in non_static_metrics if m['dual_risk'] and m['is_front']]
    if len(dual_risk_front) > 0:
        # 按combined_ttc排序，选择最小的（最紧急）
        dual_risk_front.sort(key=lambda x: x['combined_ttc'])
        selected = dual_risk_front[0]
        return {
            'selected': True,
            'agent_id': selected['agent_id'],
            'selection_reason': 'Dual_risk_front_agent',
            'metrics': selected,
            'agent_data': selected['original_result']['agent_data'],
            'lon_ttc': selected['original_result']['lon_ttc'],
            'lat_ttc': selected['original_result']['lat_ttc']
        }

    # 优先级2：前方 + 非静止（纵向风险优先）
    front_agents = [m for m in non_static_metrics if m['is_front']]
    if len(front_agents) > 0:
        # 按纵向TTC排序
        front_agents.sort(key=lambda x: x['lon_ttc_min'])
        selected = front_agents[0]
        return {
            'selected': True,
            'agent_id': selected['agent_id'],
            'selection_reason': 'Front_agent_min_lon_ttc',
            'metrics': selected,
            'agent_data': selected['original_result']['agent_data'],
            'lon_ttc': selected['original_result']['lon_ttc'],
            'lat_ttc': selected['original_result']['lat_ttc']
        }

    # 优先级3：后方 + 非静止
    rear_agents = [m for m in non_static_metrics if not m['is_front']]
    if len(rear_agents) > 0:
        # 按combined_ttc排序
        rear_agents.sort(key=lambda x: x['combined_ttc'])
        selected = rear_agents[0]
        return {
            'selected': True,
            'agent_id': selected['agent_id'],
            'selection_reason': 'Rear_agent_min_ttc',
            'metrics': selected,
            'agent_data': selected['original_result']['agent_data'],
            'lon_ttc': selected['original_result']['lon_ttc'],
            'lat_ttc': selected['original_result']['lat_ttc']
        }

    # 不应该到达这里，因为non_static_metrics不为空
    return {
        'selected': False,
        'reason': 'Unexpected selection failure'
    }


def print_pair_selection_result(selection_result: Dict):
    """打印关键pair选择结果"""
    print('\n' + '='*100)
    print('Key Agent Pair Selection Result')
    print('='*100)

    if not selection_result['selected']:
        print(f"Selection Status: FAILED")
        print(f"Reason: {selection_result['reason']}")
        return

    print(f"Selection Status: SUCCESS")
    print(f"Selected Agent ID: {selection_result['agent_id']}")
    print(f"Selection Reason: {selection_result['selection_reason']}")

    metrics = selection_result['metrics']
    print(f"\nAgent Metrics:")
    print(f"  Position: X={metrics['final_x']:.2f}m, Y={metrics['final_y']:.2f}m")
    print(f"  Is Front: {metrics['is_front']}")
    print(f"  Is Static: {metrics['is_static']}")
    print(f"  Total Speed: {metrics['total_speed']:.2f} m/s")
    print(f"  Dual Risk: {metrics['dual_risk']}")
    print(f"  Min Lon TTC: {metrics['lon_ttc_min']:.2f}s")
    print(f"  Min Lat TTC: {metrics['lat_ttc_min']:.2f}s")
    print(f"  Combined TTC: {metrics['combined_ttc']:.2f}s")


# ============================================================
# Agent Future Extraction from NuPlan
# ============================================================

def extract_agent_futures_from_nuplan(
    scenario_token: str,
    agent_positions: Dict[int, Tuple[float, float]] = None,
    agent_histories: Dict[int, np.ndarray] = None,
    ego_absolute_position: Tuple[float, float] = None,
    future_time_horizon: float = 3.0,
    num_future_poses: int = 30,
    position_match_threshold: float = 10.0
) -> Dict[int, np.ndarray]:
    """
    从NuPlan原始数据中提取所有agent的未来轨迹

    通过历史轨迹匹配建立整数ID和track_token的对应关系

    Args:
        scenario_token: 场景token
        agent_positions: 字典 {agent_id: (x, y)} agent的最后观测位置（相对坐标）
        agent_histories: 字典 {agent_id: history_array} agent的历史轨迹，shape=(T, 5)
        ego_absolute_position: ego在全局坐标系中的绝对位置 (x, y)
        future_time_horizon: 未来时间范围（秒）
        num_future_poses: 未来帧数
        position_match_threshold: 位置匹配阈值（米）

    Returns:
        字典: {agent_id: future_trajectory}
        future_trajectory shape: (num_future_poses+1, 3) -> [x, y, heading]
    """
    if not NUPLAN_AVAILABLE:
        print("[WARNING] NuPlan modules not available, cannot extract agent futures")
        return {}

    try:
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
        from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
        from nuplan.planning.utils.multithreading.worker_sequential import Sequential
        import torch

        # 数据路径配置
        data_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh"
        map_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps"
        sensor_root = "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/sensor_blobs"

        # 创建scenario builder
        builder = NuPlanScenarioBuilder(
            data_root=data_root,
            map_root=map_root,
            sensor_root=sensor_root,
            db_files=None,
            map_version="nuplan-maps-v1.0"
        )

        # 获取特定场景
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=[scenario_token],
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=1,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False
        )
        # 使用Sequential执行器
        worker = Sequential()
        scenarios = builder.get_scenarios(scenario_filter, worker)

        if len(scenarios) == 0:
            print(f"[WARNING] Scenario {scenario_token} not found")
            return {}

        scenario = scenarios[0]

        # 获取初始tracked objects (可能只有少量对象)
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

        # 获取过去帧的 tracked objects 以获取更多对象
        # 很多对象可能在当前时刻离开了，但在历史中存在
        past_tracked_objects_list = list(scenario.get_past_tracked_objects(
            iteration=0,
            time_horizon=2.0,
            num_samples=20
        ))

        # 从过去帧收集所有唯一的 track_tokens
        all_track_tokens = set()
        for obs_list in past_tracked_objects_list:
            for obj in obs_list.tracked_objects:
                all_track_tokens.add(obj.track_token)

        # 也添加当前帧的
        for obj in present_tracked_objects:
            all_track_tokens.add(obj.track_token)

        # 获取ego的绝对位置和heading（如果未提供）
        if ego_absolute_position is None:
            ego_state = scenario.initial_ego_state
            ego_absolute_position = (ego_state.rear_axle.x, ego_state.rear_axle.y)
            ego_heading = ego_state.rear_axle.heading
        else:
            # 如果提供了绝对位置，但没有 heading，需要从 scenario 获取
            ego_state = scenario.initial_ego_state
            ego_heading = ego_state.rear_axle.heading

        # 获取未来tracked objects
        future_tracked_objects_list = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=future_time_horizon,
                num_samples=num_future_poses
            )
        ]

        # 第一步：从过去帧建立 track_token 到最新位置的映射
        track_latest_positions = {}  # track_token -> (latest_time, x, y, heading)

        # 遍历所有过去帧，记录每个对象的最新位置
        for i, obs_list in enumerate(past_tracked_objects_list):
            for obj in obs_list.tracked_objects:
                track_token = obj.track_token
                # 更新最新位置（后面的帧更接近当前时刻）
                track_latest_positions[track_token] = (i, obj.center.x, obj.center.y, obj.center.heading)

        # 也添加当前帧的
        for obj in present_tracked_objects:
            track_latest_positions[obj.track_token] = (len(past_tracked_objects_list), obj.center.x, obj.center.y, obj.center.heading)

        def global_to_relative(x, y, heading, ego_x, ego_y, ego_heading):
            """将全局坐标转换为ego-relative坐标"""
            # 平移：减去ego位置
            dx = x - ego_x
            dy = y - ego_y
            # 旋转：绕ego heading旋转
            cos_h = np.cos(-ego_heading)
            sin_h = np.sin(-ego_heading)
            rel_x = dx * cos_h - dy * sin_h
            rel_y = dx * sin_h + dy * cos_h
            rel_heading = heading - ego_heading
            return rel_x, rel_y, rel_heading

        # 第2.5步：建立 track_token 到历史轨迹的映射（用于轨迹匹配）
        track_histories = {}  # track_token -> [(rel_x, rel_y), ...]

        # 按时间顺序遍历过去帧
        for obs_list in past_tracked_objects_list:
            for obj in obs_list.tracked_objects:
                track_token = obj.track_token
                rel_x, rel_y, _ = global_to_relative(
                    obj.center.x, obj.center.y, obj.center.heading,
                    ego_absolute_position[0], ego_absolute_position[1], ego_heading
                )
                if track_token not in track_histories:
                    track_histories[track_token] = []
                track_histories[track_token].append((rel_x, rel_y))

        # 添加当前帧
        for obj in present_tracked_objects:
            track_token = obj.track_token
            rel_x, rel_y, _ = global_to_relative(
                obj.center.x, obj.center.y, obj.center.heading,
                ego_absolute_position[0], ego_absolute_position[1], ego_heading
            )
            if track_token not in track_histories:
                track_histories[track_token] = []
            track_histories[track_token].append((rel_x, rel_y))

        # 第三步：建立 track_token 到未来轨迹的映射
        track_futures = {}

        # 使用所有 track_latest_positions 初始化
        for track_token, (time_idx, abs_x, abs_y, abs_heading) in track_latest_positions.items():
            rel_x, rel_y, rel_heading = global_to_relative(
                abs_x, abs_y, abs_heading,
                ego_absolute_position[0], ego_absolute_position[1], ego_heading
            )
            track_futures[track_token] = {
                'positions': [(rel_x, rel_y, rel_heading)],
                'initial_pos_relative': (rel_x, rel_y),
                'initial_pos_absolute': (abs_x, abs_y)
            }

        # 收集每个agent的未来位置（相对坐标）
        for obs in future_tracked_objects_list:
            for obj in obs:
                track_token = obj.track_token
                if track_token in track_futures:
                    # 正确的坐标转换
                    rel_x, rel_y, rel_heading = global_to_relative(
                        obj.center.x, obj.center.y, obj.center.heading,
                        ego_absolute_position[0], ego_absolute_position[1], ego_heading
                    )
                    track_futures[track_token]['positions'].append((rel_x, rel_y, rel_heading))

        # 第三步：如果提供了agent_histories，进行历史轨迹匹配；否则使用单点匹配
        result = {}

        def compute_trajectory_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
            """
            计算两条轨迹的平均距离

            Args:
                traj1: shape=(T1, 2) 的轨迹
                traj2: shape=(T2, 2) 的轨迹

            Returns:
                平均欧氏距离
            """
            # 对齐长度，使用最后 min_len 帧
            min_len = min(len(traj1), len(traj2))
            if min_len == 0:
                return float('inf')

            traj1_aligned = traj1[-min_len:]
            traj2_aligned = traj2[-min_len:]

            # 计算平均欧氏距离
            dists = np.sqrt(np.sum((traj1_aligned - traj2_aligned) ** 2, axis=1))
            return np.mean(dists)

        if agent_histories:
            # 使用历史轨迹匹配（更鲁棒）
            for agent_id, agent_hist in agent_histories.items():
                best_token = None
                best_dist = float('inf')

                # 只使用 x, y 坐标
                agent_xy = agent_hist[:, :2]

                for track_token, track_xy_list in track_histories.items():
                    track_xy = np.array(track_xy_list)

                    # 计算轨迹距离
                    dist = compute_trajectory_distance(agent_xy, track_xy)

                    if dist < best_dist:
                        best_dist = dist
                        best_token = track_token

                if best_token is not None and best_dist < position_match_threshold:
                    if best_token in track_futures:
                        positions = track_futures[best_token]['positions']
                        if len(positions) > 0:
                            result[agent_id] = np.array(positions)

        elif agent_positions:
            # 降级为单点匹配
            for agent_id, (ax, ay) in agent_positions.items():
                # 找到最近的tracked object
                best_token = None
                best_dist = float('inf')

                for track_token, data in track_futures.items():
                    tx, ty = data['initial_pos_relative']
                    dist = np.sqrt((ax - tx)**2 + (ay - ty)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_token = track_token

                if best_token is not None and best_dist < position_match_threshold:
                    positions = track_futures[best_token]['positions']
                    if len(positions) > 0:
                        result[agent_id] = np.array(positions)
        else:
            # 没有提供位置，返回所有track_futures（使用track_token作为键）
            for track_token, data in track_futures.items():
                positions = data['positions']
                if len(positions) > 0:
                    result[track_token] = np.array(positions)

        return result

    except Exception as e:
        print(f"[ERROR] Failed to extract agent futures: {e}")
        import traceback
        traceback.print_exc()
        return {}


def slice_trajectory_by_time_window(
    full_trajectory: np.ndarray,
    time_window_config: TimeWindowConfig = None,
    is_history: bool = True
) -> np.ndarray:
    """
    根据时间窗口切片轨迹数据

    Args:
        full_trajectory: 完整轨迹数组 shape=(num_frames, dim)
        time_window_config: 时间窗口配置
        is_history: True表示历史轨迹，False表示未来轨迹

    Returns:
        切片后的轨迹数组
    """
    if time_window_config is None:
        time_window_config = TimeWindowConfig()

    num_frames = len(full_trajectory)
    target_frames = time_window_config.NUM_PAST_POSES if is_history else time_window_config.NUM_FUTURE_POSES

    if is_history:
        # 历史轨迹：取最后N帧
        if num_frames >= target_frames:
            return full_trajectory[-target_frames:]
        else:
            # 如果帧数不足，用零填充开头
            pad_frames = target_frames - num_frames
            padding = np.zeros((pad_frames, full_trajectory.shape[1]), dtype=full_trajectory.dtype)
            return np.concatenate([padding, full_trajectory], axis=0)
    else:
        # 未来轨迹：取前N帧
        if num_frames >= target_frames:
            return full_trajectory[:target_frames]
        else:
            # 如果帧数不足，用零填充末尾
            pad_frames = target_frames - num_frames
            padding = np.zeros((pad_frames, full_trajectory.shape[1]), dtype=full_trajectory.dtype)
            return np.concatenate([full_trajectory, padding], axis=0)


def get_current_frame_index(
    full_trajectory: np.ndarray,
    time_window_config: TimeWindowConfig = None
) -> int:
    """
    获取当前帧索引（历史轨迹的最后一帧）

    对于历史轨迹，当前帧是历史轨迹的最后一帧
    对于未来轨迹，第0帧对应当前时刻之后的下一帧

    Args:
        full_trajectory: 完整轨迹数组
        time_window_config: 时间窗口配置

    Returns:
        当前帧在完整轨迹中的索引
    """
    if time_window_config is None:
        time_window_config = TimeWindowConfig()

    # 当前帧 = 历史轨迹的最后一帧
    return len(full_trajectory) - 1


def save_selected_pair_data(
    scenario_result: Dict,
    selection_result: Dict,
    original_df: pd.DataFrame,
    output_dir: str,
    scenario_name: str,
    agent_futures: Dict[int, np.ndarray] = None,
    time_window_config: TimeWindowConfig = None
) -> str:
    """
    Save the selected pair data to a new pkl file for diffusion training

    Args:
        scenario_result: Full scenario processing result
        selection_result: Key agent pair selection result
        original_df: Original DataFrame from pkl file (contains ground truth)
        output_dir: Output directory path
        scenario_name: Original scenario filename
        agent_futures: 字典 {agent_id: future_trajectory} 从NuPlan提取
        time_window_config: 时间窗口配置（控制历史/未来帧数）

    Returns:
        Path to saved file
    """
    if not selection_result['selected']:
        return None

    if time_window_config is None:
        time_window_config = TimeWindowConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract key components
    ego_data_full = scenario_result['ego_data']
    selected_agent_id = selection_result['agent_id']
    selected_agent_data_full = selection_result['agent_data']

    # 切片历史轨迹（取最后NUM_PAST_POSES帧）
    ego_history = slice_trajectory_by_time_window(ego_data_full, time_window_config, is_history=True)
    key_agent_history = slice_trajectory_by_time_window(selected_agent_data_full, time_window_config, is_history=True)

    # Get boundary agents (all other kept agents except the selected one)
    boundary_agents = []
    for agent_result in scenario_result['agent_results']:
        if agent_result['keep'] and agent_result['agent_id'] != selected_agent_id:
            agent_id = agent_result['agent_id']
            agent_future_full = None
            if agent_futures and agent_id in agent_futures:
                agent_future_full = agent_futures[agent_id]

            # 切片历史轨迹
            agent_history = slice_trajectory_by_time_window(agent_result['agent_data'], time_window_config, is_history=True)

            # 切片未来轨迹
            if agent_future_full is not None:
                agent_future = slice_trajectory_by_time_window(agent_future_full, time_window_config, is_history=False)
            else:
                agent_future = None

            boundary_agents.append({
                'agent_id': agent_id,
                'agent_data': agent_history,  # 切片后的历史轨迹
                'agent_future': agent_future,  # 切片后的未来轨迹
                'lon_ttc': agent_result['lon_ttc'],
                'lat_ttc': agent_result['lat_ttc'],
                'lon_risk': agent_result['lon_risk'],
                'lat_risk': agent_result['lat_risk']
            })

    # Get ground truth from original data
    ground_truth = original_df.iloc[0]['GROUND_TRUTH'] if 'GROUND_TRUTH' in original_df.columns else None
    target_future_frames = time_window_config.NUM_FUTURE_POSES

    # ego GT 未来必须有满 NUM_FUTURE_POSES 帧的真实数据；无 / 不足则**丢弃该 pair**。
    # ⚠️ 旧逻辑对短未来零 padding，cumsum 后会制造「静止尾巴」伪样本（已知 bug）——固定
    # 6s horizon 模型用不了短 pair，丢弃最干净。
    if ground_truth is None:
        return None
    ego_future_gt_2d = ground_truth[0] if ground_truth.ndim == 3 else ground_truth
    if len(ego_future_gt_2d) < target_future_frames:
        return None
    ego_future_gt_2d = ego_future_gt_2d[:target_future_frames]

    # 补全ego_future_gt为 (T,5) [x,y,heading,vx,vy] 绝对轨迹，与 key_agent_future 对齐。
    # ⚠️ GROUND_TRUTH(=ego_future_gt_2d) 是逐帧位移**增量**，必须先 cumsum 还原绝对位置，
    # 否则下游 loader 会把增量当位置再差分一次（double-diff），ego 未来被压成「停着」。
    _dt = 0.1  # NuPlan 10Hz
    ego_future_abs_xy = ego_history[-1, :2] + np.cumsum(ego_future_gt_2d, axis=0)
    ego_future_gt_full = np.zeros((target_future_frames, 5))
    ego_future_gt_full[:, :2] = ego_future_abs_xy

    # heading / 速度由绝对位置差分得到（prepend 历史末位），即真实逐帧速度（非历史恒定继承）
    dx = np.diff(ego_future_abs_xy[:, 0], prepend=ego_history[-1, 0])
    dy = np.diff(ego_future_abs_xy[:, 1], prepend=ego_history[-1, 1])
    ego_future_gt_full[:, 2] = np.arctan2(dy, dx)
    ego_future_gt_full[:, 3] = dx / _dt  # vx
    ego_future_gt_full[:, 4] = dy / _dt  # vy

    ground_truth = ego_future_gt_full

    # 提取 key_agent 未来轨迹 → (target_future_frames, 5) [x,y,heading,vx,vy]，固定 ego 锚点帧绝对。
    # 优先用 pkl 内置 AGENT_FUTURES（预处理已存，batch 自包含，无需从 NuPlan 重抽）：
    #   AGENT_FUTURES=(num_agents,T,3)[x,y,heading]，经 agent_future_process 用 initial_ego_state 做锚点，
    #   与 ego_future_gt 同在固定初始 ego 帧（已数值验证：ego/agent 逐帧距离自洽收敛）。
    # slot 用 key_agent 历史末位在 AGENT_HISTORIES 里位置匹配（鲁棒，不依赖 id↔slot 约定）。
    key_agent_future = None
    if 'AGENT_FUTURES' in original_df.columns:
        af_all = np.asarray(original_df.iloc[0]['AGENT_FUTURES'])  # (num_agents, T, 3)
        ah_all = (np.asarray(original_df.iloc[0]['AGENT_HISTORIES'])
                  if 'AGENT_HISTORIES' in original_df.columns else None)
        slot = -1
        if ah_all is not None and ah_all.shape[0] == af_all.shape[0]:
            dists = np.linalg.norm(ah_all[:, -1, :2] - key_agent_history[-1, :2][None, :], axis=1)
            cand = int(dists.argmin())
            if dists[cand] < 0.5:  # <0.5m 视为同一 agent；否则放弃 pkl 路径
                slot = cand
        if 0 <= slot < af_all.shape[0]:
            af = af_all[slot]  # (T,3)
            # 必须满 target_future_frames 帧真实数据；全零=padding agent → 丢弃
            if af.shape[0] >= target_future_frames and np.abs(af[:target_future_frames]).sum() > 0:
                af = af[:target_future_frames]
                kfut = np.zeros((target_future_frames, 5), dtype=np.float32)
                kfut[:, :2] = af[:, :2]   # 固定帧绝对 xy
                kfut[:, 2] = af[:, 2]      # heading
                # 速度由固定帧绝对位置差分(prepend 历史末位)得真实逐帧速度（与 ego 同法）
                kdx = np.diff(af[:, 0], prepend=key_agent_history[-1, 0])
                kdy = np.diff(af[:, 1], prepend=key_agent_history[-1, 1])
                kfut[:, 3] = kdx / _dt
                kfut[:, 4] = kdy / _dt
                key_agent_future = kfut

    # 回退：single-file 模式用 NuPlan 提取的 agent_futures。
    # key agent 未来必须满 NUM_FUTURE_POSES 帧真实数据；无 / 不足则**丢弃该 pair**
    # （短未来零 padding 会制造 agent「静止尾巴」伪样本——同 ego 的已知 bug）。
    if key_agent_future is None:
        key_agent_future_full = None
        if agent_futures and selected_agent_id in agent_futures:
            key_agent_future_full = agent_futures[selected_agent_id]
        if key_agent_future_full is None or len(key_agent_future_full) < target_future_frames:
            return None
        key_agent_future = slice_trajectory_by_time_window(key_agent_future_full, time_window_config, is_history=False)

    # Build the saved data structure
    saved_data = {
        # Scenario info
        'scenario_name': scenario_name,
        'num_frames_original': scenario_result['num_frames'],

        # Ego data: shape=(NUM_PAST_POSES, 5) [x, y, heading, vx, vy]
        'ego_history': ego_history,

        # Selected key agent data
        'key_agent_id': selected_agent_id,
        'key_agent_history': key_agent_history,  # shape=(NUM_PAST_POSES, 5)
        'key_agent_future': key_agent_future,  # shape=(NUM_FUTURE_POSES, 3) or None
        'key_agent_lon_ttc': selection_result['lon_ttc'],
        'key_agent_lat_ttc': selection_result['lat_ttc'],
        'selection_reason': selection_result['selection_reason'],
        'selection_metrics': selection_result['metrics'],

        # Boundary agents (for IDM simulation)
        'boundary_agents': boundary_agents,
        'num_boundary_agents': len(boundary_agents),

        # Ground truth future trajectory
        'ego_future_gt': ground_truth,  # shape=(NUM_FUTURE_POSES, 5)

        # Metadata
        'ttc_config': {
            'longitudinal_threshold': TTCConfig().LONGITUDINAL_TTC_THRESHOLD,
            'lateral_threshold': TTCConfig().LATERAL_TTC_THRESHOLD
        },

        # Time window metadata
        'time_window_config': {
            'discrete_size': time_window_config.DISCRETE_SIZE,
            'past_time_horizon': time_window_config.PAST_TIME_HORIZON,
            'future_time_horizon': time_window_config.FUTURE_TIME_HORIZON,
            'num_past_poses': time_window_config.NUM_PAST_POSES,
            'num_future_poses': time_window_config.NUM_FUTURE_POSES
        }
    }

    # Generate output filename
    output_filename = f"danger_pair_{scenario_name.replace('.pkl', '')}_agent{selected_agent_id}.pkl"
    output_file = output_path / output_filename

    # Save to pkl
    pd.to_pickle(saved_data, output_file)

    print(f"\nSaved selected pair data to: {output_file}")
    print(f"  - Ego history: {ego_history.shape} ({time_window_config.NUM_PAST_POSES} frames = {time_window_config.PAST_TIME_HORIZON}s)")
    print(f"  - Key agent (ID={selected_agent_id}) history: {key_agent_history.shape}")
    print(f"  - Key agent future: {key_agent_future.shape if key_agent_future is not None else 'None'}")
    print(f"  - Ego future GT: {ground_truth.shape if ground_truth is not None else 'None'}")
    print(f"  - Boundary agents: {len(boundary_agents)}")
    print(f"  - Time window: {time_window_config.PAST_TIME_HORIZON}s history, {time_window_config.FUTURE_TIME_HORIZON}s future")

    return str(output_file)


# ============================================================
# 主处理函数
# ============================================================

def process_scenario(file_path: str, config: TTCConfig) -> Dict:
    """
    处理单个场景，执行TTC筛选

    Args:
        file_path: 场景文件路径
        config: TTC配置

    Returns:
        处理结果字典
    """
    # 加载数据
    data = load_and_prepare_data(file_path)
    ego_data = data['ego_data']
    agents_data = data['agents_data']
    agent_ids = data['agent_ids']
    num_frames = data['num_frames']

    # 计算每个agent的TTC
    results = []
    for i, (aid, agent_data) in enumerate(zip(agent_ids, agents_data)):
        lon_ttc, lat_ttc = compute_agent_ttc_sequence(agent_data, ego_data, config)
        lon_risk = check_dimension_risk(lon_ttc, config.LONGITUDINAL_TTC_THRESHOLD)
        lat_risk = check_dimension_risk(lat_ttc, config.LATERAL_TTC_THRESHOLD)
        keep = lon_risk or lat_risk

        results.append({
            'agent_id': aid,
            'agent_data': agent_data,
            'lon_ttc': lon_ttc,
            'lat_ttc': lat_ttc,
            'lon_risk': lon_risk,
            'lat_risk': lat_risk,
            'keep': keep
        })

    return {
        'file_name': data['file_name'],
        'ego_data': ego_data,
        'num_frames': num_frames,
        'agent_results': results,
        'num_original': len(results),
        'num_kept': sum(1 for r in results if r['keep'])
    }


# ============================================================
# 输出函数
# ============================================================

def print_ego_table(ego_data: np.ndarray):
    """打印ego车辆数据表格"""
    print('\n' + '='*100)
    print('Ego Vehicle Data')
    print('='*100)

    df = pd.DataFrame({
        'Frame': range(1, len(ego_data) + 1),
        'X (m)': np.round(ego_data[:, 0], 4),
        'Y (m)': np.round(ego_data[:, 1], 4),
        'Heading (rad)': np.round(ego_data[:, 2], 4),
        'Vx (m/s)': np.round(ego_data[:, 3], 4),
        'Vy (m/s)': np.round(ego_data[:, 4], 4)
    })
    print(df.to_string(index=False))


def print_agent_ttc_table(result: Dict):
    """Print TTC details table for a single agent"""
    agent_id = result['agent_id']
    agent_data = result['agent_data']
    lon_ttc = result['lon_ttc']
    lat_ttc = result['lat_ttc']
    keep_status = 'Keep' if result['keep'] else 'Filter Out'

    print(f'\n' + '='*100)
    print(f'Agent {agent_id} - TTC Data [{keep_status}]')
    print(f'Longitudinal Risk: {"Yes" if result["lon_risk"] else "No"}, Lateral Risk: {"Yes" if result["lat_risk"] else "No"}')
    print('='*100)

    df = pd.DataFrame({
        'Frame': range(1, len(agent_data) + 1),
        'X (m)': np.round(agent_data[:, 0], 4),
        'Y (m)': np.round(agent_data[:, 1], 4),
        'Vx (m/s)': np.round(agent_data[:, 3], 4),
        'Vy (m/s)': np.round(agent_data[:, 4], 4),
        'Lon_TTC(s)': np.round(lon_ttc, 2),
        'Lat_TTC(s)': np.round(lat_ttc, 2)
    })
    print(df.to_string(index=False))


def print_summary_table(results: List[Dict], config: TTCConfig):
    """Print summary table of filtering results for all agents"""
    print('\n' + '='*100)
    print(f'Filtering Summary (Lon Threshold={config.LONGITUDINAL_TTC_THRESHOLD}s, Lat Threshold={config.LATERAL_TTC_THRESHOLD}s)')
    print('='*100)

    summary_data = []
    for r in results:
        lon_vals = r['lon_ttc'][np.isfinite(r['lon_ttc'])]
        lat_vals = r['lat_ttc'][np.isfinite(r['lat_ttc'])]

        summary_data.append({
            'Agent_ID': r['agent_id'],
            'Lon_TTC_min(s)': round(float(np.min(lon_vals)), 2) if len(lon_vals) > 0 else 'inf',
            'Lon_TTC_max(s)': round(float(np.max(lon_vals)), 2) if len(lon_vals) > 0 else 'inf',
            'Lat_TTC_min(s)': round(float(np.min(lat_vals)), 2) if len(lat_vals) > 0 else 'inf',
            'Lat_TTC_max(s)': round(float(np.max(lat_vals)), 2) if len(lat_vals) > 0 else 'inf',
            'Lon_Risk': 'Yes' if r['lon_risk'] else 'No',
            'Lat_Risk': 'Yes' if r['lat_risk'] else 'No',
            'Keep': 'Yes' if r['keep'] else 'No'
        })

    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))

    kept = sum(1 for r in results if r['keep'])
    print(f'\nKeep Rate: {kept}/{len(results)} = {kept/len(results)*100:.1f}%')


def print_full_report(result: Dict, config: TTCConfig):
    """Print full report"""
    print('\n' + '='*100)
    print(f'Scenario: {result["file_name"]}')
    print(f'Frames: {result["num_frames"]}')
    print('='*100)

    # Ego data
    print_ego_table(result['ego_data'])

    # Kept agents detailed data
    kept_agents = [r for r in result['agent_results'] if r['keep']]
    for agent_result in kept_agents:
        print_agent_ttc_table(agent_result)

    # Summary table
    print_summary_table(result['agent_results'], config)


# ============================================================
# TTC筛选报告生成函数
# ============================================================

def generate_ttc_report(
    successful_scenarios: List[Dict],
    failed_scenarios: List[Dict],
    total_pairs_saved: int,
    total_kept: int,
    total_original: int,
    output_dir: str,
    config: TTCConfig
):
    """
    生成TTC筛选处理报告

    Args:
        successful_scenarios: 成功处理的场景列表
        failed_scenarios: 失败的场景列表
        total_pairs_saved: 保存的danger_pair数量
        total_kept: 筛选出的危险车辆总数
        total_original: 原始车辆总数
        output_dir: 输出目录
        config: TTC配置
    """
    from datetime import datetime
    import os

    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    report = f"""# TTC危险场景筛选报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 筛选配置

| 参数 | 值 | 说明 |
|------|---|------|
| 纵向TTC阈值 | {config.LONGITUDINAL_TTC_THRESHOLD}秒 | 超过此值认为无纵向碰撞风险 |
| 横向TTC阈值 | {config.LATERAL_TTC_THRESHOLD}秒 | 超过此值认为无横向碰撞风险 |
| 考虑车辆尺寸 | {config.CONSIDER_VEHICLE_SIZE} | 是否使用边界框计算 |
| 尺寸缓冲区 | {config.VEHICLE_SIZE_BUFFER}米 | 安全边距 |

---

## 2. 处理统计

### 基本统计
| 指标 | 结果 |
|------|------|
| 总处理场景 | {len(successful_scenarios) + len(failed_scenarios)}个 |
| 成功处理 | **{len(successful_scenarios)}个** |
| 处理失败 | {len(failed_scenarios)}个 |
| 原始车辆总数 | {total_original}辆 |
| 危险车辆(kept) | **{total_kept}辆 ({total_kept/total_original*100:.1f}%)** |
| 保存的danger_pair | **{total_pairs_saved}个** |

### 筛选比例
- 输入场景: {len(successful_scenarios) + len(failed_scenarios)}个
- 输出danger_pair: {total_pairs_saved}个
- 有效筛选率: {total_pairs_saved/(len(successful_scenarios) + len(failed_scenarios))*100:.1f}%

---

## 3. 成功处理的场景详情

### 按选择原因统计
"""

    # 按选择原因统计
    selection_reasons = {}
    for s in successful_scenarios:
        reason = s.get('selection_reason', 'unknown')
        if reason not in selection_reasons:
            selection_reasons[reason] = 0
        selection_reasons[reason] += 1

    report += "\n| 选择原因 | 场景数 | 占比 |\n|---------|-------|------|\n"
    for reason, count in selection_reasons.items():
        pct = count / len(successful_scenarios) * 100 if successful_scenarios else 0
        report += f"| {reason} | {count} | {pct:.1f}% |\n"

    report += f"""
### 危险车辆数量分布
"""

    # 按kept_agents数量统计
    kept_distribution = {}
    for s in successful_scenarios:
        kept = s.get('kept_agents', 0)
        if kept not in kept_distribution:
            kept_distribution[kept] = 0
        kept_distribution[kept] += 1

    report += "\n| 危险车辆数 | 场景数 | 占比 |\n|-----------|-------|------|\n"
    for kept in sorted(kept_distribution.keys()):
        count = kept_distribution[kept]
        pct = count / len(successful_scenarios) * 100 if successful_scenarios else 0
        report += f"| {kept}辆 | {count} | {pct:.1f}% |\n"

    # 示例场景列表（前10个）
    report += f"""
### 示例danger_pair场景（前10个）

| 场景文件 | Key Agent ID | 选择原因 | 危险车辆数 |
|---------|-------------|---------|-----------|
"""
    for i, s in enumerate(successful_scenarios[:10]):
        report += f"| {s['file'][:30]}... | {s['key_agent_id']} | {s['selection_reason']} | {s['kept_agents']} |\n"

    report += f"""
---

## 4. 失败场景

"""
    if failed_scenarios:
        report += "\n| 场景文件 | 错误信息 |\n|---------|---------|\n"
        for f in failed_scenarios[:10]:
            report += f"| {f['file'][:30]}... | {f['error'][:50]} |\n"
        if len(failed_scenarios) > 10:
            report += f"\n共 {len(failed_scenarios)} 个失败场景，仅显示前10个\n"
    else:
        report += "\n**无失败场景**\n"

    report += f"""
---

## 5. 输出文件

- 输出目录: `{output_dir}`
- danger_pair文件数量: {total_pairs_saved}个
- 文件格式: `danger_pair_features_*.pkl`

### danger_pair数据格式

每个pkl文件包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| ego_history | ({TimeWindowConfig.NUM_PAST_POSES}, 5) | {TimeWindowConfig.PAST_TIME_HORIZON}秒历史轨迹 |
| key_agent_history | ({TimeWindowConfig.NUM_PAST_POSES}, 5) | 关键agent历史轨迹 |
| ego_future_gt | ({TimeWindowConfig.NUM_FUTURE_POSES}, 5) | {TimeWindowConfig.FUTURE_TIME_HORIZON}秒未来轨迹GT |
| key_agent_future | ({TimeWindowConfig.NUM_FUTURE_POSES}, 3)或None | agent未来轨迹 |
| key_agent_lon_ttc | array | 纵向TTC序列 |
| key_agent_lat_ttc | array | 横向TTC序列 |
| boundary_agents | list | 其他危险agent列表 |
| time_window_config | dict | 时间窗口配置信息 |

---

## 6. 后续使用

筛选后的danger_pair数据可直接用于：

1. **Diffusion模型训练**
   - `train_diffusion_generator.py` 训练危险场景生成器

2. **场景生成**
   - `generate_danger_scenes.py` 生成更危险的轨迹

---

**报告生成完毕**
"""

    # 保存报告
    report_file = output_dir / f"ttc_filter_report_{timestamp}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"TTC筛选报告已生成")
    print(f"{'='*60}")
    print(f"  📄 报告文件: {report_file}")
    print(f"{'='*60}")

    return report_file


# ============================================================
# 主函数
# ============================================================

def get_available_scenario_paths(data_dirs: List[str]) -> List[str]:
    """聚合多个目录下的 *.pkl，返回按文件名(token)排序的完整路径列表。

    排序保证多进程分片（offset/shard 切片）确定且不重不漏：每个进程独立读到同一份
    有序列表，再各取互不相交的切片。供 speed(=medium+high) / intersection(=2 类) 跨目录抽取。
    """
    paths: List[str] = []
    for d in data_dirs:
        paths.extend(str(p) for p in Path(d).glob("*.pkl"))
    paths.sort(key=lambda p: Path(p).name)
    return paths


def _parse_args():
    """CLI 参数：全部默认 None/0 → 不传参时完全保持原 ScenarioConfig 行为（可回档）。"""
    import argparse
    parser = argparse.ArgumentParser(description="TTC danger pair 抽取（支持多目录输入 + 分片并行）")
    parser.add_argument('--input-dir', nargs='+', default=None,
                        help='一个或多个 processed_<type> 目录；聚合 glob *.pkl。不传则用 config.PROCESSED_DATA_DIR')
    parser.add_argument('--output-dir', default=None,
                        help='danger pair pkl 输出目录（覆盖 SELECTED_PAIR_DIR）')
    parser.add_argument('--offset', type=int, default=0,
                        help='[分片] 排序后场景列表切片起点')
    parser.add_argument('--shard', type=int, default=None,
                        help='[分片] 切片大小；设了即开分片模式（多进程各取 [offset:offset+shard]）')
    parser.add_argument('--max-files', type=int, default=None,
                        help='最大处理文件数（None=全部；分片后再截断）')
    return parser.parse_args()


def main():
    """Main function"""
    import sys
    from io import StringIO
    from datetime import datetime

    args = _parse_args()

    # Create config
    config = TTCConfig()
    scenario_config = ScenarioConfig()

    # CLI 覆盖（默认 None → 保持原行为）
    if args.output_dir:
        scenario_config.SELECTED_PAIR_DIR = args.output_dir
    if args.max_files is not None:
        scenario_config.MAX_FILES = args.max_files

    # Create output directory
    output_dir = Path(scenario_config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if scenario_config.SINGLE_FILE:
        output_file = output_dir / f"ttc_filter_{scenario_config.SINGLE_FILE.replace('.pkl', '')}_{timestamp}.txt"
    else:
        # 分片时把 split 名 + offset 写进文件名，避免多进程/多 split 同秒 timestamp 撞名互相覆盖
        _split_tag = f"_{Path(scenario_config.SELECTED_PAIR_DIR).name}" if args.output_dir else ""
        _shard_suffix = f"_off{args.offset}" if args.shard is not None else ""
        output_file = output_dir / f"ttc_filter_batch_{timestamp}{_split_tag}{_shard_suffix}.txt"

    # Capture output
    output_buffer = StringIO()
    original_stdout = sys.stdout

    def tee_output(text):
        """Output to both screen and file"""
        print(text, end='')
        output_buffer.write(text)

    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
        def flush(self):
            for f in self.files:
                f.flush()

    # Redirect output
    sys.stdout = TeeOutput(original_stdout, output_buffer)

    try:
        print('='*100)
        print('TTC Filter Tool')
        print('='*100)
        print(f'\nConfiguration:')
        print(f'  Longitudinal TTC Threshold: {config.LONGITUDINAL_TTC_THRESHOLD}s')
        print(f'  Lateral TTC Threshold: {config.LATERAL_TTC_THRESHOLD}s')
        print(f'  Data Directory: {scenario_config.DATA_DIR}')
        print(f'  Output File: {output_file}')

        # Process scenario
        if scenario_config.SINGLE_FILE:
            # Process single scenario
            file_path = Path(scenario_config.DATA_DIR) / scenario_config.SINGLE_FILE
            if not file_path.exists():
                print(f'\n[Error] File not found: {file_path}')
                # List available scenarios
                scenarios = get_available_scenarios(scenario_config.DATA_DIR)
                print(f'\nAvailable scenario files ({len(scenarios)} total):')
                for i, s in enumerate(scenarios[:10]):
                    print(f'  {i+1}. {s}')
                if len(scenarios) > 10:
                    print(f'  ... and {len(scenarios) - 10} more')
                return

            print(f'\nProcessing scenario: {scenario_config.SINGLE_FILE}')
            result = process_scenario(str(file_path), config)
            print_full_report(result, config)

            # Select key agent pair for study
            print('\n' + '='*100)
            print('KEY AGENT PAIR SELECTION')
            print('='*100)
            print('Selecting the most suitable agent for dangerous trajectory generation study...')

            pair_config = PairSelectionConfig()
            selection_result = select_key_agent_pair(result, config, pair_config)
            print_pair_selection_result(selection_result)

            # Save selected pair data for diffusion training
            if selection_result['selected']:
                print('\n' + '='*100)
                print('SAVING SELECTED PAIR DATA')
                print('='*100)

                # Load original data to get ground truth
                original_df = load_feature_file(file_path)

                # 从文件名提取scenario token
                # 文件名格式: features_{token}.pkl
                scenario_token = scenario_config.SINGLE_FILE.replace('features_', '').replace('.pkl', '')

                # 准备agent位置和历史轨迹用于匹配
                agent_positions = {}
                agent_histories = {}
                for agent_result in result['agent_results']:
                    if agent_result['keep']:
                        agent_id = agent_result['agent_id']
                        agent_data = agent_result['agent_data']
                        # 使用最后一帧的位置
                        last_x = agent_data[-1, 0]
                        last_y = agent_data[-1, 1]
                        agent_positions[agent_id] = (last_x, last_y)
                        # 保存历史轨迹用于轨迹匹配
                        agent_histories[agent_id] = agent_data

                print(f"Extracting agent futures for scenario: {scenario_token}")
                print(f"  Found {len(agent_positions)} agents to match")

                # 从NuPlan提取agent未来轨迹
                agent_futures = extract_agent_futures_from_nuplan(
                    scenario_token,
                    agent_positions=agent_positions,
                    agent_histories=agent_histories
                )
                print(f"  Successfully extracted futures for {len(agent_futures)} agents")

                saved_file = save_selected_pair_data(
                    scenario_result=result,
                    selection_result=selection_result,
                    original_df=original_df,
                    output_dir=scenario_config.SELECTED_PAIR_DIR,
                    scenario_name=scenario_config.SINGLE_FILE,
                    agent_futures=agent_futures
                )

        else:
            # Batch processing
            # scenarios 统一为「完整路径字符串」列表（支持多目录输入）
            if args.input_dir:
                scenarios = get_available_scenario_paths(args.input_dir)
            else:
                scenarios = [str(Path(scenario_config.DATA_DIR) / n)
                             for n in get_available_scenarios(scenario_config.DATA_DIR)]
            # 分片：在排序后的完整列表上切片（确定性，不重不漏）
            if args.shard is not None:
                scenarios = scenarios[args.offset: args.offset + args.shard]
            if scenario_config.MAX_FILES:
                scenarios = scenarios[:scenario_config.MAX_FILES]

            print(f'\nBatch processing {len(scenarios)} scenarios'
                  + (f'（分片 offset={args.offset} shard={args.shard}）' if args.shard is not None else ''))

            total_original = 0
            total_kept = 0
            total_pairs_saved = 0
            successful_scenarios = []
            failed_scenarios = []

            # 创建输出目录
            pair_output_dir = Path(scenario_config.SELECTED_PAIR_DIR)
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            for i, scenario_file in enumerate(scenarios):
                # scenario_file 现在是完整路径（多目录输入）；fname 仅用于命名/日志
                file_path = Path(scenario_file)
                fname = file_path.name
                try:
                    result = process_scenario(str(file_path), config)
                    total_original += result['num_original']
                    total_kept += result['num_kept']

                    # 选择关键agent pair
                    pair_config = PairSelectionConfig()
                    selection_result = select_key_agent_pair(result, config, pair_config)

                    if selection_result['selected']:
                        # 保存danger_pair数据（不从NuPlan提取，使用None）
                        original_df = load_feature_file(file_path)
                        scenario_name = fname.replace('.pkl', '')

                        # GT未来危险度过滤：跳过ego在GT未来已规避危险的场景
                        if pair_config.MIN_GT_FUTURE_DANGER > 0:
                            gt_future_danger = _compute_gt_future_danger(
                                scenario_result=result,
                                original_df=original_df,
                                selected_agent_data=selection_result['agent_data'],
                            )
                            if gt_future_danger < pair_config.MIN_GT_FUTURE_DANGER:
                                print(f'[{i+1}/{len(scenarios)}] {fname}: '
                                      f'[Skip] gt_future_danger={gt_future_danger:.2f} < {pair_config.MIN_GT_FUTURE_DANGER}')
                                continue

                        saved_file = save_selected_pair_data(
                            scenario_result=result,
                            selection_result=selection_result,
                            original_df=original_df,
                            output_dir=scenario_config.SELECTED_PAIR_DIR,
                            scenario_name=scenario_name,
                            agent_futures=None  # 不从NuPlan提取
                        )

                        if saved_file:
                            total_pairs_saved += 1
                            successful_scenarios.append({
                                'file': fname,
                                'kept_agents': result['num_kept'],
                                'key_agent_id': selection_result['agent_id'],
                                'selection_reason': selection_result['selection_reason']
                            })

                    print(f'[{i+1}/{len(scenarios)}] {fname}: '
                          f'{result["num_kept"]}/{result["num_original"]} agents, '
                          f'pair_saved={selection_result["selected"]}')

                except Exception as e:
                    failed_scenarios.append({'file': fname, 'error': str(e)})
                    print(f'[{i+1}/{len(scenarios)}] {fname}: Error - {e}')

            print(f'\n{"="*60}')
            print(f'Batch Processing Summary')
            print(f'{"="*60}')
            print(f'  Total scenarios processed: {len(scenarios)}')
            print(f'  Successful: {len(successful_scenarios)}')
            print(f'  Failed: {len(failed_scenarios)}')
            print(f'  Total agents: {total_original}')
            print(f'  Risky agents (kept): {total_kept} ({total_kept/total_original*100:.1f}%)')
            print(f'  Danger pairs saved: {total_pairs_saved}')
            print("="*60)

        print(f'\nResults saved to: {output_file}')

    finally:
        # Restore stdout
        sys.stdout = original_stdout

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_buffer.getvalue())

        print(f'\nResults saved to: {output_file}')

        # Generate processing report if batch processing
        # 分片模式跳过：各分片只有局部统计，且会覆盖同一 report 文件；聚合统计另行汇总
        if not scenario_config.SINGLE_FILE and args.shard is None:
            generate_ttc_report(
                successful_scenarios,
                failed_scenarios,
                total_pairs_saved,
                total_kept,
                total_original,
                scenario_config.SELECTED_PAIR_DIR,
                config
            )


if __name__ == "__main__":
    main()
