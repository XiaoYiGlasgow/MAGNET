"""
使用筛选出的危险场景数据计算危险评分

演示如何将 danger_pair_*.pkl 文件与 danger_score.py 结合使用
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from danger_score import compute_danger_score, DangerScoreConfig


def load_danger_pair_data(pkl_path: str) -> dict:
    """
    加载筛选出的危险场景数据

    Args:
        pkl_path: danger_pair_*.pkl 文件路径

    Returns:
        处理后的数据字典，可直接用于 danger_score 计算
    """
    data = pd.read_pickle(pkl_path)
    return data


def prepare_trajectories_for_danger_score(data: dict) -> dict:
    """
    将 pkl 数据转换为 danger_score 所需的格式

    danger_score 期望:
    - ego_history: (T_hist, 4或5) [x, y, heading, vx, vy]
    - ego_future: (T_fut, 4或5) [x, y, heading, vx, vy]
    - agent_history: (T_hist, 4或5)
    - agent_future: (T_fut, 4或5)

    pkl 数据:
    - ego_history: (41, 5) [x, y, heading, vx, vy] ✓
    - ego_future_gt: (1, 30, 2) [x, y] ✗ 需要处理
    - key_agent_history: (41, 5) [x, y, heading, vx, vy] ✓
    - key_agent_future: (31, 3) [x, y, heading] ✗ 需要处理
    """
    # 处理 ego future
    ego_future_gt = data['ego_future_gt']
    if ego_future_gt.ndim == 3:
        ego_future_gt = ego_future_gt.squeeze(0)  # (1, 30, 2) -> (30, 2)

    # 从历史数据估计速度
    ego_history = data['ego_history']
    ego_vx_last = ego_history[-1, 3]
    ego_vy_last = ego_history[-1, 4]
    ego_heading_last = ego_history[-1, 2]

    # 构建 ego_future (30, 5)
    T_fut = len(ego_future_gt)
    ego_future = np.zeros((T_fut, 5))
    ego_future[:, :2] = ego_future_gt  # x, y
    # 估计 heading（从轨迹计算）
    if T_fut > 1:
        dx = np.diff(ego_future_gt[:, 0], prepend=ego_future_gt[0, 0])
        dy = np.diff(ego_future_gt[:, 1], prepend=ego_future_gt[0, 1])
        ego_future[:, 2] = np.arctan2(dy, dx)
    else:
        ego_future[:, 2] = ego_heading_last
    # 使用历史最后帧速度作为估计
    ego_future[:, 3] = ego_vx_last
    ego_future[:, 4] = ego_vy_last

    # 处理 key agent future
    key_agent_future = data['key_agent_future']
    if key_agent_future is not None:
        # key_agent_future: (31, 3) 包含当前位置 + 30帧未来
        # 取后30帧与 ego_future 对齐
        if len(key_agent_future) > T_fut:
            key_agent_future_aligned = key_agent_future[-T_fut:]
        else:
            key_agent_future_aligned = key_agent_future

        # 从历史估计速度
        agent_history = data['key_agent_history']
        agent_vx_last = agent_history[-1, 3]
        agent_vy_last = agent_history[-1, 4]

        # 构建 agent_future (30, 5)
        agent_future = np.zeros((T_fut, 5))
        agent_future[:, :3] = key_agent_future_aligned  # x, y, heading
        agent_future[:, 3] = agent_vx_last
        agent_future[:, 4] = agent_vy_last
    else:
        agent_future = None
        agent_history = None

    return {
        'ego_history': ego_history,
        'ego_future': ego_future,
        'agent_history': data['key_agent_history'],
        'agent_future': agent_future,
        'key_agent_id': data['key_agent_id'],
        'selection_reason': data['selection_reason'],
        'num_boundary_agents': data['num_boundary_agents']
    }


def compute_ground_truth_danger_score(data: dict, config: DangerScoreConfig = None, verbose: bool = True):
    """
    计算 ground truth 轨迹的危险评分

    评估原始 NuPlan 数据中这对轨迹的危险程度
    """
    if config is None:
        config = DangerScoreConfig()

    # 准备轨迹数据
    trajectories = prepare_trajectories_for_danger_score(data)

    if trajectories['agent_future'] is None:
        print("Warning: Key agent future is None, cannot compute danger score")
        return None

    # 计算危险评分
    result = compute_danger_score(
        ego_history=trajectories['ego_history'],
        ego_future=trajectories['ego_future'],
        agent_history=trajectories['agent_history'],
        agent_future=trajectories['agent_future'],
        config=config,
        dt=0.1,  # 10Hz
        verbose=verbose
    )

    return result, trajectories


def compute_boundary_agent_danger_scores(data: dict, config: DangerScoreConfig = None):
    """
    计算所有 boundary agents 的危险评分

    用于 IDM 模拟时确定边界车辆的影响
    """
    if config is None:
        config = DangerScoreConfig()

    ego_history = data['ego_history']

    # 处理 ego future
    ego_future_gt = data['ego_future_gt']
    if ego_future_gt.ndim == 3:
        ego_future_gt = ego_future_gt.squeeze(0)

    T_fut = len(ego_future_gt)
    ego_vx_last = ego_history[-1, 3]
    ego_vy_last = ego_history[-1, 4]
    ego_heading_last = ego_history[-1, 2]

    ego_future = np.zeros((T_fut, 5))
    ego_future[:, :2] = ego_future_gt
    if T_fut > 1:
        dx = np.diff(ego_future_gt[:, 0], prepend=ego_future_gt[0, 0])
        dy = np.diff(ego_future_gt[:, 1], prepend=ego_future_gt[0, 1])
        ego_future[:, 2] = np.arctan2(dy, dx)
    else:
        ego_future[:, 2] = ego_heading_last
    ego_future[:, 3] = ego_vx_last
    ego_future[:, 4] = ego_vy_last

    results = []
    for ba in data['boundary_agents']:
        agent_id = ba['agent_id']
        agent_history = ba['agent_data']
        agent_future_raw = ba['agent_future']

        if agent_future_raw is None:
            # 没有未来轨迹，使用历史估计
            agent_vx_last = agent_history[-1, 3]
            agent_vy_last = agent_history[-1, 4]

            agent_future = np.zeros((T_fut, 5))
            # 简单线性外推
            for t in range(T_fut):
                dt = (t + 1) * 0.1
                agent_future[t, 0] = agent_history[-1, 0] + agent_vx_last * dt
                agent_future[t, 1] = agent_history[-1, 1] + agent_vy_last * dt
            agent_future[:, 2] = agent_history[-1, 2]
            agent_future[:, 3] = agent_vx_last
            agent_future[:, 4] = agent_vy_last
        else:
            # 对齐到 ego future 的长度
            if len(agent_future_raw) > T_fut:
                agent_future_aligned = agent_future_raw[-T_fut:]
            else:
                agent_future_aligned = agent_future_raw

            agent_vx_last = agent_history[-1, 3]
            agent_vy_last = agent_history[-1, 4]

            # 使用实际长度
            actual_len = len(agent_future_aligned)
            agent_future = np.zeros((T_fut, 5))

            # 只复制实际存在的数据
            copy_cols = min(3, agent_future_aligned.shape[1])
            agent_future[:actual_len, :copy_cols] = agent_future_aligned[:, :copy_cols]
            agent_future[:, 3] = agent_vx_last
            agent_future[:, 4] = agent_vy_last

        # 计算危险评分
        result = compute_danger_score(
            ego_history=ego_history,
            ego_future=ego_future,
            agent_history=agent_history,
            agent_future=agent_future,
            config=config,
            dt=0.1,
            verbose=False
        )

        results.append({
            'agent_id': agent_id,
            'danger_score': result['total_score'],
            'danger_level': result['danger_level'],
            'min_distance': result['collision_info']['min_distance'],
            'min_lon_ttc': result['ttc_info']['min_lon_ttc'],
            'has_future_gt': agent_future_raw is not None
        })

    # 按危险评分排序
    results.sort(key=lambda x: x['danger_score'], reverse=True)

    return results


def demo_usage():
    """演示完整的使用流程"""
    print("=" * 80)
    print("危险场景数据使用示例")
    print("=" * 80)

    # 1. 加载数据
    pkl_path = str(Path(config.DANGER_PAIR_SCENARIOS_DIR) / "danger_pair_features_0a1e6c2a3a385290_agent1.pkl")
    print(f"\n1. 加载数据: {pkl_path}")
    data = load_danger_pair_data(pkl_path)

    print(f"   场景名称: {data['scenario_name']}")
    print(f"   Key Agent ID: {data['key_agent_id']}")
    print(f"   选择原因: {data['selection_reason']}")
    print(f"   边界车辆数: {data['num_boundary_agents']}")

    # 2. 计算 key agent 的危险评分
    print("\n2. 计算 Key Agent 的危险评分 (Ground Truth)")
    print("-" * 60)
    result, trajectories = compute_ground_truth_danger_score(data, verbose=True)

    # 3. 计算所有 boundary agents 的危险评分
    print("\n3. 计算所有 Boundary Agents 的危险评分")
    print("-" * 60)
    boundary_results = compute_boundary_agent_danger_scores(data)

    print(f"\nBoundary Agents 危险评分排名:")
    print(f"{'Agent ID':<10} {'危险评分':<12} {'危险等级':<10} {'最小距离':<12} {'最小TTC':<10} {'有GT':<6}")
    print("-" * 60)
    for r in boundary_results:
        print(f"{r['agent_id']:<10} {r['danger_score']:<12.2f} {r['danger_level']:<10} "
              f"{r['min_distance']:<12.2f} {r['min_lon_ttc']:<10.2f} {str(r['has_future_gt']):<6}")

    # 4. 数据用于 Diffusion 训练的示例
    print("\n4. Diffusion 训练数据格式")
    print("-" * 60)
    print(f"ego_history: {trajectories['ego_history'].shape}")
    print(f"ego_future (GT): {trajectories['ego_future'].shape}")
    print(f"agent_history: {trajectories['agent_history'].shape}")
    print(f"agent_future (GT): {trajectories['agent_future'].shape}")

    # 分离的危险指数
    print(f"\n--- 训练标签 (分离危险指数) ---")
    print(f"inherent_danger (场景固有风险): {result['inherent_danger']['total_score']:.2f}")
    print(f"  - 历史TTC: {result['inherent_danger']['ttc_info']['min_lon_ttc']:.2f}s")
    print(f"outcome_danger (结果风险): {result['outcome_danger']['total_score']:.2f}")
    print(f"  - 未来TTC: {result['outcome_danger']['ttc_info']['min_lon_ttc']:.2f}s")
    print(f"avoidance_quality (避让质量): {result['avoidance_quality']:.3f}")
    print(f"SVO角度: {result['svo_info']['ego_svo_angle_deg']:.1f}° (Ego), {result['svo_info']['agent_svo_angle_deg']:.1f}° (Agent)")

    return data, result, boundary_results


def save_text_report(
    data: dict,
    result: dict,
    boundary_results: list,
    output_dir: str = config.DANGER_PAIR_REPORTS_DIR
) -> str:
    """
    生成并保存文本报告

    Args:
        data: 原始 pkl 数据
        result: danger score 计算结果
        boundary_results: boundary agents 计算结果
        output_dir: 报告输出目录

    Returns:
        保存的报告文件路径
    """
    from pathlib import Path
    from datetime import datetime

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成报告文件名
    scenario_name = data['scenario_name'].replace('.pkl', '')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_path / f"report_{scenario_name}_{timestamp}.txt"

    # 生成报告内容
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("危险场景分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # 场景基本信息
    report_lines.append("-" * 80)
    report_lines.append("【场景基本信息】")
    report_lines.append("-" * 80)
    report_lines.append(f"场景名称: {data['scenario_name']}")
    report_lines.append(f"Key Agent ID: {data['key_agent_id']}")
    report_lines.append(f"选择原因: {data['selection_reason']}")
    report_lines.append(f"边界车辆数: {data['num_boundary_agents']}")
    report_lines.append("")

    # 危险指数分解
    report_lines.append("-" * 80)
    report_lines.append("【危险指数分解】")
    report_lines.append("-" * 80)

    inherent = result['inherent_danger']
    outcome = result['outcome_danger']

    report_lines.append("")
    report_lines.append("[历史轨迹危险指数 (inherent_danger)]")
    report_lines.append(f"  总分: {inherent['total_score']:.2f}")
    report_lines.append(f"  危险等级: {inherent['danger_level'].upper()}")
    report_lines.append(f"  最小距离: {inherent['collision_info']['min_distance']:.2f}m")
    report_lines.append(f"  最小纵向TTC: {inherent['ttc_info']['min_lon_ttc']:.2f}s")
    report_lines.append(f"  最小横向TTC: {inherent['ttc_info']['min_lat_ttc']:.2f}s")

    report_lines.append("")
    report_lines.append("[未来轨迹危险指数 (outcome_danger)]")
    report_lines.append(f"  总分: {outcome['total_score']:.2f}")
    report_lines.append(f"  危险等级: {outcome['danger_level'].upper()}")
    report_lines.append(f"  最小距离: {outcome['collision_info']['min_distance']:.2f}m")
    report_lines.append(f"  最小纵向TTC: {outcome['ttc_info']['min_lon_ttc']:.2f}s")
    report_lines.append(f"  最小横向TTC: {outcome['ttc_info']['min_lat_ttc']:.2f}s")
    if outcome['collision_info']['has_collision']:
        report_lines.append(f"  ⚠️ 碰撞时间步: {outcome['collision_info']['collision_timestep']}")

    report_lines.append("")
    report_lines.append("[避让质量]")
    report_lines.append(f"  avoidance_quality: {result['avoidance_quality']:.3f}")
    report_lines.append("  (1.0 = 完美避让, 0.0 = 未改善, <0 = 制造风险)")
    report_lines.append("")

    # SVO 分析
    report_lines.append("-" * 80)
    report_lines.append("【SVO 分析（单位圆累加模型）】")
    report_lines.append("-" * 80)

    svo = result['svo_info']

    report_lines.append("")
    report_lines.append("[Ego 车辆]")
    report_lines.append(f"  SVO 角度: {svo['ego_svo_angle_deg']:.1f}° ({svo['ego_svo_type']})")
    ego_svo = svo['ego_svo']
    report_lines.append(f"  单位圆权重: x={ego_svo['x_weight']:.3f}, y={ego_svo['y_weight']:.3f}")
    report_lines.append(f"  原始累加值: x_raw={ego_svo['x_raw']:.2f}, y_raw={ego_svo['y_raw']:.2f}")
    report_lines.append(f"  行为强度(norm): {ego_svo['norm']:.2f}")
    report_lines.append(f"  让行次数: {ego_svo['yield_count']}")
    report_lines.append(f"  抢行次数: {ego_svo['aggressive_count']}")
    report_lines.append(f"  避让次数: {ego_svo['avoidance_count']}")
    report_lines.append(f"  社会力: {ego_svo['social_force_magnitude']:.3f}")

    report_lines.append("")
    report_lines.append("[Agent 车辆]")
    report_lines.append(f"  SVO 角度: {svo['agent_svo_angle_deg']:.1f}° ({svo['agent_svo_type']})")
    agent_svo = svo['agent_svo']
    report_lines.append(f"  单位圆权重: x={agent_svo['x_weight']:.3f}, y={agent_svo['y_weight']:.3f}")
    report_lines.append(f"  原始累加值: x_raw={agent_svo['x_raw']:.2f}, y_raw={agent_svo['y_raw']:.2f}")
    report_lines.append(f"  行为强度(norm): {agent_svo['norm']:.2f}")
    report_lines.append(f"  让行次数: {agent_svo['yield_count']}")
    report_lines.append(f"  抢行次数: {agent_svo['aggressive_count']}")
    report_lines.append(f"  避让次数: {agent_svo['avoidance_count']}")
    report_lines.append(f"  社会力: {agent_svo['social_force_magnitude']:.3f}")

    report_lines.append("")
    report_lines.append("[交互分析]")
    report_lines.append(f"  交互类型: {svo['interaction_type']}")
    report_lines.append(f"  SVO 角度差: {svo['svo_angle_diff']:.1f}°")
    report_lines.append(f"  合作分数: {svo['cooperation_score']:.3f}")
    report_lines.append(f"  冲突倾向: {svo['conflict_tendency']:.3f}")
    report_lines.append(f"  冲突烈度: {svo['conflict_intensity']:.3f}")
    report_lines.append("")

    # Boundary Agents 分析
    report_lines.append("-" * 80)
    report_lines.append("【Boundary Agents 危险排名】")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append(f"{'Agent ID':<10} {'危险评分':<12} {'危险等级':<10} {'最小距离':<12} {'最小TTC':<10} {'有GT':<6}")
    report_lines.append("-" * 60)

    for r in boundary_results:
        ttc_str = f"{r['min_lon_ttc']:.2f}s" if r['min_lon_ttc'] != float('inf') else "inf"
        report_lines.append(
            f"{r['agent_id']:<10} {r['danger_score']:<12.2f} {r['danger_level']:<10} "
            f"{r['min_distance']:<12.2f} {ttc_str:<10} {str(r['has_future_gt']):<6}"
        )
    report_lines.append("")

    # Diffusion 训练标签总结
    report_lines.append("-" * 80)
    report_lines.append("【Diffusion 训练标签】")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("输入条件:")
    report_lines.append(f"  inherent_danger: {inherent['total_score']:.2f}")
    report_lines.append(f"  ego_svo_angle: {svo['ego_svo_angle_deg']:.1f}°")
    report_lines.append(f"  agent_svo_angle: {svo['agent_svo_angle_deg']:.1f}°")
    report_lines.append("")
    report_lines.append("训练目标:")
    report_lines.append(f"  outcome_danger: {outcome['total_score']:.2f}")
    report_lines.append(f"  avoidance_quality: {result['avoidance_quality']:.3f}")
    report_lines.append("")

    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n报告已保存到: {report_file}")

    return str(report_file)


if __name__ == "__main__":
    data, result, boundary_results = demo_usage()

    # 保存文本报告
    save_text_report(data, result, boundary_results)
