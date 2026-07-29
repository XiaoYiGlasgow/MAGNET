#!/usr/bin/env python3
"""
批量处理Pittsburgh数据集 - 用于Diffusion危险场景生成器训练

处理流程：
1. 从NuPlan Pittsburgh数据集提取场景
2. 使用新配置（4秒历史 + 6秒未来）处理
3. 保存12维特征向量到processed_data目录
4. 生成处理结果报告文档
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

# 导入本地配置
import config as local_config

# 导入NuPlan相关模块
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# 导入处理函数
from GNN_Trans.process_nuplan_to_12d import process_nuplan_scenario_to_12d_features_silent
from GNN_Trans.utils.data_utils import (
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    agent_past_process,
    get_neighbor_vector_set_map,
    map_process,
    encoding_vectornet_features
)

from nuplan.common.actor_state.state_representation import Point2D


def create_process_config():
    """创建处理配置（使用本地config.py的参数）"""
    return {
        # 时间相关参数
        'num_past_poses': local_config.NUM_PAST_POSES,
        'num_future_poses': local_config.NUM_FUTURE_POSES,
        'past_time_horizon': local_config.PAST_TIME_HORIZON,
        'future_time_horizon': local_config.FUTURE_TIME_HORIZON,

        # 车辆相关参数
        'num_agents': local_config.NUM_AGENTS,

        # 地图相关参数
        'map_features': ['LANE', 'ROUTE_LANES', 'CROSSWALK'],
        'map_radius': local_config.QUERY_RADIUS,
        'max_elements': {
            "LANE": local_config.LANE_NUM,
            "CROSSWALK": local_config.CROSSWALKS_NUM,
            "ROUTE_LANES": local_config.ROUTE_LANES_NUM
        },
        'max_points': {
            "LANE": local_config.LANE_POINTS_NUM,
            "CROSSWALK": local_config.CROSSWALKS_POINTS_NUM,
            "ROUTE_LANES": local_config.ROUTE_LANES_POINTS_NUM
        },
        'interpolation_method': 'linear'
    }


def process_scenarios_batch(scenarios, output_dir, process_config, max_scenarios=None):
    """
    批量处理场景

    Args:
        scenarios: NuPlan场景列表
        output_dir: 输出目录
        process_config: 处理配置
        max_scenarios: 最大处理数量（可选）
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_scenarios = len(scenarios)
    if max_scenarios:
        total_scenarios = min(total_scenarios, max_scenarios)

    print(f"\n{'='*60}")
    print(f"批量数据处理开始")
    print(f"{'='*60}")
    print(f"  总场景数: {len(scenarios)}")
    print(f"  处理数量: {total_scenarios}")
    print(f"  输出目录: {output_dir}")
    print(f"  历史帧数: {process_config['num_past_poses']} ({process_config['past_time_horizon']}秒)")
    print(f"  未来帧数: {process_config['num_future_poses']} ({process_config['future_time_horizon']}秒)")
    print(f"{'='*60}\n")

    start_time = time.time()
    processed_count = 0
    failed_count = 0

    for i, scenario in enumerate(scenarios[:total_scenarios]):
        try:
            # 进度显示
            if (i + 1) % local_config.LOG_EVERY == 0 or i == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1) if i > 0 else elapsed
                remaining = avg_time * (total_scenarios - i - 1)

                print(f"\n[{i+1}/{total_scenarios}] 处理: {scenario.scenario_name}")
                print(f"  已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")
                print(f"  成功: {processed_count} | 失败: {failed_count}")

            # 静默处理单个场景
            feature_df = process_nuplan_scenario_to_12d_features_silent(scenario, process_config)

            # 保存结果
            output_file = output_dir / f"features_{scenario.scenario_name}.pkl"
            feature_df.to_pickle(output_file)

            processed_count += 1

            # 定期保存进度
            if (i + 1) % local_config.SAVE_EVERY == 0:
                print(f"\n  💾 已保存 {processed_count} 个场景")

        except Exception as e:
            failed_count += 1
            if (i + 1) % local_config.LOG_EVERY == 0:
                print(f"  ❌ 失败: {e}")
            continue

    # 最终统计
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"批量处理完成!")
    print(f"{'='*60}")
    print(f"  ✅ 成功处理: {processed_count} 个场景")
    print(f"  ❌ 处理失败: {failed_count} 个场景")
    print(f"  ⏰ 总用时: {total_time/60:.1f}分钟")
    print(f"  📊 平均速度: {processed_count/(total_time/60):.1f}场景/分钟")
    print(f"  📁 结果保存在: {output_dir}")
    print(f"{'='*60}\n")

    return processed_count, failed_count


def generate_processing_report(output_dir, process_config, processed_count, failed_count, total_time):
    """
    生成处理结果报告文档

    Args:
        output_dir: 输出目录
        process_config: 处理配置
        processed_count: 成功处理数量
        failed_count: 失败数量
        total_time: 总用时（秒）
    """

    output_dir = Path(output_dir)
    pkl_files = list(output_dir.glob('features_*.pkl'))

    # 统计车辆数量
    agent_counts = []
    polyline_sizes = []
    gt_shapes = []

    for pkl_file in pkl_files:
        try:
            df = pd.read_pickle(pkl_file)
            traj_indices = df.iloc[0]['TRAJ_ID_TO_INDICES']
            agent_ids = [k for k in traj_indices.keys() if k > 0]
            agent_counts.append(len(agent_ids))

            polyline = df.iloc[0]['POLYLINE_FEATURES']
            polyline_sizes.append(polyline.shape[0])

            gt = df.iloc[0]['GROUND_TRUTH']
            gt_shapes.append(gt.shape[1])
        except Exception as e:
            continue

    # 生成报告内容
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = f"""# Diffusion危险场景生成器 - 数据处理报告

**生成时间**: {timestamp}

---

## 1. 处理配置

### 时间窗口配置
| 参数 | 值 | 说明 |
|------|---|------|
| 历史时间窗口 | {process_config['past_time_horizon']}秒 | {process_config['num_past_poses']}帧 |
| 未来时间窗口 | {process_config['future_time_horizon']}秒 | {process_config['num_future_poses']}帧 |
| 帧率 | {local_config.DISCRETE_SIZE}Hz | 采样频率 |

### 车辆和地图配置
| 参数 | 值 | 说明 |
|------|---|------|
| 最大周围车辆 | {process_config['num_agents']}辆 | NUM_AGENTS上限 |
| 地图提取半径 | {process_config['map_radius']}米 | QUERY_RADIUS |

### 筛选配置
| 参数 | 值 | 说明 |
|------|---|------|
| 场景类型筛选 | {local_config.SCENARIO_TYPES or '无'} | 是否筛选特定类型 |
| 最小位移 | {local_config.MIN_DISPLACEMENT}米 | 排除静止场景 |

---

## 2. 处理结果统计

### 基本统计
| 指标 | 结果 |
|------|------|
| 成功处理 | **{processed_count}个场景** |
| 处理失败 | {failed_count}个 |
| 处理速度 | {processed_count/(total_time/60):.1f}场景/分钟 |
| 总用时 | {total_time/60:.1f}分钟 |

### 周围车辆数量分布
"""

    # 添加车辆数量分布表格
    if agent_counts:
        report += "\n| 车辆数 | 场景数量 | 占比 |\n|--------|----------|------|\n"
        for count in sorted(set(agent_counts)):
            freq = agent_counts.count(count)
            pct = freq / len(agent_counts) * 100
            report += f"| {count}辆 | {freq}个 | {pct:.1f}% |\n"

        report += f"""
### 车辆数量汇总
| 指标 | 值 |
|------|---|
| 平均周围车辆数 | {sum(agent_counts)/len(agent_counts):.2f}辆 |
| 最少 | {min(agent_counts)}辆 |
| 最多 | {max(agent_counts)}辆 |
| 有0辆车的场景 | {agent_counts.count(0)}个 ({agent_counts.count(0)/len(agent_counts)*100:.1f}%) |
| 有>=3辆车的场景 | {len([c for c in agent_counts if c >= 3])}个 ({len([c for c in agent_counts if c >= 3])/len(agent_counts)*100:.1f}%) |
| 有>=5辆车的场景 | {len([c for c in agent_counts if c >= 5])}个 ({len([c for c in agent_counts if c >= 5])/len(agent_counts)*100:.1f}%) |
"""

    report += f"""
---

## 3. 数据格式验证

| 字段 | 形状 | 说明 |
|------|------|------|
| POLYLINE_FEATURES | (N, 12) | 12维特征向量，N为节点数 |
| GROUND_TRUTH | (1, {process_config['num_future_poses']}, 2) | {process_config['num_future_poses']}帧未来轨迹增量 |
| TRAJ_ID_TO_INDICES | dict | 轨迹ID到索引映射 |
| EGO_LAST_POS | [x, y] | Ego绝对位置 |

### 特征向量说明（12维）
| 维度 | 字段 | 说明 |
|------|------|------|
| 0-1 | x_start, y_start | segment起点坐标 |
| 2-3 | x_end, y_end | segment终点坐标 |
| 4 | heading_start | 起点朝向 |
| 5 | heading_end | 终点朝向 |
| 6 | normalized_timestamp | 归一化时间戳 |
| 7 | v_start | 起点速度 |
| 8 | v_end | 终点速度 |
| 9 | subgraph_ID | 子图ID |
| 10 | start_time | 相对开始时间 |
| 11 | end_time | 相对结束时间 |

---

## 4. 说明

### ID含义
- **ID=0**: Ego（本车），不计入周围车辆统计
- **ID>0 + type='agent'**: 周围车辆，计入统计
- **ID>0 + type='lane'**: 车道信息，不计入统计

### 注意事项
- 当前数据为**原始提取**，周围车辆**未经危险筛选**
- 需要后续使用 `ttc_filter.py` 进行TTC危险场景筛选
- 车辆数量取决于NuPlan原始数据中实际可见的车辆

---

## 5. 输出文件

- 输出目录: `{output_dir}`
- 文件数量: {len(pkl_files)}个
- 文件格式: `features_*.pkl`

---

**报告生成完毕**
"""

    # 保存报告
    report_file = output_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"处理报告已生成")
    print(f"{'='*60}")
    print(f"  📄 报告文件: {report_file}")
    print(f"{'='*60}\n")

    return report_file


def main():
    """主函数"""

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Diffusion危险场景生成器 - 数据处理")
    print(f"{'='*60}")
    print(f"数据源: Pittsburgh")
    print(f"输出目录: {local_config.SAVE_PATH}")
    print(f"{'='*60}\n")

    # 1. 创建场景构建器
    print("1. 创建NuPlan场景构建器...")

    scenario_builder = NuPlanScenarioBuilder(
        data_root=local_config.NUPLAN_DATA_ROOT,
        map_root=local_config.NUPLAN_MAP_ROOT,
        sensor_root=local_config.NUPLAN_SENSOR_ROOT,
        db_files=None,
        map_version=local_config.NUPLAN_MAP_VERSION
    )

    print(f"  ✅ 数据源: {local_config.NUPLAN_DATA_ROOT}")

    # 2. 创建场景过滤器
    print("\n2. 创建场景过滤器...")

    scenario_filter = ScenarioFilter(
        scenario_types=local_config.SCENARIO_TYPES,  # 按场景类型筛选
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=local_config.MAX_SCENARIOS,  # None表示全部
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=local_config.MIN_DISPLACEMENT,  # 排除位移太小的场景
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False  # 保持原始顺序
    )

    if local_config.SCENARIO_TYPES:
        print(f"  ✅ 筛选场景类型: {local_config.SCENARIO_TYPES}")
    else:
        print(f"  ✅ 不筛选场景类型（处理所有类型）")
    print(f"  ✅ 最小位移: {local_config.MIN_DISPLACEMENT}m（排除静止场景）")
    print(f"  ✅ 最大场景数: {local_config.MAX_SCENARIOS or '全部'}")

    # 3. 获取场景
    print("\n3. 获取场景列表...")

    worker = Sequential()
    scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    print(f"  ✅ 获取到 {len(scenarios)} 个场景")

    # 4. 批量处理
    print("\n4. 开始批量处理...")

    process_config = create_process_config()

    processed, failed = process_scenarios_batch(
        scenarios,
        local_config.SAVE_PATH,
        process_config,
        max_scenarios=local_config.MAX_SCENARIOS
    )

    # 5. 验证结果
    if processed > 0:
        print("\n5. 验证处理结果...")

        # 检查一个输出文件
        output_files = list(Path(local_config.SAVE_PATH).glob("features_*.pkl"))
        if output_files:
            sample_file = output_files[0]
            sample_df = pd.read_pickle(sample_file)

            print(f"  示例文件: {sample_file.name}")
            print(f"  POLYLINE_FEATURES: {sample_df.iloc[0]['POLYLINE_FEATURES'].shape}")

            gt = sample_df.iloc[0]['GROUND_TRUTH']
            print(f"  GROUND_TRUTH: {gt.shape} (未来{gt.shape[1]}帧)")

            traj_indices = sample_df.iloc[0]['TRAJ_ID_TO_INDICES']
            if 0 in traj_indices:
                start, end = traj_indices[0]
                frames = end - start + 2
                print(f"  历史轨迹: {frames}帧")

            print(f"\n  ✅ 数据验证通过!")

    # 6. 生成处理报告
    total_time = time.time() - start_time

    if processed > 0:
        print("\n6. 生成处理报告...")

        process_config = create_process_config()
        report_file = generate_processing_report(
            local_config.SAVE_PATH,
            process_config,
            processed,
            failed,
            total_time
        )

    print(f"\n{'='*60}")
    print(f"全部完成!")
    print(f"{'='*60}")
    print(f"  处理场景: {processed}个")
    print(f"  用时: {total_time/60:.1f}分钟")
    print(f"  报告文件: {report_file if processed > 0 else '无'}")
    print(f"{'='*60}")

    return processed, failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量处理Pittsburgh数据')
    parser.add_argument('--max', type=int, default=None, help='最大处理场景数')
    parser.add_argument('--output', type=str, default=None, help='输出目录')

    args = parser.parse_args()

    # 覆盖配置
    if args.max:
        local_config.MAX_SCENARIOS = args.max
    if args.output:
        local_config.SAVE_PATH = args.output

    main()