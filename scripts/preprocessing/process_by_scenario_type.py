#!/usr/bin/env python3

"""
按场景类型分类处理nuPlan数���
支持选择特定场景类型和数量限制
"""

import sys
import argparse
import time
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List, Optional
import config

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')
# 确保同目录(scripts/preprocessing)优先于顶层同名 shim，避免 process_nuplan_to_12d 被遮蔽
# （经顶层 shim 调用时本目录不在 path 上，sibling import 会错误命中顶层空壳 shim）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# 导入处理函数
try:
    from process_nuplan_to_12d import process_nuplan_scenario_to_12d_features_silent, create_default_config
except ImportError as e:
    print(f"错误：无法导入处理函数: {e}")
    print("请确保 process_nuplan_to_12d.py 文件存在且包含所需函数")
    sys.exit(1)


def get_default_paths() -> dict:
    """获取默认路径配置"""
    base_path = Path('/mnt/local/Xiao_Yi/nuplan-devkit')
    return {
        'data_root': str(base_path / 'nuplan/dataset/nuplan-v1.1/splits/train_pittsburgh'),
        'map_root': str(base_path / 'nuplan/dataset/maps'),
        'sensor_root': str(base_path / 'nuplan/dataset/nuplan-v1.1/sensor_blobs'),
        'map_version': 'nuplan-maps-v1.0'
    }


def print_progress_bar(current: int, total: int, prefix: str = 'Progress', suffix: str = 'Complete', length: int = 50) -> None:
    """打印进度条"""
    if total == 0:
        return
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)


def analyze_scenario_types(data_root: str) -> List[str]:
    """分析数据集中的场景类型分布"""
    print("分析场景类型分布...")
    
    paths = get_default_paths()
    
    try:
        scenario_builder = NuPlanScenarioBuilder(
            data_root=data_root,
            map_root=paths['map_root'],
            sensor_root=paths['sensor_root'],
            db_files=None,
            map_version=paths['map_version']
        )
    except Exception as e:
        print(f"错误：无法创建场景构建器: {e}")
        return []

    # 获取少量场景来分析类型
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=1000,  # 限制分析数量以提高速度
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=True
    )

    worker = Sequential()
    
    try:
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
    except Exception as e:
        print(f"错误：无法获取场景列表: {e}")
        return []

    # 统计场景类型
    type_counts = defaultdict(int)
    for scenario in scenarios:
        scenario_type = scenario.scenario_type
        type_counts[scenario_type] += 1

    total_scenarios = len(scenarios)
    print(f"\n发现 {len(type_counts)} 种场景类型，总共 {total_scenarios} 个场景:")

    for scenario_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_scenarios) * 100
        print(f"   {scenario_type}: {count} ({percentage:.1f}%)")

    return list(type_counts.keys())


def batch_process_with_progress(scenario_list: List, config: dict, output_dir: Path) -> tuple[int, int]:
    """带总进度条的批量处理"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_scenarios = len(scenario_list)
    success_count = 0
    failed_count = 0
    start_time = time.time()

    print(f"开始处理 {total_scenarios} 个场景")
    print(f"输出目录: {output_dir}")
    print()  # 空行，为进度条留出空间

    error_log = []

    for i, scenario in enumerate(scenario_list):
        try:
            print_progress_bar(i + 1, total_scenarios, f"处理中")

            # 处理场景数据
            result = process_nuplan_scenario_to_12d_features_silent(scenario, config)

            if result is not None:
                # 保存到文件
                output_file = output_dir / f"{scenario.token}.pkl"
                result.to_pickle(output_file)
                success_count += 1

        except Exception as e:
            failed_count += 1
            error_msg = f"场景 {scenario.scenario_name}: {str(e)}"
            error_log.append(error_msg)
            
            if failed_count <= 3:  # 只打印前3个错误的详细信息
                print(f"\n错误处理{error_msg}")
                if failed_count == 1:  # 只在第一个错误时打印详细堆栈
                    print(f"详细错误信息: {traceback.format_exc()}")

    print(f"\n处理完成!")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {(success_count / total_scenarios * 100):.1f}%")

    elapsed_time = time.time() - start_time
    print(f"总耗时: {elapsed_time:.1f}秒")

    if error_log and len(error_log) > 3:
        print(f"\n更多错误信息（{len(error_log)} 个错误）:")
        for i, error in enumerate(error_log[3:10]):  # 显示第4-10个错误
            print(f"   {i+4}. {error}")
        if len(error_log) > 10:
            print(f"   ... 和其他 {len(error_log) - 10} 个错误")

    return success_count, failed_count


def process_scenarios_by_type(scenario_types: List[str], max_count: int, output_base_dir: str,
                              offset: int = 0, shard: Optional[int] = None) -> None:
    """按场景类型处理场景数据。

    分片并行：传 shard（切片大小）时，强制 shuffle=False（确定性顺序，多进程切片不重不漏），
    每进程只处理该类型 scenarios[offset:offset+shard]。多个分片进程写同一 processed_<type> 目录
    （token 命名，无文件冲突）。不传 shard 时保持原行为（shuffle=True，处理前 max_count 个）。
    """
    paths = get_default_paths()
    config = create_default_config()

    print(f"目标场景类型: {scenario_types}")
    print(f"每类型最大处理数量: {max_count}")
    print(f"输出基础目录: {output_base_dir}")
    print()

    try:
        scenario_builder = NuPlanScenarioBuilder(
            data_root=paths['data_root'],
            map_root=paths['map_root'],
            sensor_root=paths['sensor_root'],
            db_files=None,
            map_version=paths['map_version']
        )
    except Exception as e:
        print(f"错误：无法创建场景构建器: {e}")
        return

    worker = Sequential()
    total_success = 0
    total_failed = 0

    for scenario_type in scenario_types:
        print(f"\n{'='*60}")
        print(f"处理场景类型: {scenario_type}")
        print(f"{'='*60}")

        sharding = shard is not None
        # 分片时只需取到切片末端（offset+shard），确定性顺序；否则取前 max_count（保留 shuffle）
        limit = (offset + shard) if sharding else max_count
        scenario_filter = ScenarioFilter(
            scenario_types=[scenario_type],
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=limit,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=(not sharding)
        )

        try:
            scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
        except Exception as e:
            print(f"错误：无法获取 {scenario_type} 场景: {e}")
            continue

        if sharding:
            scenarios = scenarios[offset:offset + shard]

        if not scenarios:
            print(f"警告：没有找到 {scenario_type} 类型的场景（offset={offset} shard={shard}）")
            continue

        print(f"找到 {len(scenarios)} 个 {scenario_type} 场景"
              + (f"（分片 offset={offset} shard={shard}）" if sharding else ""))

        # 为每种场景类型创建输出目录
        output_dir = Path(output_base_dir) / f"processed_{scenario_type}"

        # 处理场景
        success, failed = batch_process_with_progress(scenarios, config, output_dir)
        total_success += success
        total_failed += failed

        print(f"\n{scenario_type} 处理结果:")
        print(f"  成功: {success}")
        print(f"  失败: {failed}")
        print(f"  成功率: {(success / len(scenarios) * 100):.1f}%")

    print(f"\n{'='*60}")
    print(f"总体处理结果:")
    print(f"总成功: {total_success}")
    print(f"总失败: {total_failed}")
    if total_success + total_failed > 0:
        print(f"总成功率: {(total_success / (total_success + total_failed) * 100):.1f}%")
    print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='按场景类型处理nuPlan数据')
    parser.add_argument('--analyze', action='store_true', help='分析数据集中的场景类型分布')
    parser.add_argument('--process', action='store_true', help='处理指定类型的场景')
    parser.add_argument('--types', nargs='+', default=['high_magnitude_speed'],
                       help='要处理的场景类型列表')
    parser.add_argument('--count', type=int, default=1000,
                       help='每种类型处理的最大场景数量')
    parser.add_argument('--output-dir', default=config.PROCESSED_DATA_BY_TYPE_DIR,
                       help='输出目录')
    parser.add_argument('--offset', type=int, default=0,
                       help='[分片] 该类型 scenarios 切片起点索引')
    parser.add_argument('--shard', type=int, default=None,
                       help='[分片] 切片大小；设了即开分片模式（确定性顺序 shuffle=False，多进程切片并行）')

    args = parser.parse_args()

    if not args.analyze and not args.process:
        print("错误：请指定 --analyze 或 --process 参数")
        parser.print_help()
        return

    paths = get_default_paths()

    if args.analyze:
        print("分析模式：检查数据集中的场景类型分布")
        available_types = analyze_scenario_types(paths['data_root'])
        if available_types:
            print(f"\n可用的场景类型: {available_types}")
        return

    if args.process:
        print("处理模式：按指定类处理场景数据")
        process_scenarios_by_type(args.types, args.count, args.output_dir,
                                  offset=args.offset, shard=args.shard)


if __name__ == '__main__':
    main()

"""
使用说明：

1. 分析可用的场景类型：
   python process_by_scenario_type.py --analyze

2. 处理特定场景类型（默认high_magnitude_speed）：
   python process_by_scenario_type.py --process --types medium_magnitude_speed --count 20000
   
3. 处理多种场景类型：
   python process_by_scenario_type.py --process --types high_magnitude_speed low_magnitude_speed --count 2000
   
4. 自定义输出目录：
   python process_by_scenario_type.py --process --types high_magnitude_speed --count 5000 --output-dir /custom/path

输出结构：
- 输出目录：/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/data/processed_data_by_type/
- 每种场景类型会创建子目录：processed_<scenario_type>/
- 例如：data/processed_data_by_type/processed_high_magnitude_speed/

常用场景类型：
- high_magnitude_speed: 高速场景
- low_magnitude_speed: 低速场景  
- following_lane_with_lead: 跟车场景
- changing_lane: 变道场景
- starting_unprotected_cross_turn: 无保护左转场景
"""
