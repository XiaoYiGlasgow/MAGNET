#!/usr/bin/env python3
"""
simple_nuboard_launcher.py - 简化的 nuBoard 启动器
解决配置问题，直接启动nuBoard
"""

import os
import sys
import subprocess
import glob
from pathlib import Path


def find_latest_simulation():
    """查找最新的仿真结果"""
    output_base = "./nuplan_simulation_output"

    if not os.path.exists(output_base):
        print(f"❌ 找不到仿真输出目录: {output_base}")
        return None

    # 查找所有仿真结果目录
    pattern = os.path.join(output_base, "gnn_transformer_simulation_*")
    result_dirs = glob.glob(pattern)

    if not result_dirs:
        print(f"❌ 在 {output_base} 中找不到仿真结果")
        return None

    # 按时间排序，获取最新的
    result_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = result_dirs[0]

    print(f"📁 找到最新仿真结果: {latest_dir}")
    return latest_dir


def check_simulation_files(sim_path):
    """检查仿真文件是否存在"""
    required_dirs = ["simulation", "metrics", "aggregator_metric"]
    found_files = False

    for dir_name in required_dirs:
        dir_path = os.path.join(sim_path, dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            if len(files) > 0:
                print(f"✅ {dir_name}: {len(files)} 个文件")
                found_files = True
            else:
                print(f"⚠️ {dir_name}: 目录为空")
        else:
            print(f"❌ {dir_name}: 目录不存在")

    return found_files


def launch_nuboard_subprocess(simulation_path, port=5006):
    """使用subprocess启动nuBoard"""
    try:
        # 设置环境变量
        env = os.environ.copy()

        # 构建命令
        cmd = [
            sys.executable,  # 使用当前Python解释器
            '-m', 'nuplan.planning.script.run_nuboard',
            f'simulation_path=["{os.path.abspath(simulation_path)}"]',
            f'port_number={port}'
        ]

        print(f"🚀 启动命令: {' '.join(cmd)}")
        print(f"🌐 启动后请访问: http://localhost:{port}")

        # 启动进程
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("⏳ 正在启动nuBoard...")

        # 等待启动或错误
        try:
            stdout, stderr = process.communicate(timeout=30)
            if process.returncode != 0:
                print(f"❌ nuBoard启动失败")
                print(f"错误输出: {stderr}")
                return False
            else:
                print("✅ nuBoard启动成功!")
                return True
        except subprocess.TimeoutExpired:
            print("✅ nuBoard正在运行中...")
            print(f"🌐 请在浏览器中访问: http://localhost:{port}")
            return True

    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False


def launch_nuboard_direct():
    """直接运行nuBoard模块"""
    try:
        import hydra
        from omegaconf import DictConfig

        # 查找最新仿真结果
        sim_path = find_latest_simulation()
        if not sim_path:
            return False

        # 检查文件
        if not check_simulation_files(sim_path):
            print("⚠️ 仿真文件不完整，但仍尝试启动...")

        # 创建配置
        cfg = DictConfig({
            'simulation_path': [os.path.abspath(sim_path)],
            'port_number': 5006,
            'resource_prefix': '',
        })

        print(f"📁 仿真路径: {cfg.simulation_path[0]}")
        print(f"🌐 端口: {cfg.port_number}")

        # 尝试导入并运行nuBoard
        try:
            from nuplan.planning.script.run_nuboard import main as main_nuboard
            print("🚀 正在启动nuBoard...")
            main_nuboard(cfg)
            return True
        except Exception as e:
            print(f"❌ 直接启动失败: {e}")
            return False

    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False


def main():
    """主函数"""
    print("🎯 nuBoard 启动器")

    # 方法1: 查找并启动最新仿真结果
    sim_path = find_latest_simulation()

    if sim_path:
        print(f"\n📊 检查仿真文件:")
        check_simulation_files(sim_path)

        print(f"\n🚀 尝试启动nuBoard...")

        # 先尝试subprocess方式
        success = launch_nuboard_subprocess(sim_path)

        if not success:
            print("\n🔄 subprocess方式失败，尝试直接启动...")
            success = launch_nuboard_direct()

        if not success:
            print(f"\n💡 手动启动方法:")
            print(f"   cd /mnt/local/Xiao_Yi/nuplan-devkit")
            print(
                f"   python -m nuplan.planning.script.run_nuboard simulation_path='[\"{os.path.abspath(sim_path)}\"]'")
            print(f"   ")
            print(f"   或者:")
            print(
                f"   python nuplan/planning/script/run_nuboard.py simulation_path='[\"{os.path.abspath(sim_path)}\"]'")

    else:
        print("\n❌ 未找到仿真结果")
        print("💡 请先运行: python run_gnn_simulation.py")


if __name__ == "__main__":
    main()