#!/usr/bin/env python3
"""
启动nuboard演示训练好的GNN_Trans模型
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """设置环境变量和路径"""
    # 确保在正确的conda环境中
    if 'nuplan' not in os.environ.get('CONDA_DEFAULT_ENV', ''):
        print("⚠️ 请确保激活nuplan conda环境")
        print("运行: conda activate nuplan")
    
    # 设置项目路径
    project_root = "/mnt/local/Xiao_Yi/nuplan-devkit"
    os.chdir(project_root)
    
    return project_root

def run_simulation_with_gnn_trans():
    """运行带有GNN_Trans模型的仿真"""
    
    cmd = [
        "python", "nuplan/planning/script/run_simulation.py",
        "+simulation=closed_loop_nonreactive_agents",
        "model=gnn_transformer_model",
        "planner=gnn_transformer_planner",
        "scenario_builder=nuplan",
        "scenario_filter=all_scenarios", 
        "scenario_filter.limit_total_scenarios=5",  # 限制5个场景用于演示
        "scenario_filter.scenario_types=[starting_straight_traffic_light_intersection_traversal]",
        "scenario_builder.data_root=/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini",
        "scenario_builder.map_root=/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/maps",
        "worker=sequential",
        "verbose=true"
    ]
    
    print("🚀 开始运行GNN_Trans模型仿真...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ 仿真完成")
            return True
        else:
            print(f"❌ 仿真失败: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 仿真超时")
        return False
    except Exception as e:
        print(f"❌ 运行仿真时出错: {e}")
        return False

def launch_nuboard():
    """启动nuboard可视化"""
    
    # 查找最新的仿真结果
    output_dir = Path("./outputs")
    if not output_dir.exists():
        print("❌ 未找到仿真输出目录")
        return False
    
    # 查找.nuboard文件
    nuboard_files = list(output_dir.rglob("*.nuboard"))
    if not nuboard_files:
        print("❌ 未找到.nuboard文件")
        return False
    
    latest_nuboard = max(nuboard_files, key=os.path.getmtime)
    print(f"📊 启动nuboard: {latest_nuboard}")
    
    cmd = [
        "python", "nuplan/planning/script/run_nuboard.py", 
        f"simulation_path={latest_nuboard.parent}",
        "port_number=5006"
    ]
    
    print("🌐 启动nuboard可视化界面...")
    print("访问地址: http://localhost:5006")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 nuboard已关闭")

def main():
    """主函数"""
    print("🎯 GNN_Trans模型nuboard演示")
    
    # 设置环境
    project_root = setup_environment()
    
    # 选择模式
    print("\n选择演示模式:")
    print("1. 运行新的仿真并启动nuboard")
    print("2. 直接启动nuboard (使用现有结果)")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        print("\n🔄 模式1: 运行新仿真")
        if run_simulation_with_gnn_trans():
            launch_nuboard()
        else:
            print("❌ 仿真失败，无法启动nuboard")
    
    elif choice == "2":
        print("\n📊 模式2: 直接启动nuboard")
        launch_nuboard()
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main()
