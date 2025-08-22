#!/usr/bin/env python3
"""
GNN_Trans配置切换和训练脚本
快速切换不同邻接矩阵配置并进行训练
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

def modify_adjacency_config(config_name):
    """
    修改dataset.py中的邻接矩阵配置

    Args:
        config_name: 配置名称，可选:
            - "default": 默认全连接
            - "ego_only": 只有ego连接其他元素
            - "no_ego_road": ego不连接道路元素
            - "nearby_only": 只连接15米内元素
    """
    dataset_file = Path('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/utils/dataset.py')

    # 读取文件内容
    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找并替换配置行
    old_line = 'custom_builder = get_adjacency_config("default")'
    new_line = f'custom_builder = get_adjacency_config("{config_name}")'

    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        # 如果找不到默认配置，查找任何get_adjacency_config调用
        import re
        pattern = r'custom_builder = get_adjacency_config\("[\w_]+"\)'
        content = re.sub(pattern, new_line, content)

    # 写回文件
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 已切换到邻接矩阵配置: {config_name}")

def clean_processed_data():
    """删除已处理的数据，强制重新处理"""
    print("ℹ️  现在数据按配置分类保存，无需清理！")
    print("   每种配置的数据会自动保存在独立文件夹中：")
    print("   - processed/config_default/")
    print("   - processed/config_ego_only/")
    print("   - processed/config_no_ego_road/")
    print("   - processed/config_nearby_only/")
    return True  # 返回True表示"清理"成功

def run_training():
    """运行训练脚本"""
    print("🚀 开始训练...")
    train_script = '/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/train.py'

    try:
        # 使用实时输出，不缓冲
        process = subprocess.Popen(['python', train_script],
                                 cwd='/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)

        # 实时显示输出
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())

        process.wait()

        if process.returncode == 0:
            print("✅ 训练完成!")
        else:
            print("❌ 训练失败!")

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")

def run_evaluation():
    """运行评估脚本"""
    print("📊 开始评估...")
    eval_script = '/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/evaluate.py'

    try:
        # 同样使用实时输出
        process = subprocess.Popen(['python', eval_script],
                                 cwd='/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)

        # 实时显示输出
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())

        process.wait()

        if process.returncode == 0:
            print("✅ 评估完成!")
        else:
            print("❌ 评估失败!")

    except Exception as e:
        print(f"❌ 评估过程出错: {e}")

def main():
    """主函数"""
    print("🔧 GNN_Trans 配置切换和训练工具")
    print("=" * 50)

    # 可用的配置选项
    configs = {
        "1": ("default", "默认全连接模式"),
        "2": ("ego_only", "ego为中心的星型连接"),
        "3": ("no_ego_road", "禁止ego连接道路元素"),
        "4": ("nearby_only", "只连接15米内的元素")
    }

    print("可用的邻接矩阵配置:")
    for key, (config_name, description) in configs.items():
        print(f"  {key}. {config_name} - {description}")

    # 用户选择配置
    choice = input("\n请选择配置 (1-4): ").strip()

    if choice not in configs:
        print("❌ 无效选择!")
        return

    config_name, description = configs[choice]
    print(f"\n选择的配置: {config_name} - {description}")

    # 确认是否继续
    confirm = input("是否继续? (y/n): ").strip().lower()
    if confirm != 'y':
        print("操作取消")
        return

    # 执行步骤
    print("\n开始执行...")

    # 1. 修改配置
    modify_adjacency_config(config_name)

    # 2. 清理已处理的数据
    clean_processed_data()

    # 3. 询问是否运行训练
    train_choice = input("是否运行训练? (y/n): ").strip().lower()
    if train_choice == 'y':
        run_training()

        # 4. 询问是否运行评估
        eval_choice = input("是否运行评估? (y/n): ").strip().lower()
        if eval_choice == 'y':
            run_evaluation()

    print(f"\n✅ 所有操作完成!")
    print(f"当前配置: {config_name}")
    print("生成的文件将包含配置名称，便于后续查看和对比")

if __name__ == "__main__":
    main()
