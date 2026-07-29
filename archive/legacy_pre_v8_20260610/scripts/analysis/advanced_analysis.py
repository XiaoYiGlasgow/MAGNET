#!/usr/bin/env python3
"""
Advanced analysis tools for GNN_Trans training results
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
RESULTS_BASE_DIR = config.ALL_RESULTS_DIR

def create_output_directory() -> str:
    """创建统一的输出目录"""
    output_dir = config.ANALYSIS_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_training_data() -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """按场景和连接方式分组加载训练数据"""
    base_dir = RESULTS_BASE_DIR
    
    scenario_types = [
        "high_lateral_acceleration",
        "high_magnitude_speed", 
        "medium_magnitude_speed",
        "near_multiple_vehicles",
        "traversing_intersection"
    ]
    
    adjacency_types = [
        "default_adjacency",
        "ego_only_adjacency", 
        "nearby_only_adjacency",
        "no_ego_road_adjacency"
    ]
    
    # 按场景分组的数据结构: scenario -> adjacency -> data
    grouped_data = {}
    
    for scenario in scenario_types:
        grouped_data[scenario] = {}
        for adjacency in adjacency_types:
            folder_name = f"results_type_{scenario}_{adjacency}"
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.exists(folder_path):
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isdir(item_path) and item.startswith('training_'):
                        log_file = os.path.join(item_path, 'logs', 'training_log.txt')
                        data = extract_training_data(log_file)
                        
                        if data:
                            grouped_data[scenario][adjacency] = data
                            break
    
    return grouped_data

def extract_training_data(log_file_path: str) -> Dict[str, List[float]]:
    """从日志文件提取训练数据"""
    if not os.path.exists(log_file_path):
        return {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'Epoch\s+(\d+)/\d+:\s+Train=([\d.]+),\s+Val=([\d.]+),\s+ADE=([\d.]+)m,\s+FDE=([\d.]+)m'
        matches = re.findall(pattern, content)
        
        if not matches:
            return {}
        
        epochs, train_losses, val_losses, ades, fdes = zip(*[
            (int(match[0]), float(match[1]), float(match[2]), float(match[3]), float(match[4]))
            for match in matches
        ])
        
        return {
            'epochs': list(epochs),
            'train_losses': list(train_losses),
            'val_losses': list(val_losses),
            'ades': list(ades),
            'fdes': list(fdes)
        }
    except Exception:
        return {}

def create_scenario_comparison_plots(grouped_data: Dict, output_dir: str) -> None:
    """为每个场景创建四种连接方式的对比图"""
    adjacency_labels = {
        'default_adjacency': 'Default',
        'ego_only_adjacency': 'Ego Only', 
        'nearby_only_adjacency': 'Nearby Only',
        'no_ego_road_adjacency': 'No Ego Road'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for scenario, adj_data in grouped_data.items():
        if not adj_data:
            continue
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        scenario_title = scenario.replace('_', ' ').title()
        fig.suptitle(f'{scenario_title}: Adjacency Configuration Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Training Loss
        for i, (adj_type, data) in enumerate(adj_data.items()):
            if data.get('train_losses'):
                epochs = data['epochs']
                losses = data['train_losses']
                ax1.plot(epochs, losses, color=colors[i % 4], 
                        label=adjacency_labels.get(adj_type, adj_type), 
                        linewidth=2, alpha=0.8)
        
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Validation Loss
        for i, (adj_type, data) in enumerate(adj_data.items()):
            if data.get('val_losses'):
                epochs = data['epochs']
                losses = data['val_losses']
                ax2.plot(epochs, losses, color=colors[i % 4], 
                        label=adjacency_labels.get(adj_type, adj_type), 
                        linewidth=2, alpha=0.8)
        
        ax2.set_title('Validation Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        
        # ADE
        for i, (adj_type, data) in enumerate(adj_data.items()):
            if data.get('ades'):
                epochs = data['epochs']
                ades = data['ades']
                ax3.plot(epochs, ades, color=colors[i % 4], 
                        label=adjacency_labels.get(adj_type, adj_type), 
                        linewidth=2, alpha=0.8)
        
        ax3.set_title('ADE (Average Displacement Error)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ADE (meters)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 100)
        
        # FDE
        for i, (adj_type, data) in enumerate(adj_data.items()):
            if data.get('fdes'):
                epochs = data['epochs']
                fdes = data['fdes']
                ax4.plot(epochs, fdes, color=colors[i % 4], 
                        label=adjacency_labels.get(adj_type, adj_type), 
                        linewidth=2, alpha=0.8)
        
        ax4.set_title('FDE (Final Displacement Error)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('FDE (meters)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 100)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f'scenario_comparison_{scenario}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"场景对比图已保存: {filename}")

def create_adjacency_comparison_plots(grouped_data: Dict, output_dir: str) -> None:
    """为每种连接方式创建不同场景的对比图"""
    scenario_labels = {
        'high_lateral_acceleration': 'High Lateral Acc',
        'high_magnitude_speed': 'High Speed',
        'medium_magnitude_speed': 'Medium Speed', 
        'near_multiple_vehicles': 'Near Multiple Vehicles',
        'traversing_intersection': 'Traversing Intersection'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    adjacency_types = ['default_adjacency', 'ego_only_adjacency', 
                      'nearby_only_adjacency', 'no_ego_road_adjacency']
    
    for adj_type in adjacency_types:
        adj_title = adj_type.replace('_adjacency', '').replace('_', ' ').title()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{adj_title} Configuration: Scenario Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Training Loss
        for i, (scenario, adj_data) in enumerate(grouped_data.items()):
            if adj_type in adj_data and adj_data[adj_type].get('train_losses'):
                data = adj_data[adj_type]
                epochs = data['epochs']
                losses = data['train_losses']
                ax1.plot(epochs, losses, color=colors[i % 5], 
                        label=scenario_labels.get(scenario, scenario), 
                        linewidth=2, alpha=0.8)
        
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 100)
        
        # Validation Loss
        for i, (scenario, adj_data) in enumerate(grouped_data.items()):
            if adj_type in adj_data and adj_data[adj_type].get('val_losses'):
                data = adj_data[adj_type]
                epochs = data['epochs']
                losses = data['val_losses']
                ax2.plot(epochs, losses, color=colors[i % 5], 
                        label=scenario_labels.get(scenario, scenario), 
                        linewidth=2, alpha=0.8)
        
        ax2.set_title('Validation Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 100)
        
        # ADE
        for i, (scenario, adj_data) in enumerate(grouped_data.items()):
            if adj_type in adj_data and adj_data[adj_type].get('ades'):
                data = adj_data[adj_type]
                epochs = data['epochs']
                ades = data['ades']
                ax3.plot(epochs, ades, color=colors[i % 5], 
                        label=scenario_labels.get(scenario, scenario), 
                        linewidth=2, alpha=0.8)
        
        ax3.set_title('ADE (Average Displacement Error)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ADE (meters)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 100)
        
        # FDE
        for i, (scenario, adj_data) in enumerate(grouped_data.items()):
            if adj_type in adj_data and adj_data[adj_type].get('fdes'):
                data = adj_data[adj_type]
                epochs = data['epochs']
                fdes = data['fdes']
                ax4.plot(epochs, fdes, color=colors[i % 5], 
                        label=scenario_labels.get(scenario, scenario), 
                        linewidth=2, alpha=0.8)
        
        ax4.set_title('FDE (Final Displacement Error)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('FDE (meters)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 100)
        
        plt.tight_layout()
        
        # 保存图片
        filename = f'adjacency_comparison_{adj_type}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"连接方式对比图已保存: {filename}")

def create_performance_summary_table(grouped_data: Dict, output_dir: str) -> None:
    """创建性能汇总表"""
    summary_data = []
    
    for scenario, adj_data in grouped_data.items():
        for adj_type, data in adj_data.items():
            if data.get('ades') and data.get('fdes'):
                best_ade = min(data['ades'])
                best_fde = min(data['fdes'])
                best_ade_epoch = data['epochs'][data['ades'].index(best_ade)]
                best_fde_epoch = data['epochs'][data['fdes'].index(best_fde)]
                final_epoch = max(data['epochs'])
                
                summary_data.append({
                    'Scenario': scenario.replace('_', ' ').title(),
                    'Adjacency_Type': adj_type.replace('_adjacency', '').replace('_', ' ').title(),
                    'Best_ADE': best_ade,
                    'Best_ADE_Epoch': best_ade_epoch,
                    'Best_FDE': best_fde,
                    'Best_FDE_Epoch': best_fde_epoch,
                    'Final_Epoch': final_epoch,
                    'Final_Train_Loss': data['train_losses'][-1] if data.get('train_losses') else 0,
                    'Final_Val_Loss': data['val_losses'][-1] if data.get('val_losses') else 0
                })
    
    df = pd.DataFrame(summary_data)
    
    # 保存详细表格
    csv_path = os.path.join(output_dir, 'performance_summary_detailed.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 创建透视表 - 场景vs连接方式的ADE对比
    pivot_ade = df.pivot(index='Scenario', columns='Adjacency_Type', values='Best_ADE')
    pivot_fde = df.pivot(index='Scenario', columns='Adjacency_Type', values='Best_FDE')
    
    # 保存透视表
    pivot_ade.to_csv(os.path.join(output_dir, 'ade_comparison_matrix.csv'), encoding='utf-8-sig')
    pivot_fde.to_csv(os.path.join(output_dir, 'fde_comparison_matrix.csv'), encoding='utf-8-sig')
    
    print("性能汇总表已保存")
    print("\nADE对比矩阵 (场景 vs 连接方式):")
    print(pivot_ade.round(4))
    print("\nFDE对比矩阵 (场景 vs 连接方式):")
    print(pivot_fde.round(4))

def create_best_configuration_analysis(grouped_data: Dict, output_dir: str) -> None:
    """分析最佳配置"""
    results = []
    
    for scenario, adj_data in grouped_data.items():
        scenario_results = {}
        for adj_type, data in adj_data.items():
            if data.get('ades'):
                best_ade = min(data['ades'])
                scenario_results[adj_type] = best_ade
        
        if scenario_results:
            best_adj = min(scenario_results, key=scenario_results.get)
            best_ade = scenario_results[best_adj]
            
            results.append({
                'Scenario': scenario.replace('_', ' ').title(),
                'Best_Adjacency': best_adj.replace('_adjacency', '').replace('_', ' ').title(),
                'Best_ADE': best_ade,
                'Improvement_vs_Default': (scenario_results.get('default_adjacency', best_ade) - best_ade) / scenario_results.get('default_adjacency', best_ade) * 100 if scenario_results.get('default_adjacency') else 0
            })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'best_configurations.csv'), index=False, encoding='utf-8-sig')
    
    print("\n最佳配置分析:")
    print(df.to_string(index=False))

def main():
    """主函数"""
    print("正在加载训练数据...")
    grouped_data = load_training_data()
    
    if not grouped_data:
        print("未找到训练数据!")
        return
    
    total_experiments = sum(len(adj_data) for adj_data in grouped_data.values())
    print(f"已加载 {len(grouped_data)} 个场景，共 {total_experiments} 个实验")
    
    # 创建输出目录
    output_dir = create_output_directory()
    print(f"输出目录: {output_dir}")
    
    print("正在创建场景对比图...")
    create_scenario_comparison_plots(grouped_data, output_dir)
    
    print("正在创建连接方式对比图...")
    create_adjacency_comparison_plots(grouped_data, output_dir)
    
    print("正在创建性能汇总表...")
    create_performance_summary_table(grouped_data, output_dir)
    
    print("正在分析最佳配置...")
    create_best_configuration_analysis(grouped_data, output_dir)
    
    print("所有分析已完成!")

if __name__ == "__main__":
    main()
