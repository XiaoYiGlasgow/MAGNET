#!/usr/bin/env python3
"""
分析MAGNET训练结果的脚本
"""
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import re

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

RESULTS_BASE_DIR = config.ALL_RESULTS_DIR

def extract_training_info_from_json(json_file_path: str) -> Dict:
    """从training_progress.json文件中提取训练信息"""
    if not os.path.exists(json_file_path):
        return {}

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从JSON中提取信息
        epochs = data.get('epochs', [])
        train_losses = data.get('train_losses', [])
        val_losses = data.get('val_losses', [])
        ades = data.get('val_ades', [])
        fdes = data.get('val_fdes', [])

        # 找到最佳模型
        best_ade_idx = ades.index(min(ades)) if ades else 0
        best_fde_idx = fdes.index(min(fdes)) if fdes else 0

        return {
            'total_epochs': len(epochs),
            'final_epoch': max(epochs) if epochs else 0,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'ades': ades,
            'fdes': fdes,
            'best_ade_epoch': epochs[best_ade_idx] if epochs and ades else 0,
            'best_fde_epoch': epochs[best_fde_idx] if epochs and fdes else 0,
            'best_ade': min(ades) if ades else 0,
            'best_fde': min(fdes) if fdes else 0
        }
    except:
        return {}

def extract_training_info_from_log(log_file_path: str) -> Dict:
    """从日志文件中提取训练信息"""
    if not os.path.exists(log_file_path):
        return {}
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用新的正则表达式匹配实际格式
        # 格式: Epoch  XX/YY: Train=X.XXXX, Val=X.XXXX, ADE=X.XXm, FDE=X.XXm
        epoch_pattern = r'Epoch\s+(\d+)/(\d+):\s+Train=([\d.]+),\s+Val=([\d.]+),\s+ADE=([\d.]+)m,\s+FDE=([\d.]+)m'
        matches = re.findall(epoch_pattern, content)
        
        if not matches:
            return {}
        
        epochs = []
        train_losses = []
        val_losses = []
        ades = []
        fdes = []
        
        for match in matches:
            epoch_num, total_epochs, train_loss, val_loss, ade, fde = match
            epochs.append(int(epoch_num))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            ades.append(float(ade))
            fdes.append(float(fde))
        
        # 找到最佳模型
        best_ade_idx = ades.index(min(ades)) if ades else 0
        best_fde_idx = fdes.index(min(fdes)) if fdes else 0
        
        return {
            'total_epochs': len(epochs),
            'final_epoch': max(epochs) if epochs else 0,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'ades': ades,
            'fdes': fdes,
            'best_ade_epoch': epochs[best_ade_idx] if epochs and ades else 0,
            'best_fde_epoch': epochs[best_fde_idx] if epochs and fdes else 0,
            'best_ade': min(ades) if ades else 0,
            'best_fde': min(fdes) if fdes else 0
        }
    except Exception as e:
        print(f"解析日志文件失败: {log_file_path}, 错误: {e}")
        return {}

def analyze_results_folder(base_path: str) -> Dict:
    """分析单个results文件夹"""
    results = {}
    
    # 查找training子文件夹
    try:
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path) and item.startswith('training_'):
                # 检查日志文件
                log_file = os.path.join(item_path, 'logs', 'training_log.txt')
                
                info = extract_training_info_from_log(log_file)
                
                if info:
                    results[item] = info
                    break  # 只取第一个有效结果
    except:
        pass
    
    return results

def find_training_logs(directory: str) -> List[str]:
    """查找给定目录下的所有training_log.txt文件"""
    training_logs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "training_log.txt":
                training_logs.append(os.path.join(root, file))
    return training_logs

def parse_training_log(file_path: str) -> pd.DataFrame:
    """解析单个training_log.txt文件为DataFrame"""
    pattern = r"Epoch\s+(\d+)/\d+:\s+Train=([\d.]+),\s+Val=([\d.]+),\s+ADE=([\d.]+)m,\s+FDE=([\d.]+)m"
    data = {"Epoch": [], "Train Loss": [], "Val Loss": [], "ADE": [], "FDE": []}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        matches = re.findall(pattern, content)
        for match in matches:
            epoch, train_loss, val_loss, ade, fde = match
            data["Epoch"].append(int(epoch))
            data["Train Loss"].append(float(train_loss))
            data["Val Loss"].append(float(val_loss))
            data["ADE"].append(float(ade))
            data["FDE"].append(float(fde))

    return pd.DataFrame(data)

def aggregate_logs(log_files: List[str]) -> pd.DataFrame:
    """将所有训练日志汇总到一个DataFrame中"""
    all_data = []
    for log_file in log_files:
        experiment_name = os.path.basename(os.path.dirname(log_file))
        df = parse_training_log(log_file)
        df["Experiment"] = experiment_name
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def save_to_csv(data: pd.DataFrame, output_path: str) -> None:
    """将汇总数据保存到CSV文件"""
    data.to_csv(output_path, index=False)

def main():
    base_dir = RESULTS_BASE_DIR
    output_csv = os.path.join(config.ANALYSIS_RESULTS_DIR, "training_summary.csv")
    
    # 定义场景类型和连接方式
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
    
    all_results = {}
    
    print("开始分析MAGNET训练结果...")
    
    # 分析每个结果文件夹
    for scenario in scenario_types:
        all_results[scenario] = {}
        for adjacency in adjacency_types:
            folder_name = f"results_type_{scenario}_{adjacency}"
            folder_path = os.path.join(base_dir, folder_name)
            
            if os.path.exists(folder_path):
                print(f"正在分析: {folder_name}")
                results = analyze_results_folder(folder_path)
                all_results[scenario][adjacency] = results
            else:
                print(f"文件夹不存在: {folder_name}")
    
    # 生成汇总报告
    print("\n" + "="*100)
    print("MAGNET训练结果汇总报告")
    print("="*100)
    
    # 创建汇总表格数据
    summary_data = []
    
    for scenario in scenario_types:
        scenario_display = scenario.replace('_', ' ').title()
        print(f"\n【场景类型: {scenario_display}】")
        print("-" * 80)
        
        for adjacency in adjacency_types:
            adj_name = adjacency.replace('_adjacency', '').replace('_', ' ').title()
            print(f"\n  连接方式: {adj_name}")
            
            if adjacency in all_results[scenario] and all_results[scenario][adjacency]:
                results = all_results[scenario][adjacency]
                
                # 取第一个有效结果
                first_result = list(results.values())[0]
                
                final_epoch = first_result.get('final_epoch', 0)
                best_ade_epoch = first_result.get('best_ade_epoch', 0)
                best_fde_epoch = first_result.get('best_fde_epoch', 0)
                best_ade = first_result.get('best_ade', 0)
                best_fde = first_result.get('best_fde', 0)
                
                print(f"    训练停止epoch: {final_epoch}")
                print(f"    最佳ADE epoch: {best_ade_epoch}")
                print(f"    最佳FDE epoch: {best_fde_epoch}")
                print(f"    最佳ADE: {best_ade:.4f}米" if best_ade > 0 else "    最佳ADE: N/A")
                print(f"    最佳FDE: {best_fde:.4f}米" if best_fde > 0 else "    最佳FDE: N/A")
                
                # 显示loss下降趋势
                train_losses = first_result.get('train_losses', [])
                val_losses = first_result.get('val_losses', [])
                
                if train_losses and val_losses:
                    print(f"    初始Train Loss: {train_losses[0]:.6f}")
                    print(f"    最终Train Loss: {train_losses[-1]:.6f}")
                    print(f"    Train Loss下降: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
                    print(f"    初始Val Loss: {val_losses[0]:.6f}")
                    print(f"    最终Val Loss: {val_losses[-1]:.6f}")
                    print(f"    Val Loss下降: {((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%")
                    
                    # 添加到汇总数据
                    summary_data.append({
                        '场景类型': scenario_display,
                        '连接方式': adj_name,
                        '训练停止epoch': final_epoch,
                        '最佳ADE_epoch': best_ade_epoch,
                        '最佳FDE_epoch': best_fde_epoch,
                        '最佳ADE(米)': f"{best_ade:.4f}" if best_ade > 0 else "N/A",
                        '最佳FDE(米)': f"{best_fde:.4f}" if best_fde > 0 else "N/A",
                        '初始Train_Loss': f"{train_losses[0]:.6f}",
                        '最终Train_Loss': f"{train_losses[-1]:.6f}",
                        'Train_Loss下降率': f"{((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%",
                        '初始Val_Loss': f"{val_losses[0]:.6f}",
                        '最终Val_Loss': f"{val_losses[-1]:.6f}",
                        'Val_Loss下降率': f"{((val_losses[0] - val_losses[-1]) / val_losses[0] * 100):.1f}%"
                    })
                else:
                    print("    无loss数据")
                    summary_data.append({
                        '场景类型': scenario_display,
                        '连接方式': adj_name,
                        '训练停止epoch': 'N/A',
                        '最佳ADE_epoch': 'N/A',
                        '最佳FDE_epoch': 'N/A',
                        '最佳ADE(米)': 'N/A',
                        '最佳FDE(米)': 'N/A',
                        '初始Train_Loss': 'N/A',
                        '最终Train_Loss': 'N/A',
                        'Train_Loss下降率': 'N/A',
                        '初始Val_Loss': 'N/A',
                        '最终Val_Loss': 'N/A',
                        'Val_Loss下降率': 'N/A'
                    })
            else:
                print("    无有效训练数据")
                summary_data.append({
                    '场景类型': scenario_display,
                    '连接方式': adj_name,
                    '训练停止epoch': 'N/A',
                    '最佳ADE_epoch': 'N/A',
                    '最佳FDE_epoch': 'N/A',
                    '最佳ADE(米)': 'N/A',
                    '最佳FDE(米)': 'N/A',
                    '初始Train_Loss': 'N/A',
                    '最终Train_Loss': 'N/A',
                    'Train_Loss下降率': 'N/A',
                    '初始Val_Loss': 'N/A',
                    '最终Val_Loss': 'N/A',
                    'Val_Loss下降率': 'N/A'
                })
    
    # 生成CSV汇总表格
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_file = os.path.join(base_dir, "training_results_summary.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n汇总表格已保存到: {csv_file}")
        
        # 显示汇总表格
        print("\n" + "="*150)
        print("详细汇总表格")
        print("="*150)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df.to_string(index=False))
    
    # 保存详细结果到JSON文件
    output_file = os.path.join(base_dir, "training_results_detailed.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 解析所有训练日志，提取指标，并将结果保存为CSV表格
    print("\n开始解析所有训练日志...")
    log_files = find_training_logs(logs_dir)
    if not log_files:
        print("未找到训练日志文件。")
    else:
        all_data = aggregate_logs(log_files)
        save_to_csv(all_data, output_csv)
        print(f"训练摘要已保存到: {output_csv}")

if __name__ == "__main__":
    main()
