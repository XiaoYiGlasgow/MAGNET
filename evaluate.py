#!/usr/bin/env python3

"""
GNN_Trans模型评估脚本
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import json
from pathlib import Path

# 添加路径
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset

def evaluate_model():
    """评估模型性能"""
    print("🚀 开始评估GNN_Trans模型...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_dataset = GraphDataset(config.TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 加载模型
    model = TransformerPredictor().to(device)
    
    # 查找最新的模型权重
    weight_path = Path(config.WEIGHT_PATH)
    weight_files = list(weight_path.glob("*.pth"))
    
    if not weight_files:
        print("❌ 未找到训练好的模型权重")
        return
    
    latest_weight = max(weight_files, key=os.path.getctime)
    print(f"加载模型权重: {latest_weight}")
    
    model.load_state_dict(torch.load(latest_weight, map_location=device))
    model.eval()
    
    # 评估指标
    total_loss = 0.0
    ade_list = []  # Average Displacement Error
    fde_list = []  # Final Displacement Error
    num_samples = 0
    
    print("开始评估...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            
            # 获取真实轨迹
            batch_size = data.y.shape[0]
            y_true = data.y.view(batch_size, -1).to(device)
            
            # MSE损失
            mse_loss = torch.nn.functional.mse_loss(output, y_true)
            total_loss += mse_loss.item() * batch_size
            
            # 计算ADE和FDE
            pred_traj = output.view(batch_size, -1, 2)  # [batch, time_steps, 2]
            true_traj = y_true.view(batch_size, -1, 2)
            
            # ADE: 平均位移误差
            displacement_errors = torch.norm(pred_traj - true_traj, dim=2)  # [batch, time_steps]
            ade = torch.mean(displacement_errors, dim=1)  # [batch]
            ade_list.extend(ade.cpu().numpy())
            
            # FDE: 最终位移误差
            fde = displacement_errors[:, -1]  # [batch]
            fde_list.extend(fde.cpu().numpy())
            
            num_samples += batch_size
            
            if i % 50 == 0:
                print(f"已评估: {i * batch_size}/{len(test_dataset)} 样本")
    
    # 计算最终指标
    avg_mse = total_loss / num_samples
    avg_ade = np.mean(ade_list)
    avg_fde = np.mean(fde_list)
    
    print(f"\n📊 评估结果:")
    print(f"   MSE Loss: {avg_mse:.6f}")
    print(f"   ADE (平均位移误差): {avg_ade:.4f} 米")
    print(f"   FDE (最终位移误差): {avg_fde:.4f} 米")
    
    # 保存评估结果
    results = {
        'mse_loss': avg_mse,
        'ade': avg_ade,
        'fde': avg_fde,
        'num_samples': num_samples
    }
    
    result_file = Path(config.WEIGHT_PATH) / 'evaluation_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 评估结果已保存到: {result_file}")
    
    # 绘制误差分布图
    plot_error_distribution(ade_list, fde_list)
    
    return results

def plot_error_distribution(ade_list, fde_list):
    """绘制误差分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ADE分布
    ax1.hist(ade_list, bins=50, alpha=0.7, color='blue')
    ax1.set_xlabel('ADE (米)')
    ax1.set_ylabel('频次')
    ax1.set_title('平均位移误差分布')
    ax1.axvline(np.mean(ade_list), color='red', linestyle='--', label=f'均值: {np.mean(ade_list):.3f}')
    ax1.legend()
    
    # FDE分布
    ax2.hist(fde_list, bins=50, alpha=0.7, color='green')
    ax2.set_xlabel('FDE (米)')
    ax2.set_ylabel('频次')
    ax2.set_title('最终位移误差分布')
    ax2.axvline(np.mean(fde_list), color='red', linestyle='--', label=f'均值: {np.mean(fde_list):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = Path(config.WEIGHT_PATH) / 'error_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"误差分布图已保存到: {plot_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()
