import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import numpy as np

# ✅ 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset, get_current_adjacency_config

from setproctitle import setproctitle

setproctitle("Yi Xiao Demo")

# 🚀 Demo模式：限制场景数量
DEMO_SCENES = 100
DEMO_EPOCHS = 3

def restore_trajectory_from_increments(origin, increments):
    """
    将增量轨迹还原为绝对轨迹
    """
    batch_size, time_steps, _ = increments.shape
    absolute_trajectory = torch.zeros_like(increments)
    absolute_trajectory[:, 0, :] = origin + increments[:, 0, :]
    for i in range(1, time_steps):
        absolute_trajectory[:, i, :] = absolute_trajectory[:, i - 1, :] + increments[:, i, :]
    return absolute_trajectory

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            total_loss += loss.item() * len(batch.y)
            total_samples += len(batch.y)
    return total_loss / total_samples

def train():
    print("🚀 开始Demo训练 (限制100个场景)...")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    try:
        # 加载数据集
        print("📊 正在加载数据集...")
        dataset = GraphDataset(
            root=config.SAVE_PATH
        )
        
        # 🎯 限制数据集大小到100个场景
        if len(dataset) > DEMO_SCENES:
            indices = list(range(DEMO_SCENES))
            dataset = Subset(dataset, indices)
            print(f"🎯 Demo模式：使用 {len(dataset)} 个场景 (原数据集: {DEMO_SCENES}+)")
        else:
            print(f"📊 数据集大小: {len(dataset)} 个场景")
        
        # 划分训练和验证集 - 处理小数据集的情况
        if len(dataset) < 5:
            print("⚠️  数据集太小，使用全部数据进行训练，不划分验证集")
            train_dataset = dataset
            val_dataset = dataset  # 使用相同数据集进行验证
        else:
            train_indices, val_indices = train_test_split(
                range(len(dataset)), test_size=0.2, random_state=42
            )
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
        
        print(f"🏃 训练集: {len(train_dataset)} 个场景")
        print(f"✅ 验证集: {len(val_dataset)} 个场景")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # 初始化模型
        print("🧠 初始化模型...")
        model = TransformerPredictor(
            input_dim=config.POLYLINE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=0.1
        ).to(device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # 训练记录
        train_losses = []
        val_losses = []
        
        print(f"🎯 开始训练 (Demo: {DEMO_EPOCHS} epochs)...")
        
        for epoch in range(DEMO_EPOCHS):
            print(f"\n📅 Epoch {epoch+1}/{DEMO_EPOCHS}")
            
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            train_samples = 0
            
            for batch in tqdm(train_loader, desc="Training"):
                batch = batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item() * len(batch.y)
                train_samples += len(batch.y)
            
            avg_train_loss = epoch_train_loss / train_samples
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            avg_val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(avg_val_loss)
            
            print(f"📊 训练损失: {avg_train_loss:.6f}")
            print(f"📊 验证损失: {avg_val_loss:.6f}")
        
        # 保存模型和结果
        current_config = get_current_adjacency_config()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_name = f"demo_model_{current_config}_{timestamp}.pth"
        
        torch.save(model.state_dict(), model_name)
        print(f"💾 模型已保存: {model_name}")
        
        # 保存训练记录
        results = {
            'config': current_config,
            'demo_scenes': DEMO_SCENES,
            'demo_epochs': DEMO_EPOCHS,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss
        }
        
        results_name = f"demo_results_{current_config}_{timestamp}.json"
        with open(results_name, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 结果已保存: {results_name}")
        
        print("\n✅ Demo训练完成!")
        print(f"📈 最终训练损失: {avg_train_loss:.6f}")
        print(f"📈 最终验证损失: {avg_val_loss:.6f}")
        
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
