#!/usr/bin/env python3
"""
GNN_Trans 快速Demo测试脚本
基于原始train.py，限制处理100个场景进行快速测试
所有输出存储在demo文件夹中
支持切换邻接矩阵连接模式
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset, get_current_adjacency_config
from adjacency_configs import get_adjacency_config
from setproctitle import setproctitle

setproctitle("Yi Xiao Demo")

# Demo配置
DEMO_SCENES = 100
DEMO_EPOCHS = 3

# 邻接矩阵连接模式选择
# 可选模式: "default", "ego_only", "no_ego_road", "nearby_only", "selective"
CONNECTION_MODE = "default"  # 在这里修改你想要的连接模式

def create_demo_folders(connection_mode):
    """创建demo文件夹结构，文件夹名包含连接模式"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    demo_dir = f"demo_results_{timestamp}_{connection_mode}"

    folders = {
        'base': demo_dir,
        'models': os.path.join(demo_dir, 'models'),
        'results': os.path.join(demo_dir, 'results'),
        'logs': os.path.join(demo_dir, 'logs'),
        'data': os.path.join(demo_dir, 'data')
    }

    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)

    print(f"📁 创建demo文件夹: {demo_dir}")
    print(f"🔗 使用连接模式: {connection_mode}")
    return folders

def restore_trajectory_from_increments(origin, increments):
    """
    将增量轨迹还原为绝对轨迹
    """
    batch_size, time_steps, _ = increments.shape
    absolute_trajectory = torch.zeros_like(increments)

    # 第一个点：起始位置 + 第一个增量
    absolute_trajectory[:, 0, :] = origin + increments[:, 0, :]

    # 后续点：累加增量
    for i in range(1, time_steps):
        absolute_trajectory[:, i, :] = absolute_trajectory[:, i - 1, :] + increments[:, i, :]

    return absolute_trajectory

def quick_validate(model, val_loader, criterion, device):
    """包含ADE和FDE计算的验证函数"""
    model.eval()
    val_loss = 0.0
    ade_list = []
    fde_list = []
    num_samples = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)

            batch_size = data.y.shape[0]
            y = data.y.view(batch_size, -1).to(device)

            if output.shape[0] != batch_size:
                output = output.expand(batch_size, -1)

            loss = criterion(output, y)
            val_loss += loss.item() * batch_size

            # 计算ADE/FDE
            pred_increments = output.view(batch_size, -1, 2)  # [batch, time_steps, 2]
            true_increments = y.view(batch_size, -1, 2)

            # 起始位置为原点
            origin = torch.zeros(batch_size, 2, device=device)

            # 还原为绝对轨迹
            pred_absolute = restore_trajectory_from_increments(origin, pred_increments)
            true_absolute = restore_trajectory_from_increments(origin, true_increments)

            # 计算绝对轨迹的ADE和FDE
            displacement_errors = torch.norm(pred_absolute - true_absolute, dim=2)  # [batch, time_steps]
            ade = torch.mean(displacement_errors, dim=1)  # [batch]
            fde = displacement_errors[:, -1]  # [batch]

            ade_list.extend(ade.cpu().numpy())
            fde_list.extend(fde.cpu().numpy())
            num_samples += batch_size

    model.train()
    avg_ade = sum(ade_list) / len(ade_list) if ade_list else 0
    avg_fde = sum(fde_list) / len(fde_list) if fde_list else 0
    return val_loss / num_samples if num_samples > 0 else 0, avg_ade, avg_fde

def select_adjacency_config():
    """让用户选择邻接矩阵配置"""
    configs = {
        "1": ("default", "默认全连接模式"),
        "2": ("ego_only", "ego为中心的星型连接"),
        "3": ("no_ego_road", "禁止ego连接道路元素"),
        "4": ("nearby_only", "只连接15米内的元素")
    }

    print("\n🔧 可用的邻接矩阵配置:")
    for key, (config_name, description) in configs.items():
        print(f"  {key}. {config_name} - {description}")

    while True:
        choice = input("\n请选择配置 (1-4): ").strip()
        if choice in configs:
            config_name, description = configs[choice]
            print(f"✅ 选择了配置: {config_name} - {description}")
            return config_name
        else:
            print("❌ 无效选择，请重新输入!")

def set_adjacency_config(config_name):
    """设置邻接矩阵配置"""
    from utils.dataset import get_adjacency_config

    # 通过调用函数来设置配置
    get_adjacency_config(config_name)
    print(f"🔧 已切换到邻接矩阵配置: {config_name}")

def main():
    print("🚀 GNN_Trans 快速Demo测试")
    print("=" * 50)
    print(f"🎯 限制场景数: {DEMO_SCENES}")
    print(f"⚡ 训练轮数: {DEMO_EPOCHS}")

    # 选择邻接矩阵配置
    config_name = select_adjacency_config()
    set_adjacency_config(config_name)

    # 创建demo文件夹 - 使用用户选择的配置名称
    demo_folders = create_demo_folders(config_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")

    try:
        # 加载数据集
        print("📊 加载数据集...")
        full_dataset = GraphDataset(config.TRAIN_PATH)
        print(f"📊 原始数据集: {len(full_dataset)} 个样本")

        # 限制数据集大小
        if len(full_dataset) > DEMO_SCENES:
            demo_indices = list(range(DEMO_SCENES))
            demo_dataset = Subset(full_dataset, demo_indices)
            print(f"🎯 Demo数据集: {len(demo_dataset)} 个样本")
        else:
            demo_dataset = full_dataset
            print(f"📊 使用全部数据: {len(demo_dataset)} 个样本")

        # 保存demo数据集信息
        data_info = {
            'total_samples': len(full_dataset),
            'demo_samples': len(demo_dataset),
            'demo_indices': demo_indices if len(full_dataset) > DEMO_SCENES else list(range(len(demo_dataset)))
        }
        with open(os.path.join(demo_folders['data'], 'dataset_info.json'), 'w') as f:
            json.dump(data_info, f, indent=2)

        # 划分训练验证集
        if len(demo_dataset) < 5:
            print("⚠️  数据集太小，跳过验证集划分")
            train_dataset = demo_dataset
            val_dataset = demo_dataset
            train_indices = val_indices = list(range(len(demo_dataset)))
        else:
            indices = list(range(len(demo_dataset)))
            train_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=config.SEED
            )
            train_dataset = Subset(demo_dataset, train_indices)
            val_dataset = Subset(demo_dataset, val_indices)

        print(f"🏃 训练集: {len(train_dataset)} 个样本")
        print(f"✅ 验证集: {len(val_dataset)} 个样本")

        # 保存数据集划分信息
        split_info = {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        }
        with open(os.path.join(demo_folders['data'], 'data_split.json'), 'w') as f:
            json.dump(split_info, f, indent=2)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

        # 初始化模型
        print("🧠 初始化模型...")
        model = TransformerPredictor().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 训练记录
        train_losses = []
        val_losses = []

        print(f"🎯 开始训练 ({DEMO_EPOCHS} epochs)...")

        for epoch in range(DEMO_EPOCHS):
            print(f"\n📅 Epoch {epoch+1}/{DEMO_EPOCHS}")

            # 训练阶段
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for data in tqdm(train_loader, desc="Training"):
                data = data.to(device)
                optimizer.zero_grad()

                output = model(data)
                batch_size = data.y.shape[0]
                y = data.y.view(batch_size, -1).to(device)

                if output.shape[0] != batch_size:
                    output = output.expand(batch_size, -1)

                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)

            # 验证阶段
            avg_val_loss, avg_ade, avg_fde = quick_validate(model, val_loader, criterion, device)
            val_losses.append(avg_val_loss)

            print(f"📊 训练损失: {avg_train_loss:.6f}")
            print(f"📊 验证损失: {avg_val_loss:.6f}")
            print(f"📏 平均ADE: {avg_ade:.6f}")
            print(f"📏 平均FDE: {avg_fde:.6f}")

        # 保存结果到demo文件夹
        current_config = get_current_adjacency_config()

        # 保存模型
        model_name = f"demo_model_{current_config}.pth"
        model_path = os.path.join(demo_folders['models'], model_name)
        torch.save(model.state_dict(), model_path)
        print(f"💾 模型已保存: {model_path}")

        # 保存训练记录
        results = {
            'adjacency_config': config_name,  # 添加邻接矩阵配置信息
            'config': current_config,
            'demo_scenes': DEMO_SCENES,
            'demo_epochs': DEMO_EPOCHS,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'avg_ade': avg_ade,
            'avg_fde': avg_fde,
            'model_path': model_path,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        results_path = os.path.join(demo_folders['results'], 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 结果已保存: {results_path}")

        # 创建训练日志
        log_content = f"""Demo训练日志
================
时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
邻接矩阵配置: {config_name}
配置: {current_config}
场景数: {DEMO_SCENES}
训练轮数: {DEMO_EPOCHS}
设备: {device}

数据集信息:
- 原始数据集: {len(full_dataset)} 个样本
- Demo数据集: {len(demo_dataset)} 个样本
- 训练集: {len(train_dataset)} 个样本
- 验证集: {len(val_dataset)} 个样本

训练结果:
- 最终训练损失: {avg_train_loss:.6f}
- 最终验证损失: {avg_val_loss:.6f}
- 平均ADE: {avg_ade:.6f}米
- 平均FDE: {avg_fde:.6f}米

文件保存位置:
- 模型: {model_path}
- 结果: {results_path}
- 数据信息: {os.path.join(demo_folders['data'], 'dataset_info.json')}
- 数据划分: {os.path.join(demo_folders['data'], 'data_split.json')}
"""

        log_path = os.path.join(demo_folders['logs'], 'training_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        print(f"📝 日志已保存: {log_path}")

        print("\n✅ Demo训练完成!")
        print(f"📁 所有文件保存在: {demo_folders['base']}")
        print(f"🔧 使用的邻接矩阵配置: {config_name}")
        print(f"📈 最终训练损失: {avg_train_loss:.6f}")
        print(f"📈 最终验证损失: {avg_val_loss:.6f}")
        print(f"📏 平均ADE: {avg_ade:.6f}米")
        print(f"📏 平均FDE: {avg_fde:.6f}米")

    except Exception as e:
        print(f"❌ Demo训练失败: {e}")
        import traceback
        error_log = traceback.format_exc()

        # 保存错误日志
        if 'demo_folders' in locals():
            error_path = os.path.join(demo_folders['logs'], 'error_log.txt')
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"错误时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误信息: {str(e)}\n\n")
                f.write("详细错误:\n")
                f.write(error_log)
            print(f"❌ 错误日志已保存: {error_path}")

        print(error_log)

if __name__ == "__main__":
    main()
