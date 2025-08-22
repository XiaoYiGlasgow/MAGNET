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
import logging

# ✅ 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset
from adjacency_configs import get_adjacency_config

# 直接从utils.dataset导入get_current_adjacency_config
try:
    from utils.dataset import get_current_adjacency_config
except ImportError:
    # 如果导入失败，定义一个备用函数
    def get_current_adjacency_config():
        return "default"

from setproctitle import setproctitle

setproctitle("Yi Xiao")

def setup_logging(log_file_path):
    """设置详细的日志记录"""
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)

    # 移除已有的handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_progress_info(folders, epoch, total_epochs, train_loss, val_loss, ade, fde,
                      learning_rate, elapsed_time, estimated_remaining):
    """保存详细的训练进度信息"""
    progress_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epoch': epoch + 1,
        'total_epochs': total_epochs,
        'progress_percentage': ((epoch + 1) / total_epochs) * 100,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'ade': ade,
        'fde': fde,
        'learning_rate': learning_rate,
        'elapsed_time_minutes': elapsed_time / 60,
        'estimated_remaining_minutes': estimated_remaining / 60,
        'status': 'training'
    }

    # 保存到progress.json文件
    progress_file = os.path.join(folders['base'], 'progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress_info, f, indent=2)

    # 保存历史记录
    history_file = os.path.join(folders['logs'], 'progress_history.jsonl')
    with open(history_file, 'a') as f:
        f.write(json.dumps(progress_info) + '\n')

def select_adjacency_config_for_training():
    """让用户选择完整训练使用的邻接矩阵配置"""
    configs = {
        "1": ("default", "默认全连接模式 - 信息最丰富"),
        "2": ("ego_only", "ego为中心的星型连接 - 计算高效"),
        "3": ("no_ego_road", "禁止ego连接道路元素 - 测试道路信息重要性"),
        "4": ("nearby_only", "只连接15米内的元素 - 平衡性能和效率")
    }

    print("\n🔧 选择完整训练使用的邻接矩阵配置:")
    print("=" * 50)
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

def set_adjacency_config_for_training(config_name):
    """设置完整训练的邻接矩阵配置"""
    # 直接使用已导入的函数
    get_adjacency_config(config_name)
    print(f"🔧 已设置完整训练邻接矩阵配置: {config_name}")

def create_training_folders(config_name, dataset_size, batch_size, learning_rate):
    """创建训练结果文件夹，包含训练基本信息"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    training_dir = f"training_{timestamp}_{config_name}_ds{dataset_size}_bs{batch_size}_lr{learning_rate:.0e}"

    folders = {
        'base': training_dir,
        'models': os.path.join(training_dir, 'models'),
        'results': os.path.join(training_dir, 'results'),
        'logs': os.path.join(training_dir, 'logs'),
        'plots': os.path.join(training_dir, 'plots')
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    print(f"📁 创建训练文件夹: {training_dir}")
    return folders

def restore_trajectory_from_increments(origin, increments):
    """
    将增量轨迹还原为绝对轨迹

    Args:
        origin: 起始绝对位置 [batch_size, 2]
        increments: 增量序列 [batch_size, time_steps, 2]

    Returns:
        absolute_trajectory: 绝对轨迹 [batch_size, time_steps, 2]
    """
    batch_size, time_steps, _ = increments.shape
    absolute_trajectory = torch.zeros_like(increments)

    # 第一个点：起始位置 + 第一个增量
    absolute_trajectory[:, 0, :] = origin + increments[:, 0, :]

    # 后续点：累加增量
    for i in range(1, time_steps):
        absolute_trajectory[:, i, :] = absolute_trajectory[:, i - 1, :] + increments[:, i, :]

    return absolute_trajectory


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    ade_list = []
    fde_list = []
    num_samples = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)

            # ✅ 修复：正确处理批次维度和目标张量形状
            batch_size = data.y.shape[0]
            y = data.y.view(batch_size, -1).to(device)  # [batch_size, 60]

            # 确保输出形状匹配
            if output.shape[0] != batch_size:
                # 如果模型输出的批次维度不匹配，需要调整
                output = output.expand(batch_size, -1)

            loss = criterion(output, y)
            val_loss += loss.item() * batch_size

            # ✅ 修正的ADE/FDE计算：先还原为绝对轨迹
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
    return val_loss / num_samples, avg_ade, avg_fde


class EarlyStopping:
    """Early stopping utility class"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 🔍 DEBUG: 打印配置文件路径
    print(f"[DEBUG] Using config.py from: {config.__file__}")
    print(f"[DEBUG] TRAIN_PATH = {config.TRAIN_PATH}")
    dataset_pt_path = os.path.join(config.TRAIN_PATH, "processed", "dataset.pt")
    print(f"[DEBUG] dataset.pt exists? {os.path.exists(dataset_pt_path)}")
    print(f"📁 Using TRAIN_PATH: {config.TRAIN_PATH}")

    full_dataset = GraphDataset(config.TRAIN_PATH)
    print(f"📊 Loaded dataset with {len(full_dataset)} samples")

    # 🔧 临时数据量限制 - 用于快速测试
    USE_SMALL_DATASET = True  # 设置为False使用完整数据集
    SMALL_DATASET_SIZE = 1000  # 使用1000个样本进行快速测试

    if USE_SMALL_DATASET and len(full_dataset) > SMALL_DATASET_SIZE:
        print(f"🧪 Using small dataset for quick testing: {SMALL_DATASET_SIZE} samples")
        small_indices = list(range(min(SMALL_DATASET_SIZE, len(full_dataset))))
        full_dataset = Subset(full_dataset, small_indices)
        print(f"📊 Reduced dataset size: {len(full_dataset)} samples")

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=config.VALIDATION_SPLIT, random_state=config.SEED)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = TransformerPredictor().to(device)
    criterion = nn.MSELoss()

    # 🔧 降低学习率并改进学习率调度
    initial_lr = 0.0001  # 从0.001降至0.0001
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)

    # 使用ReduceLROnPlateau调度器，当验证损失停止改善时自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # 🛑 添加早停机制
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    os.makedirs(config.WEIGHT_PATH, exist_ok=True)
    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    lr_list = []  # 记录学习率变化
    ade_list, fde_list = [], []  # 新增：记录ADE和FDE历史

    # 选择邻接矩阵配置
    adjacency_config = select_adjacency_config_for_training()
    set_adjacency_config_for_training(adjacency_config)

    # 创建训练文件夹
    folders = create_training_folders(adjacency_config, len(train_dataset), config.BATCH_SIZE, initial_lr)

    # 设置日志记录
    log_file_path = os.path.join(folders['logs'], 'training_log.txt')
    logger = setup_logging(log_file_path)

    logger.info(f"🚀 Starting training with improved settings:")
    logger.info(f"   - Initial learning rate: {initial_lr}")
    logger.info(f"   - Early stopping patience: 15 epochs")
    logger.info(f"   - Validation every epoch")
    logger.info(f"   - Device: {device}")
    logger.info(f"   - Adjacency config: {get_current_adjacency_config()}")
    logger.info(f"   - Total epochs: {config.EPOCHS}")
    logger.info(f"   - Train samples: {len(train_dataset)}")
    logger.info(f"   - Val samples: {len(val_dataset)}")
    logger.info(f"   - Batch size: {config.BATCH_SIZE}")
    logger.info("=" * 70)

    # 创建总体进度条
    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress", position=0)

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start_time = time.time()

        # 为每个epoch创建详细的batch进度条
        batch_pbar = tqdm(train_loader,
                         desc=f"Epoch {epoch + 1:3d}",
                         position=1,
                         leave=False,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for i, data in enumerate(batch_pbar):
            data = data.to(device)

            if i == 0 and epoch == 0:
                print(f"[DEBUG] data.y.shape: {data.y.shape}")
                print(f"[DEBUG] Expected output shape: [batch_size, {config.OUT_CHANNELS}]")

            output = model(data)
            batch_size = data.y.shape[0]
            y = data.y.view(batch_size, -1).to(device)

            if output.shape[0] != batch_size:
                output = output.expand(batch_size, -1)

            if i == 0 and epoch == 0:
                print(f"[DEBUG] model output shape: {output.shape}")
                print(f"[DEBUG] target y shape: {y.shape}")

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()

            # 添加梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

            # 更新batch进度条显示当前loss
            if i % 10 == 0:  # 每10个batch更新一次显示
                current_avg_loss = epoch_loss / num_samples
                batch_pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        batch_pbar.close()

        avg_loss = epoch_loss / num_samples
        train_loss_list.append(avg_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        # 🔄 每个epoch都进行验证
        print(f"\n📊 Running validation for epoch {epoch + 1}...")
        val_loss, avg_ade, avg_fde = validate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        ade_list.append(avg_ade)  # 保存ADE到历史记录
        fde_list.append(avg_fde)  # 保存FDE到历史记录

        # 更新学习率调度器
        scheduler.step(val_loss)

        # 计算epoch耗时
        epoch_time = time.time() - start_time

        # 详细的epoch结果显示
        print(f"✅ Epoch {epoch + 1:3d}/{config.EPOCHS} Results:")
        print(f"   📈 Train Loss: {avg_loss:.6f}")
        print(f"   📉 Val Loss:   {val_loss:.6f}")
        print(f"   🎯 ADE:        {avg_ade:.4f} meters")
        print(f"   🎯 FDE:        {avg_fde:.4f} meters")
        print(f"   ⚡ Learning Rate: {current_lr:.2e}")
        print(f"   ⏱️  Time:      {epoch_time:.1f}s")

        # 更新总进度条
        epoch_pbar.set_postfix({
            'Best_Val': f'{best_val_loss:.6f}',
            'Current_Val': f'{val_loss:.6f}',
            'ADE': f'{avg_ade:.4f}m'
        })

        # 保存最佳模型到统一文件夹
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 获取当前邻接矩阵配置名称
            adj_config = get_current_adjacency_config()
            best_model_name = f"best_model_{adj_config}.pth"
            torch.save(model.state_dict(), os.path.join(folders['models'], best_model_name))
            print(f"   💾 🏆 New best model saved! (val_loss: {val_loss:.6f})")

        # 🛑 检查早停条件
        if early_stopping(val_loss, model):
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
            epoch_pbar.close()
            break

        # 保存训练进度信息
        estimated_remaining_time = (config.EPOCHS - epoch - 1) * (time.time() - start_time)
        save_progress_info(folders, epoch, config.EPOCHS, avg_loss, val_loss, avg_ade, avg_fde,
                          current_lr, epoch_time, estimated_remaining_time)

        logger.info(f"✅ Epoch {epoch + 1:3d}/{config.EPOCHS} Results:")
        logger.info(f"   📈 Train Loss: {avg_loss:.6f}")
        logger.info(f"   📉 Val Loss:   {val_loss:.6f}")
        logger.info(f"   🎯 ADE:        {avg_ade:.4f} meters")
        logger.info(f"   🎯 FDE:        {avg_fde:.4f} meters")
        logger.info(f"   ⚡ Learning Rate: {current_lr:.2e}")
        logger.info(f"   ⏱️  Time:      {epoch_time:.1f}s")
        logger.info("-" * 70)

    epoch_pbar.close()

    # 保存最终模型到统一文件夹
    adj_config = get_current_adjacency_config()
    final_model_name = f"final_model_{adj_config}.pth"
    torch.save(model.state_dict(), os.path.join(folders['models'], final_model_name))
    print(f"🏁 Training complete. Final model saved as: {final_model_name}")

    # 保存详细的训练记录到统一文件夹
    loss_record_name = f"loss_record_{adj_config}_ep{len(train_loss_list)}_bs{config.BATCH_SIZE}_lr{initial_lr}.json"
    loss_record_path = os.path.join(folders['results'], loss_record_name)
    with open(loss_record_path, 'w') as f:
        json.dump({
            "adjacency_config": adj_config,
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "ade_history": ade_list,  # 新增：ADE历史数据
            "fde_history": fde_list,  # 新增：FDE历史数据
            "learning_rates": lr_list,
            "best_val_loss": float(best_val_loss),
            "final_ade": ade_list[-1] if ade_list else None,  # 新增：最终ADE
            "final_fde": fde_list[-1] if fde_list else None,  # 新增：最终FDE
            "best_ade": min(ade_list) if ade_list else None,  # 新增：最佳ADE
            "best_fde": min(fde_list) if fde_list else None,  # 新增：最佳FDE
            "epochs_trained": len(train_loss_list),
            "early_stopped": len(train_loss_list) < config.EPOCHS,
            "dataset_size": len(train_dataset),
            "validation_size": len(val_dataset),
            "batch_size": config.BATCH_SIZE,
            "initial_learning_rate": initial_lr,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"📁 Detailed training record saved to {loss_record_path}")

    # 绘制改进的loss曲线图并保存到统一文件夹
    plt.figure(figsize=(15, 5))

    # 子图1: 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss', alpha=0.7)
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training/Validation Loss - {adj_config}')  # 添加配置信息到标题
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2: 学习率变化
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(lr_list) + 1), lr_list, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图3: 损失差异
    plt.subplot(1, 3, 3)
    loss_diff = np.array(val_loss_list) - np.array(train_loss_list[:len(val_loss_list)])
    plt.plot(range(1, len(loss_diff) + 1), loss_diff, label='Val - Train Loss', color='red')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Monitor')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_name = f"training_analysis_{adj_config}_ep{len(train_loss_list)}_lr{initial_lr}.png"
    loss_plot_path = os.path.join(folders['plots'], plot_name)
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comprehensive training analysis saved to {loss_plot_path}")

    # 保存训练配置信息到统一文件夹
    config_info = {
        "training_info": {
            "adjacency_config": adj_config,
            "dataset_size": len(train_dataset),
            "validation_size": len(val_dataset),
            "batch_size": config.BATCH_SIZE,
            "initial_learning_rate": initial_lr,
            "epochs_planned": config.EPOCHS,
            "epochs_trained": len(train_loss_list),
            "early_stopped": len(train_loss_list) < config.EPOCHS,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_info": {
            "model_type": "TransformerPredictor",
            "loss_function": "MSELoss",
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "early_stopping_patience": 15
        },
        "results": {
            "best_val_loss": float(best_val_loss),
            "final_train_loss": train_loss_list[-1] if train_loss_list else None,
            "final_val_loss": val_loss_list[-1] if val_loss_list else None
        }
    }

    config_path = os.path.join(folders['logs'], f"training_config_{adj_config}.json")
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)

    print(f"📋 Training configuration saved to {config_path}")
    print(f"\n🎉 所有训练文件已保存到文件夹: {folders['base']}")
    print(f"   📂 模型文件: {folders['models']}")
    print(f"   📊 结果文件: {folders['results']}")
    print(f"   📈 图表文件: {folders['plots']}")
    print(f"   📋 日志文件: {folders['logs']}")
