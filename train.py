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
from utils.dataset import GraphDataset

from setproctitle import setproctitle

setproctitle("Yi Xiao")


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

    print(f"🚀 Starting training with improved settings:")
    print(f"   - Initial learning rate: {initial_lr}")
    print(f"   - Early stopping patience: 15 epochs")
    print(f"   - Validation every epoch")
    print(f"   - Device: {device}")

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start_time = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", leave=False)
        for i, data in enumerate(progress):
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

        avg_loss = epoch_loss / num_samples
        train_loss_list.append(avg_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        # 🔄 每个epoch都进行验证
        val_loss, avg_ade, avg_fde = validate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)

        # 更新学习率调度器
        scheduler.step(val_loss)

        print(f"✅ Epoch {epoch + 1:3d} | Train: {avg_loss:.6f} | Val: {val_loss:.6f} | "
              f"ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f} | LR: {current_lr:.2e} | "
              f"Time: {time.time() - start_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.WEIGHT_PATH, "best_model.pth"))
            print(f"💾 Saved best model at epoch {epoch + 1} (val_loss: {val_loss:.6f})")

        # 🛑 检查早停条件
        if early_stopping(val_loss, model):
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(config.WEIGHT_PATH, "final_model.pth"))
    print("🏁 Training complete.")

    # 保存详细的训练记录
    loss_record_path = os.path.join(config.WEIGHT_PATH,
                                    f"loss_record_improved_ep{len(train_loss_list)}_bs{config.BATCH_SIZE}_lr{initial_lr}.json")
    with open(loss_record_path, 'w') as f:
        json.dump({
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "learning_rates": lr_list,
            "best_val_loss": float(best_val_loss),
            "epochs_trained": len(train_loss_list),
            "early_stopped": len(train_loss_list) < config.EPOCHS
        }, f, indent=2)

    print(f"📁 Detailed training record saved to {loss_record_path}")

    # 绘制改进的loss曲线图
    plt.figure(figsize=(15, 5))

    # 子图1: 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss', alpha=0.7)
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
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
    loss_plot_path = os.path.join(config.WEIGHT_PATH,
                                  f"improved_training_analysis_ep{len(train_loss_list)}_lr{initial_lr}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comprehensive training analysis saved to {loss_plot_path}")

