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

# ✅ 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset

from setproctitle import setproctitle

setproctitle("Yi Xiao")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
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
            num_samples += batch_size

    model.train()
    return val_loss / num_samples


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
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.DECAY_LR_EVERY, gamma=config.DECAY_LR_FACTOR)

    os.makedirs(config.WEIGHT_PATH, exist_ok=True)
    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []

    tqdm.write("Start training...")
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start_time = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}", leave=False)
        for i, data in enumerate(progress):
            data = data.to(device)

            # ✅ 调试：打印张量形状
            if i == 0 and epoch == 0:
                print(f"[DEBUG] data.y.shape: {data.y.shape}")
                print(f"[DEBUG] Expected output shape: [batch_size, {config.OUT_CHANNELS}]")

            output = model(data)

            # ✅ 修复：正确处理批次维度和目标张量形状
            batch_size = data.y.shape[0]
            y = data.y.view(batch_size, -1).to(device)  # [batch_size, 60]

            # ✅ 确保输出形状匹配批次大小
            if output.shape[0] != batch_size:
                # 如果模型输出批次维度为1，需要扩展到匹配实际批次大小
                output = output.expand(batch_size, -1)

            # ✅ 调试：打印第一个epoch的第一个batch的形状
            if i == 0 and epoch == 0:
                print(f"[DEBUG] model output shape: {output.shape}")
                print(f"[DEBUG] target y shape: {y.shape}")

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

            if (i + 1) % config.SHOW_EVERY == 0:
                tqdm.write(f"Epoch {epoch + 1} | Iter {i + 1} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_samples
        train_loss_list.append(avg_loss)
        scheduler.step()

        # ✅ 每几个epoch验证一次，避免过于频繁
        if (epoch + 1) % config.VAL_EVERY == 0:
            val_loss = validate(model, val_loader, criterion, device)
            val_loss_list.append(val_loss)

            tqdm.write(
                f"✅ Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time() - start_time:.2f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(config.WEIGHT_PATH, "best_model.pth"))
                tqdm.write(f"Saved best model at epoch {epoch + 1}")
        else:
            tqdm.write(
                f"✅ Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s"
            )

    torch.save(model.state_dict(), os.path.join(config.WEIGHT_PATH, "final_model.pth"))
    tqdm.write("🏁 Training complete. Final model saved.")

    # ✅ 保存 loss 列表为 JSON 文件，便于后续分析/重画图
    loss_record_path = os.path.join(config.WEIGHT_PATH,
                                    f"loss_record_ep{config.EPOCHS}_bs{config.BATCH_SIZE}_lr{config.LEARNING_RATE}.json")
    with open(loss_record_path, 'w') as f:
        json.dump({
            "train_loss": train_loss_list,
            "val_loss": val_loss_list
        }, f)

    tqdm.write(f"📁 Loss list saved to {loss_record_path}")

    # 可视化 loss 曲线
    if val_loss_list:  # 只有当有验证损失时才绘制
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Train Loss')
        val_epochs = [i * config.VAL_EVERY for i in range(1, len(val_loss_list) + 1)]
        plt.plot(val_epochs, val_loss_list, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(config.WEIGHT_PATH,
                                      f"loss_plot_ep{config.EPOCHS}_bs{config.BATCH_SIZE}_lr{config.LEARNING_RATE}.svg")
        plt.savefig(loss_plot_path)
        plt.close()
        tqdm.write(f"📊 Loss plot saved to {loss_plot_path}")