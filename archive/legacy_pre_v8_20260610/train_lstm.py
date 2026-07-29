import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict, List
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Clear cached modules
modules_to_clear = ['utils', 'utils.dataset', 'config', 'dataset']
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]

import config

# Import dataset
try:
    from utils.dataset import GraphDataset
    print("Successfully imported GraphDataset from utils.dataset")
except ImportError as e:
    print(f"Import error: {e}")
    dataset_path = os.path.join(current_dir, 'utils')
    if dataset_path not in sys.path:
        sys.path.insert(0, dataset_path)
    try:
        from dataset import GraphDataset
        print("Successfully imported GraphDataset from dataset")
    except ImportError as e2:
        print(f"Backup import also failed: {e2}")
        sys.exit(1)


class LSTMTrajectoryPredictor(nn.Module):
    """LSTM-based trajectory predictor for nuPlan data"""

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 2,
                 dropout: float = 0.1,
                 sequence_length: int = 20,
                 prediction_length: int = 60):
        super(LSTMTrajectoryPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder for future predictions
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
        Returns:
            predictions: [batch_size, prediction_length, output_dim]
        """
        batch_size = x.size(0)

        # Encode input sequence
        encoder_output, (hidden, cell) = self.encoder(x)

        # Initialize decoder input with zeros
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=x.device)
        decoder_hidden = (hidden, cell)

        predictions = []

        # Decode step by step
        for _ in range(self.prediction_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Project to output space
            pred_step = self.output_layer(decoder_output.squeeze(1))
            predictions.append(pred_step)

            # Use prediction as next input
            decoder_input = pred_step.unsqueeze(1)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)
        return predictions


def extract_ego_features(data) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ego history and target from data"""
    try:
        # Extract ego history
        if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
            ego_hist = data.ego_history

            # Handle different formats
            if ego_hist.dim() == 3:
                # [batch_size, timesteps, 2]
                ego_history = ego_hist[:, :20, :2] if ego_hist.size(1) >= 20 else ego_hist
            elif ego_hist.dim() == 2:
                # [timesteps, 2] or [batch*timesteps, 2]
                if ego_hist.size(0) >= 20:
                    ego_history = ego_hist[:20, :2].unsqueeze(0)
                else:
                    padded = torch.zeros(20, 2, device=ego_hist.device)
                    padded[:ego_hist.size(0)] = ego_hist[:, :2]
                    ego_history = padded.unsqueeze(0)
            else:
                # Fallback to zeros
                ego_history = torch.zeros(1, 20, 2, device=ego_hist.device)
        else:
            ego_history = torch.zeros(1, 20, 2)

        # Extract target trajectory
        if hasattr(data, 'y') and data.y.numel() > 0:
            target = data.y
            if target.dim() == 1:
                if target.size(0) == 120:
                    target = target.view(1, 60, 2)
                else:
                    target = torch.zeros(1, 60, 2)
            elif target.dim() == 2:
                if target.size(0) == 60 and target.size(1) == 2:
                    target = target.unsqueeze(0)
                else:
                    target = torch.zeros(1, 60, 2)
            else:
                target = torch.zeros(1, 60, 2)
        else:
            target = torch.zeros(1, 60, 2)

        return ego_history, target

    except Exception as e:
        print(f"Error extracting features: {e}")
        return torch.zeros(1, 20, 2), torch.zeros(1, 60, 2)


def calculate_trajectory_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                                ego_history: torch.Tensor) -> Tuple[float, float]:
    """Calculate ADE and FDE for trajectory predictions"""
    batch_size = predictions.size(0)
    
    # 将增量转换为绝对轨迹坐标
    pred_traj = torch.zeros_like(predictions)
    target_traj = torch.zeros_like(targets)
    
    for b in range(batch_size):
        # 从ego历史的最后一个位置开始
        current_pred_pos = ego_history[b, -1, :].clone()
        current_target_pos = ego_history[b, -1, :].clone()
        
        for t in range(predictions.size(1)):
            # 累加增量得到绝对位置
            current_pred_pos += predictions[b, t, :]
            current_target_pos += targets[b, t, :]
            
            pred_traj[b, t, :] = current_pred_pos
            target_traj[b, t, :] = current_target_pos
    
    # 分析轨迹数据
    pred_total_distance = 0
    target_total_distance = 0
    
    for b in range(batch_size):
        # 计算预测轨迹的总行驶距离
        pred_diffs = torch.diff(pred_traj[b], dim=0)
        pred_distances = torch.norm(pred_diffs, dim=1)
        pred_total_distance += torch.sum(pred_distances).item()
        
        # 计算真实轨迹的总行驶距离
        target_diffs = torch.diff(target_traj[b], dim=0)
        target_distances = torch.norm(target_diffs, dim=1)
        target_total_distance += torch.sum(target_distances).item()
    
    # 计算绝对轨迹的欧几里得距离
    distances = torch.norm(pred_traj - target_traj, dim=2)
    
    # ADE: 所有时间步距离的平均值
    ade = torch.mean(distances).item()
    
    # FDE: 最后一个时间步的距离
    fde = torch.mean(distances[:, -1]).item()
    
    # 打印分析信息（仅第一次）
    if torch.rand(1).item() < 0.01:  # 1%概率打印，避免过多输出
        print(f"[ANALYSIS] Pred total distance: {pred_total_distance/batch_size:.2f}m")
        print(f"[ANALYSIS] Target total distance: {target_total_distance/batch_size:.2f}m")
        print(f"[ANALYSIS] Start pos: [{ego_history[0, -1, 0]:.2f}, {ego_history[0, -1, 1]:.2f}]")
        print(f"[ANALYSIS] Final pred pos: [{pred_traj[0, -1, 0]:.2f}, {pred_traj[0, -1, 1]:.2f}]")
        print(f"[ANALYSIS] Final target pos: [{target_traj[0, -1, 0]:.2f}, {target_traj[0, -1, 1]:.2f}]")
        print(f"[ANALYSIS] Max prediction increment: {torch.max(torch.abs(predictions)).item():.6f}")
        print(f"[ANALYSIS] Max target increment: {torch.max(torch.abs(targets)).item():.6f}")
    
    return ade, fde


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                  max_samples: int = 100) -> Tuple[float, float, float]:
    """Evaluate model and return loss, ADE, FDE"""
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_samples = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in data_loader:
            try:
                ego_history, target = extract_ego_features(data)

                if ego_history.size(1) < 10 or target.size(1) < 30:
                    continue

                ego_history = ego_history.to(device)
                target = target.to(device)

                ego_increments = torch.diff(ego_history, dim=1, prepend=ego_history[:, :1, :])
                # y is already position increments (local vehicle frame); do NOT diff again.
                target_increments = target

                predictions = model(ego_increments)
                loss = criterion(predictions, target_increments)

                # Calculate metrics
                ade, fde = calculate_trajectory_metrics(predictions, target_increments, ego_history)

                total_loss += loss.item()
                total_ade += ade
                total_fde += fde
                num_samples += 1

                if num_samples >= max_samples:
                    break

            except Exception:
                continue

    if num_samples == 0:
        return float('inf'), float('inf'), float('inf')

    return total_loss / num_samples, total_ade / num_samples, total_fde / num_samples


def train_lstm_model(dataset_path: str,
                     epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 1e-3,
                     validation_split: float = 0.2,
                     save_path: str = None) -> Dict:
    """Train LSTM trajectory predictor"""

    print(f"Loading dataset from: {dataset_path}")

    # Load dataset
    try:
        full_dataset = GraphDataset(dataset_path)
        print(f"Dataset loaded with {len(full_dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return {}

    # Split dataset
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split, random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # batch_size=1 for simplicity
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMTrajectoryPredictor().to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    train_losses = []
    val_losses = []
    train_ades = []
    train_fdes = []
    val_ades = []
    val_fdes = []
    best_val_loss = float('inf')

    print(f"Starting training on {device}")
    print("=" * 80)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            try:
                ego_history, target = extract_ego_features(data)

                if ego_history.size(1) < 10 or target.size(1) < 30:
                    continue

                ego_history = ego_history.to(device)
                target = target.to(device)

                ego_increments = torch.diff(ego_history, dim=1, prepend=ego_history[:, :1, :])
                # y is already position increments (local vehicle frame); do NOT diff again.
                target_increments = target

                optimizer.zero_grad()

                predictions = model(ego_increments)
                loss = criterion(predictions, target_increments)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_samples += 1

                # Limit training samples per epoch for efficiency
                if train_samples >= 100:
                    break

            except Exception:
                continue

        if train_samples == 0:
            print("No valid training samples found")
            break

        avg_train_loss = train_loss / train_samples
        train_losses.append(avg_train_loss)

        # Evaluate on training set (subset)
        train_loss_eval, train_ade, train_fde = evaluate_model(model, train_loader, device, max_samples=200)
        train_ades.append(train_ade)
        train_fdes.append(train_fde)

        # Evaluate on validation set
        val_loss_eval, val_ade, val_fde = evaluate_model(model, val_loader, device, max_samples=200)
        val_losses.append(val_loss_eval)
        val_ades.append(val_ade)
        val_fdes.append(val_fde)

        scheduler.step(val_loss_eval)

        # Print detailed progress
        print(f"Epoch {epoch+1:3d}/{epochs}: "
              f"Loss: {avg_train_loss:.4f}/{val_loss_eval:.4f} | "
              f"ADE: {train_ade:.2f}/{val_ade:.2f}m | "
              f"FDE: {train_fde:.2f}/{val_fde:.2f}m | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss_eval < best_val_loss and save_path:
            best_val_loss = val_loss_eval
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss_eval,
                'train_ade': train_ade,
                'train_fde': train_fde,
                'val_ade': val_ade,
                'val_fde': val_fde,
                'model_config': {
                    'input_dim': 2,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'output_dim': 2,
                    'sequence_length': 20,
                    'prediction_length': 60
                }
            }, save_path)
            print(f"★ New best model saved! Val ADE: {val_ade:.2f}m, FDE: {val_fde:.2f}m")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ades': train_ades,
        'train_fdes': train_fdes,
        'val_ades': val_ades,
        'val_fdes': val_fdes,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict()
    }


def main():
    """Main training function"""
    print("LSTM Trajectory Predictor Training")
    print("=" * 60)

    # Training configuration
    dataset_path = config.TRAIN_PATH
    model_save_path = os.path.join(config.LSTM_RESULTS_DIR, "trained_lstm_model.pth")

    # Training parameters
    epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    validation_split = 0.1

    print(f"Dataset path: {dataset_path}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model save path: {model_save_path}")

    # Train model
    results = train_lstm_model(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=validation_split,
        save_path=model_save_path
    )

    if not results:
        print("Training failed")
        return

    # Plot training curves
    if results['train_losses'] and results['val_losses']:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(results['train_losses']) + 1)

        plt.plot(epochs_range, results['train_losses'], label='Training Loss')
        plt.plot(epochs_range, results['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('LSTM Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = os.path.join(config.LSTM_RESULTS_DIR, "training_curve.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to {plot_path}")
        plt.show()

    print("=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    main()
