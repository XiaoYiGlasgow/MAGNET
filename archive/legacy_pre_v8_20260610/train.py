import os
import sys
import time
import json
from typing import Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
repo_root = str(Path(current_dir).parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
repo_models_dir = os.path.join(repo_root, "models")
if repo_models_dir not in sys.path:
    sys.path.insert(0, repo_models_dir)

# 强制清除可能的缓存模块
modules_to_clear = ['utils', 'utils.dataset', 'dataset', 'config']
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]

import config

try:
    from transformer_predictor import TransformerPredictor
except ImportError as e:
    print(f"模型导入错误: {e}")
    print(f"检查路径: {repo_root}/models")
    sys.exit(1)

try:
    from src.shared.utils.dataset import GraphDataset, get_adjacency_config
    print(f"[SUCCESS] 成功导入GraphDataset和get_adjacency_config")
except ImportError as e:
    print(f"数据集导入错误: {e}")
    sys.exit(1)

def restore_trajectory_from_increments(reference_origin: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
    """Restore absolute trajectory from incremental movements."""
    batch_size = reference_origin.shape[0]
    num_timesteps = increments.shape[1]

    # Initialize trajectory with reference origin
    trajectory = torch.zeros(batch_size, num_timesteps, 2, device=reference_origin.device)

    # First timestep is the reference origin
    current_pos = reference_origin

    for t in range(num_timesteps):
        current_pos = current_pos + increments[:, t, :]
        trajectory[:, t, :] = current_pos

    return trajectory

def convert_to_vehicle_frame(global_points: torch.Tensor, ego_position: torch.Tensor, ego_heading: float) -> torch.Tensor:
    """
    将全局坐标转换为车辆局部坐标系
    车辆局部坐标系：x轴指向前方，y轴指向左方
    """
    # 计算相对位置
    relative_points = global_points - ego_position
    
    # 旋转到车辆坐标系
    cos_h = torch.cos(-ego_heading)
    sin_h = torch.sin(-ego_heading)
    
    local_x = relative_points[..., 0] * cos_h - relative_points[..., 1] * sin_h
    local_y = relative_points[..., 0] * sin_h + relative_points[..., 1] * cos_h
    
    return torch.stack([local_x, local_y], dim=-1)

def convert_to_global_frame(local_points: torch.Tensor, ego_position: torch.Tensor, ego_heading: float) -> torch.Tensor:
    """
    将车辆局部坐标系转换为全局坐标
    """
    cos_h = torch.cos(ego_heading)
    sin_h = torch.sin(ego_heading)
    
    global_x = local_points[..., 0] * cos_h - local_points[..., 1] * sin_h
    global_y = local_points[..., 0] * sin_h + local_points[..., 1] * cos_h
    
    global_points = torch.stack([global_x, global_y], dim=-1) + ego_position
    
    return global_points

def create_training_folders(adjacency_config: str, dataset_size: int, batch_size: int, learning_rate: float) -> dict:
    """Create training folders for organizing results."""
    import os
    scenario_type = "unknown"
    if hasattr(config, 'TRAIN_PATH'):
        path_parts = config.TRAIN_PATH.split(os.sep)
        for part in path_parts:
            # 精确匹配场景类型文件夹，避免误匹配包含"processed_"的其他文件夹
            if part.startswith('processed_') and part != 'processed_data_by_type':
                # 从 "processed_high_magnitude_speed" 提取 "high_magnitude_speed"
                scenario_name = part.replace('processed_', '', 1)  # 只替换第一个匹配
                scenario_type = f"type_{scenario_name}"
                break
            elif part.startswith('type_'):
                scenario_type = part
                break
    
    print(f"Extracted scenario type: {scenario_type} from TRAIN_PATH: {config.TRAIN_PATH}")

    # Create main results folder
    main_folder = f"results_{scenario_type}_{adjacency_config}_adjacency"
    main_path = os.path.join(config.ALL_RESULTS_DIR, main_folder)

    # Create training session subfolder with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_folder = f"training_{timestamp}_ds{dataset_size}_bs{batch_size}_lr{learning_rate:.0e}"
    session_path = os.path.join(main_path, session_folder)

    folders = {
        'base': session_path,
        'models': os.path.join(session_path, 'models'),
        'results': os.path.join(session_path, 'results'),
        'logs': os.path.join(session_path, 'logs')
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


def setup_logging(log_file_path: str):
    """Setup logging configuration."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_progress_info(folders: dict, epoch: int, total_epochs: int, train_loss: float,
                      val_loss: float, ade: float, fde: float, lr: float,
                      epoch_time: float, estimated_remaining: float):
    """Save training progress information."""
    progress_info = {
        'epoch': epoch + 1,
        'total_epochs': total_epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'ade': ade,
        'fde': fde,
        'learning_rate': lr,
        'epoch_time': epoch_time,
        'estimated_remaining_time': estimated_remaining,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }

    progress_file = os.path.join(folders['base'], 'training_progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress_info, f, indent=2)


# Global adjacency config state
_current_adjacency_config = "default"


def select_adjacency_config_for_training() -> str:
    """Interactive adjacency configuration selection for training."""
    configs = {
        "1": ("default", "Default: Full connectivity between all entities"),
        "2": ("ego_only", "Ego Only: Only ego vehicle connects to other entities"),
        "3": ("no_ego_road", "No Ego-Road: Ego doesn't connect to road elements"),
        "4": ("nearby_only", "Nearby Only: Only connect nearby entities"),
        "5": ("selective", "Selective: Custom connection patterns")
    }

    print("\n" + "="*60)
    print("SELECT ADJACENCY MATRIX CONFIGURATION")
    print("="*60)

    for key, (config_name, description) in configs.items():
        print(f"{key}. {config_name.upper()}: {description}")

    print("="*60)

    while True:
        try:
            choice = input("Select configuration (1-5) [default=1]: ")


            if not choice:
                choice = "1"

            if choice in configs:
                selected_config = configs[choice][0]
                print(f"Selected: {selected_config.upper()}")
                print("="*60)
                return selected_config
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\nUsing default configuration.")
            return "default"


def set_adjacency_config_for_training(config_name: str):
    """Set the current adjacency configuration."""
    global _current_adjacency_config
    _current_adjacency_config = config_name


def get_current_adjacency_config() -> str:
    """Get the current adjacency configuration."""
    return _current_adjacency_config


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """Validate the model and return validation loss, ADE, and FDE metrics."""
    model.eval()
    val_loss = 0.0
    num_samples = 0
    ade_list = []
    fde_list = []
    increment_errors = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # 添加详细的数据检查
            if i == 0:
                print(f"  [DATA CHECK] Before processing:")
                print(f"    Data type: {type(data)}")
                print(f"    Available keys: {sorted(list(data.keys()))}")
                print(f"    Has ego_history (hasattr): {hasattr(data, 'ego_history')}")
                print(f"    'ego_history' in keys: {'ego_history' in data.keys()}")

                # 尝试多种访问方式
                try:
                    ego_hist_getattr = getattr(data, 'ego_history', None)
                    print(f"    getattr result: {ego_hist_getattr is not None}")
                    if ego_hist_getattr is not None:
                        print(f"    getattr shape: {ego_hist_getattr.shape}")
                except Exception as e:
                    print(f"    getattr error: {e}")

                try:
                    ego_hist_direct = data.ego_history
                    print(f"    Direct access: {ego_hist_direct is not None}")
                    if ego_hist_direct is not None:
                        print(f"    Direct shape: {ego_hist_direct.shape}")
                except Exception as e:
                    print(f"    Direct access error: {e}")

            # 保存所有可能的自定义属性
            custom_attrs = {}
            attr_names = ['ego_history', 'ego_last_pos', 'gnn_x', 'gnn_edge_index',
                         'gnn_edge_attr', 'traj_id_to_indices', 'cluster', 'valid_len', 'time_step_len']

            for attr_name in attr_names:
                if hasattr(data, attr_name):
                    custom_attrs[attr_name] = getattr(data, attr_name)

            # 设备转移
            data = data.to(device)
            
            # 恢复自定义属性
            for attr_name, attr_value in custom_attrs.items():
                if not hasattr(data, attr_name):
                    if isinstance(attr_value, torch.Tensor):
                        setattr(data, attr_name, attr_value.to(device))
                    else:
                        setattr(data, attr_name, attr_value)

            # 添加调试信息 - 在设备转移后立即检查
            if i == 0:
                print(f"  [DEVICE TRANSFER DEBUG] After .to(device):")
                print(f"    hasattr(data, 'ego_history'): {hasattr(data, 'ego_history')}")
                if hasattr(data, 'ego_history'):
                    print(f"    ego_history shape: {data.ego_history.shape}")
                    print(f"    ego_history device: {data.ego_history.device}")
                else:
                    print(f"    Saved custom_attrs keys: {list(custom_attrs.keys())}")
                    print(f"    ego_history was saved: {'ego_history' in custom_attrs}")

            output = model(data)

            # Determine batch size based on data.y dimensions
            if data.y.dim() == 2:
                batch_size = data.y.shape[0] // 60
            elif data.y.dim() == 3:
                batch_size = data.y.shape[0]
            else:
                batch_size = 1

            pred_increments = output.view(batch_size, 60, 2)
            true_increments = data.y.view(batch_size, 60, 2)

            # Handle reference origin for trajectory restoration
            # 修复坐标系不匹配问题 - 改进ego_history检查逻辑
            reference_origin = torch.zeros(batch_size, 2, device=device)  # 默认值
            ego_history_found = False

            if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
                try:
                    ego_hist = data.ego_history

                    # 处理不同的批量格式
                    if ego_hist.dim() == 1:
                        # 展平的批量: [batch_size*20*2]
                        expected_size = batch_size * 20 * 2
                        if ego_hist.shape[0] == expected_size:
                            ego_hist_reshaped = ego_hist.view(batch_size, 20, 2)
                            reference_origin = ego_hist_reshaped[:, -1, :]
                            ego_history_found = True
                        elif ego_hist.shape[0] == 40:  # 单样本的20*2展平
                            ego_hist_reshaped = ego_hist.view(20, 2).unsqueeze(0).repeat(batch_size, 1, 1)
                            reference_origin = ego_hist_reshaped[:, -1, :]
                            ego_history_found = True

                    elif ego_hist.dim() == 2:
                        if ego_hist.shape[0] == batch_size * 20 and ego_hist.shape[1] == 2:
                            # 批量展平: [batch_size*20, 2]
                            ego_hist_reshaped = ego_hist.view(batch_size, 20, 2)
                            reference_origin = ego_hist_reshaped[:, -1, :]
                            ego_history_found = True
                        elif ego_hist.shape[0] == 20 and ego_hist.shape[1] == 2:
                            # 单样本: [20, 2]
                            ego_hist_reshaped = ego_hist.unsqueeze(0).repeat(batch_size, 1, 1)
                            reference_origin = ego_hist_reshaped[-1, :].unsqueeze(0).expand(batch_size, -1)
                            ego_history_found = True

                    elif ego_hist.dim() == 3 and ego_hist.shape[0] == batch_size:
                        # 正确的批量格式: [batch_size, 20, 2]
                        reference_origin = ego_hist[:, -1, :]
                        ego_history_found = True

                except Exception as e:
                    ego_history_found = False

            # 备用：使��ego_last_pos
            if not ego_history_found and hasattr(data, 'ego_last_pos') and data.ego_last_pos.numel() > 0:
                # 备用：使用ego_last_pos，但需要检查坐标系
                if data.ego_last_pos.dim() == 1:
                    if data.ego_last_pos.shape[0] == batch_size * 2:
                        reference_origin = data.ego_last_pos.view(batch_size, 2)
                    elif data.ego_last_pos.shape[0] == 2:
                        reference_origin = data.ego_last_pos.unsqueeze(0).expand(batch_size, -1)
                    else:
                        reference_origin = torch.zeros(batch_size, 2, device=device)
                elif data.ego_last_pos.dim() == 2 and data.ego_last_pos.shape[0] == batch_size:
                    reference_origin = data.ego_last_pos
                else:
                    reference_origin = torch.zeros(batch_size, 2, device=device)
            else:
                reference_origin = torch.zeros(batch_size, 2, device=device)

            pred_trajectory = restore_trajectory_from_increments(reference_origin, pred_increments)
            true_trajectory = restore_trajectory_from_increments(reference_origin, true_increments)

            increment_error = torch.norm(pred_increments - true_increments, dim=2)
            avg_increment_error = torch.mean(increment_error, dim=1)
            increment_errors.extend(avg_increment_error.cpu().numpy())

            if i == 0:
                print(f"[HISTORY-BASED] Using ego history trajectory for prediction")
                print(f"  Reference origin sample: [{reference_origin[0, 0]:.2f}, {reference_origin[0, 1]:.2f}]")
                print(f"  True final position: [{true_trajectory[0, -1, 0]:.2f}, {true_trajectory[0, -1, 1]:.2f}]")
                print(f"  Pred final position: [{pred_trajectory[0, -1, 0]:.2f}, {pred_trajectory[0, -1, 1]:.2f}]")
                print(f"  Average increment error: {avg_increment_error[0]:.2f}m")

                # Debug: Show ADE and FDE for first 5 samples in validation batch
                position_errors = torch.norm(pred_trajectory - true_trajectory, dim=2)
                sample_ade = torch.mean(position_errors, dim=1)
                sample_fde = position_errors[:, -1]

                num_debug_samples = min(5, batch_size)
                print(f"  First {num_debug_samples} samples - ADE/FDE:")
                for j in range(num_debug_samples):
                    print(f"    Sample {j+1}: ADE={sample_ade[j]:.2f}m, FDE={sample_fde[j]:.2f}m")

                # Debug: Analyze high-error samples
                high_error_threshold = 30.0  # FDE > 30m认为是高误差样本
                high_error_mask = sample_fde > high_error_threshold
                num_high_error = high_error_mask.sum().item()

                if num_high_error > 0:
                    print(f"  High-error samples ({num_high_error} samples with FDE > {high_error_threshold}m):")
                    for j in range(batch_size):
                        if high_error_mask[j]:
                            # 分析异常样本的特征
                            true_traj_sample = true_trajectory[j]  # [60, 2]
                            pred_traj_sample = pred_trajectory[j]  # [60, 2]

                            # 计算轨迹长度
                            true_diffs = true_traj_sample[1:] - true_traj_sample[:-1]
                            true_speeds = torch.norm(true_diffs, dim=1)
                            true_total_distance = torch.sum(true_speeds)

                            # 计算方向变化
                            true_directions = true_diffs / (torch.norm(true_diffs, dim=1, keepdim=True) + 1e-8)
                            direction_changes = torch.acos(torch.clamp(
                                torch.sum(true_directions[:-1] * true_directions[1:], dim=1), -1, 1
                            ))
                            avg_direction_change = torch.mean(direction_changes) * 180 / 3.14159  # 转换为度

                            print(f"    Sample {j+1}: FDE={sample_fde[j]:.2f}m, TotalDist={true_total_distance:.1f}m, "
                                  f"AvgDirChange={avg_direction_change:.1f}°, MaxSpeed={true_speeds.max():.2f}m/s")

                print(f"  Batch statistics: ADE range=[{sample_ade.min():.2f}, {sample_ade.max():.2f}]m")
                print(f"  Batch statistics: FDE range=[{sample_fde.min():.2f}, {sample_fde.max():.2f}]m")

                # Debug: Show actual increment values with detailed statistics
                true_inc_norms = torch.norm(true_increments[0], dim=1)
                pred_inc_norms = torch.norm(pred_increments[0], dim=1)
                print(f"  True increment norm: {true_inc_norms.mean():.2f}m (std={true_inc_norms.std():.3f})")
                print(f"  Pred increment norm: {pred_inc_norms.mean():.2f}m (std={pred_inc_norms.std():.3f})")
                print(f"  True increment range: [{true_inc_norms.min():.3f}, {true_inc_norms.max():.3f}]")
                print(f"  First 5 true increments: {true_increments[0][:5]}")

                # Debug ego history usage
                print(f"  [DEBUG] hasattr(data, 'ego_history'): {hasattr(data, 'ego_history')}")
                if hasattr(data, 'ego_history'):
                    print(f"  [DEBUG] data.ego_history type: {type(data.ego_history)}")
                    print(f"  [DEBUG] data.ego_history.numel(): {data.ego_history.numel()}")
                    print(f"  [DEBUG] data.ego_history.shape: {data.ego_history.shape}")

                if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
                    ego_hist = data.ego_history
                    print(f"  [DEBUG] Processing ego_history with shape: {ego_hist.shape}, batch_size: {batch_size}")

                    # Handle different batch formats - 修复批处理格式检测
                    if ego_hist.dim() == 2:
                        # 批处理格式: [batch_size*20, 2] 或 [batch_size, 20*2]
                        if ego_hist.shape[0] == batch_size * 20 and ego_hist.shape[1] == 2:
                            # 格式: [batch_size*20, 2] -> [batch_size, 20, 2]
                            ego_hist_reshaped = ego_hist.view(batch_size, 20, 2)
                        elif ego_hist.shape[0] == batch_size and ego_hist.shape[1] == 40:
                            # 格式: [batch_size, 40] -> [batch_size, 20, 2]
                            ego_hist_reshaped = ego_hist.view(batch_size, 20, 2)
                        else:
                            ego_hist_reshaped = ego_hist
                    elif ego_hist.dim() == 3:
                        # Already correct format [batch_size, 20, 2]
                        ego_hist_reshaped = ego_hist
                    elif ego_hist.dim() == 1:
                        # Flattened batch: [batch_size*20*2]
                        expected_size = batch_size * 20 * 2
                        if ego_hist.shape[0] == expected_size:
                            ego_hist_reshaped = ego_hist.view(batch_size, 20, 2)
                        else:
                            ego_hist_reshaped = ego_hist.view(-1, 2)
                    else:
                        ego_hist_reshaped = ego_hist.view(-1, 2)

                    print(f"  [DEBUG] ego_hist_reshaped.shape: {ego_hist_reshaped.shape}")

                    if ego_hist_reshaped.dim() == 3 and ego_hist_reshaped.shape[1] == 20:
                        # Calculate statistics for first sample
                        first_sample_hist = ego_hist_reshaped[0]  # [20, 2]
                        ego_hist_norms = torch.norm(first_sample_hist, dim=1)
                        non_zero_count = (ego_hist_norms > 0.001).sum().item()
                        print(f"  Ego history: shape={ego_hist.shape}, batch_shape={ego_hist_reshaped.shape}")
                        print(f"  Ego history non_zero_steps={non_zero_count}/20, norm_mean={ego_hist_norms.mean():.3f}m")
                    else:
                        ego_hist_norms = torch.norm(ego_hist_reshaped, dim=-1 if ego_hist_reshaped.dim() > 1 else 0)
                        non_zero_count = (ego_hist_norms > 0.001).sum().item()
                        print(f"  Ego history: shape={ego_hist.shape}, reshaped={ego_hist_reshaped.shape}, non_zero_steps={non_zero_count}")
                else:
                    print("  Ego history: NOT FOUND - model using fallback features")

            y_flat = data.y.view(batch_size, -1)
            if output.shape[0] != batch_size:
                output = output.expand(batch_size, -1)

            loss = criterion(output, y_flat)
            val_loss += loss.item() * batch_size

            position_errors = torch.norm(pred_trajectory - true_trajectory, dim=2)
            ade = torch.mean(position_errors, dim=1)
            fde = position_errors[:, -1]

            ade_list.extend(ade.cpu().numpy())
            fde_list.extend(fde.cpu().numpy())
            num_samples += batch_size

    model.train()
    avg_ade = sum(ade_list) / len(ade_list) if ade_list else 0
    avg_fde = sum(fde_list) / len(fde_list) if fde_list else 0
    avg_increment_error = sum(increment_errors) / len(increment_errors) if increment_errors else 0

    print(f"  Validation summary: Avg increment error={avg_increment_error:.2f}m")

    return val_loss / num_samples, avg_ade, avg_fde


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class ImprovedTrajectoryLoss(nn.Module):
    """Improved trajectory loss combining multiple error metrics."""

    def __init__(self, ade_weight: float = 1.0, fde_weight: float = 2.0,
                 smoothness_weight: float = 0.5, direction_weight: float = 0.3):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.smoothness_weight = smoothness_weight
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
        # Handle different input shapes
        if target.dim() == 2:
            if target.shape[1] == 2:
                batch_size = target.shape[0] // 60
                target = target.view(batch_size, 60, 2)
            elif target.shape[1] == 120:
                batch_size = target.shape[0]
                target = target.view(batch_size, 60, 2)
            else:
                raise ValueError(f"Unexpected target shape: {target.shape}")
        elif target.dim() == 3:
            batch_size = target.shape[0]
        else:
            raise ValueError(f"Unexpected target shape: {target.shape}")

        pred_traj = pred.view(batch_size, 60, 2)
        true_traj = target

        # ADE Loss (Average Displacement Error)
        displacement_errors = torch.norm(pred_traj - true_traj, dim=2)
        ade_loss = torch.mean(displacement_errors)

        # FDE Loss (Final Displacement Error) - emphasize final position
        fde_loss = torch.norm(pred_traj[:, -1, :] - true_traj[:, -1, :], dim=1).mean()

        # Smoothness Loss - penalize abrupt changes in trajectory
        pred_diffs = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
        true_diffs = true_traj[:, 1:, :] - true_traj[:, :-1, :]
        smoothness_loss = torch.mean(torch.norm(pred_diffs - true_diffs, dim=2))

        # Direction Loss - preserve overall trajectory direction
        pred_direction = pred_traj[:, -1, :] - pred_traj[:, 0, :]
        true_direction = true_traj[:, -1, :] - true_traj[:, 0, :]

        # Normalize directions to unit vectors
        pred_direction_norm = torch.norm(pred_direction, dim=1, keepdim=True)
        true_direction_norm = torch.norm(true_direction, dim=1, keepdim=True)

        # Avoid division by zero
        pred_direction_norm = torch.clamp(pred_direction_norm, min=1e-8)
        true_direction_norm = torch.clamp(true_direction_norm, min=1e-8)

        pred_direction_unit = pred_direction / pred_direction_norm
        true_direction_unit = true_direction / true_direction_norm

        # Cosine similarity loss (1 - cosine similarity)
        cosine_sim = torch.sum(pred_direction_unit * true_direction_unit, dim=1)
        direction_loss = torch.mean(1 - cosine_sim)

        # Combined loss
        total_loss = (self.ade_weight * ade_loss +
                     self.fde_weight * fde_loss +
                     self.smoothness_weight * smoothness_loss +
                     self.direction_weight * direction_loss)

        if return_components:
            return {
                'total_loss': total_loss,
                'ade_loss': ade_loss,
                'fde_loss': fde_loss,
                'smoothness_loss': smoothness_loss,
                'direction_loss': direction_loss,
                'weighted_ade': self.ade_weight * ade_loss,
                'weighted_fde': self.fde_weight * fde_loss,
                'weighted_smoothness': self.smoothness_weight * smoothness_loss,
                'weighted_direction': self.direction_weight * direction_loss
            }

        return total_loss


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(f"[DEBUG] Using config.py from: {config.__file__}")
    print(f"[DEBUG] TRAIN_PATH = {config.TRAIN_PATH}")
    dataset_pt_path = os.path.join(config.TRAIN_PATH, "processed", "dataset.pt")
    print(f"[DEBUG] dataset.pt exists? {os.path.exists(dataset_pt_path)}")
    print(f"Using TRAIN_PATH: {config.TRAIN_PATH}")

    # 强制验证数据集类型
    print(f"[DEBUG] Importing GraphDataset from: {GraphDataset.__module__}")

    full_dataset = GraphDataset(config.TRAIN_PATH)
    print(f"Loaded dataset with {len(full_dataset)} samples")

    # 验证数据集是否包含ego_history
    test_sample = full_dataset[0]
    print(f"[DEBUG] Test sample type: {type(test_sample)}")
    print(f"[DEBUG] Test sample keys: {list(test_sample.keys())}")
    print(f"[DEBUG] Test sample has ego_history: {hasattr(test_sample, 'ego_history')}")

    if not hasattr(test_sample, 'ego_history'):
        print("[ERROR] Dataset does not contain ego_history! Regenerating...")
        # 删除所有processed文件并重新生成
        import shutil
        processed_dir = os.path.join(config.TRAIN_PATH, "processed")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        full_dataset = GraphDataset(config.TRAIN_PATH)
        test_sample = full_dataset[0]
        print(f"[DEBUG] After regeneration - has ego_history: {hasattr(test_sample, 'ego_history')}")
    else:
        print(f"[SUCCESS] Dataset contains ego_history with shape: {test_sample.ego_history.shape}")

    USE_SMALL_DATASET = False  # 改为False使用完整数据集
    SMALL_DATASET_SIZE = 1000

    if USE_SMALL_DATASET and len(full_dataset) > SMALL_DATASET_SIZE:
        print(f"Using small dataset for quick testing: {SMALL_DATASET_SIZE} samples")
        small_indices = list(range(min(SMALL_DATASET_SIZE, len(full_dataset))))
        full_dataset = Subset(full_dataset, small_indices)
        print(f"Reduced dataset size: {len(full_dataset)} samples")

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=config.VALIDATION_SPLIT, random_state=config.SEED)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    model = TransformerPredictor().to(device)

    criterion = ImprovedTrajectoryLoss(
        ade_weight=0.5,           # ADE权重
        fde_weight=3.0,           # 提高FDE权重，更注重最终位置
        smoothness_weight=0.3,    # 平滑度权重
        direction_weight=0.5      # 方向一致性权重
    )

    # Increase initial learning rate for better convergence
    initial_lr = 5e-4  # Increased from 1e-5 to 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-3, betas=(0.9, 0.999))

    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-6
    )

    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    os.makedirs(config.WEIGHT_PATH, exist_ok=True)
    best_val_loss = float('inf')
    train_loss_list, val_loss_list = [], []
    lr_list = []
    ade_list, fde_list = [], []
    loss_components_history = []  # 用于记录每个epoch的损失组成部分

    adjacency_config = select_adjacency_config_for_training()
    set_adjacency_config_for_training(adjacency_config)

    folders = create_training_folders(adjacency_config, len(train_dataset), config.BATCH_SIZE, initial_lr)

    log_file_path = os.path.join(folders['logs'], 'training_log.txt')
    logger = setup_logging(log_file_path)

    logger.info(f"Starting training with history-based architecture:")
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

    epoch_pbar = tqdm(range(config.EPOCHS), desc="Training Progress")

    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start_time = time.time()

        for i, data in enumerate(train_loader):
            data = data.to(device)

            if i == 0 and epoch == 0:
                print(f"[DEBUG] data.y.shape: {data.y.shape}")
                print(f"[DEBUG] Expected output shape: [batch_size, {config.OUT_CHANNELS}]")

            output = model(data)

            if data.y.dim() == 2:
                batch_size = data.y.shape[0] // 60
            elif data.y.dim() == 3:
                batch_size = data.y.shape[0]
            else:
                batch_size = 1

            y = data.y.view(batch_size, -1).to(device)

            if output.shape[0] != batch_size:
                if output.shape[0] == 1:
                    output = output.repeat(batch_size, 1)
                else:
                    output = output[:batch_size]

            if i == 0 and epoch == 0:
                print(f"[DEBUG] model output shape: {output.shape}")
                print(f"[DEBUG] target y shape: {y.shape}")

            optimizer.zero_grad()
            # 记录损失组成部分 - 在训练时也记录
            with torch.no_grad():
                sample_output = output[:1]  # 取第一个样本
                sample_y = y[:1]
                components = criterion.forward(sample_output, sample_y, return_components=True)

                if i == 0:  # 第一个batch时打印损失组成
                    print(f"  Loss components - ADE: {components['ade_loss']:.4f}, FDE: {components['fde_loss']:.4f}")
                    print(f"  Weighted - ADE: {components['weighted_ade']:.4f}, FDE: {components['weighted_fde']:.4f}")

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        avg_loss = epoch_loss / num_samples
        train_loss_list.append(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        print(f"\nRunning validation for epoch {epoch + 1}...")
        val_loss, avg_ade, avg_fde = validate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        ade_list.append(avg_ade)
        fde_list.append(avg_fde)

        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        epoch_pbar.set_postfix({
            'Train_Loss': f'{avg_loss:.4f}',
            'Val_Loss': f'{val_loss:.4f}',
            'ADE': f'{avg_ade:.2f}m',
            'FDE': f'{avg_fde:.2f}m',
            'LR': f'{current_lr:.2e}',
            'Time': f'{epoch_time:.1f}s'
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            adj_config = get_current_adjacency_config()
            best_model_name = f"best_model_{adj_config}.pth"
            torch.save(model.state_dict(), os.path.join(folders['models'], best_model_name))

        if early_stopping(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            epoch_pbar.close()
            break

        estimated_remaining_time = (config.EPOCHS - epoch - 1) * (time.time() - start_time)
        save_progress_info(folders, epoch, config.EPOCHS, avg_loss, val_loss, avg_ade, avg_fde,
                          current_lr, epoch_time, estimated_remaining_time)

        # 记录损失组成部分
        with torch.no_grad():
            components = criterion.forward(output, y, return_components=True)
            loss_components_history.append(components)

        logger.info(f"Epoch {epoch + 1:3d}/{config.EPOCHS}: Train={avg_loss:.4f}, Val={val_loss:.4f}, "
                   f"ADE={avg_ade:.2f}m, FDE={avg_fde:.2f}m, Time={epoch_time:.1f}s")

    epoch_pbar.close()

    adj_config = get_current_adjacency_config()
    final_model_name = f"final_model_{adj_config}.pth"
    torch.save(model.state_dict(), os.path.join(folders['models'], final_model_name))
    print(f"Training complete. Final model saved as: {final_model_name}")

    # Convert loss components history to JSON-serializable format
    def tensor_to_float(obj):
        """Convert tensor values to float for JSON serialization"""
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_float(item) for item in obj]
        return obj

    serializable_loss_components = tensor_to_float(loss_components_history)

    # 保存详细的训练记录，包括损失组成部分
    loss_record_name = f"loss_record_{adj_config}_ep{len(train_loss_list)}_bs{config.BATCH_SIZE}_lr{initial_lr}.json"
    loss_record_path = os.path.join(folders['results'], loss_record_name)
    with open(loss_record_path, 'w') as f:
        json.dump({
            "adjacency_config": adj_config,
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "ade_history": ade_list,
            "fde_history": fde_list,
            "learning_rates": lr_list,
            "loss_components_history": serializable_loss_components,
            "loss_weights": {
                "ade_weight": criterion.ade_weight,
                "fde_weight": criterion.fde_weight,
                "smoothness_weight": criterion.smoothness_weight,
                "direction_weight": criterion.direction_weight
            },
            "best_val_loss": float(best_val_loss),
            "final_ade": ade_list[-1] if ade_list else None,
            "final_fde": fde_list[-1] if fde_list else None,
            "best_ade": min(ade_list) if ade_list else None,
            "best_fde": min(fde_list) if fde_list else None,
            "epochs_trained": len(train_loss_list),
            "early_stopped": len(train_loss_list) < config.EPOCHS,
            "dataset_size": len(train_dataset),
            "validation_size": len(val_dataset),
            "batch_size": config.BATCH_SIZE,
            "initial_learning_rate": initial_lr,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"Detailed training record saved to {loss_record_path}")

    # 额外保存一个专门的损失组成部分文件，便于分析
    loss_components_file = os.path.join(folders['results'], f"loss_components_{adj_config}.json")
    with open(loss_components_file, 'w') as f:
        json.dump({
            "loss_weights": {
                "ade_weight": criterion.ade_weight,
                "fde_weight": criterion.fde_weight,
                "smoothness_weight": criterion.smoothness_weight,
                "direction_weight": criterion.direction_weight
            },
            "loss_components_by_epoch": serializable_loss_components,  # 使用已转换的可序列化数据
            "training_summary": {
                "total_epochs": len(loss_components_history),
                "final_epoch_components": serializable_loss_components[-1] if serializable_loss_components else None
            }
        }, f, indent=2)

    print(f"Loss components analysis saved to {loss_components_file}")

    # 在日志中记录损失函数配置总结
    logger.info("=" * 70)
    logger.info("Loss Function Configuration Summary:")
    logger.info(f"  ADE Loss Weight: {criterion.ade_weight} (Average Displacement Error)")
    logger.info(f"  FDE Loss Weight: {criterion.fde_weight} (Final Displacement Error)")
    logger.info(f"  Smoothness Loss Weight: {criterion.smoothness_weight} (Trajectory Smoothness)")
    logger.info(f"  Direction Loss Weight: {criterion.direction_weight} (Direction Consistency)")
    if loss_components_history:
        final_components = loss_components_history[-1]
        logger.info(f"Final Epoch Loss Components:")
        logger.info(f"  - ADE: {final_components['ade_loss']:.4f} (weighted: {final_components['weighted_ade']:.4f})")
        logger.info(f"  - FDE: {final_components['fde_loss']:.4f} (weighted: {final_components['weighted_fde']:.4f})")
        logger.info(f"  - Smoothness: {final_components['smoothness_loss']:.4f} (weighted: {final_components['weighted_smoothness']:.4f})")
        logger.info(f"  - Direction: {final_components['direction_loss']:.4f} (weighted: {final_components['weighted_direction']:.4f})")
    logger.info("=" * 70)
