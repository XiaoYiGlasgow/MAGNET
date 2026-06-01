import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 强制清除可能的缓存模块
modules_to_clear = ['utils', 'utils.dataset', 'config', 'dataset']
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]

import config

# 尝试导入数据集
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
    """
    LSTM-based trajectory predictor for nuPlan data
    """

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

        # Encoder processes history increments into a recurrent state.
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder autoregressively predicts future increments.
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]

        Returns:
            predictions: Tensor of shape [batch_size, prediction_length, output_dim]
        """
        batch_size = x.size(0)

        _, (hidden, cell) = self.encoder(x)

        predictions = []
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=x.device)
        decoder_hidden = (hidden, cell)

        for _ in range(self.prediction_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            pred_step = self.output_layer(decoder_output.squeeze(1))
            predictions.append(pred_step)
            decoder_input = pred_step.unsqueeze(1)

        # 堆叠所有预测步骤
        predictions = torch.stack(predictions, dim=1)  # [batch_size, prediction_length, output_dim]

        return predictions


class SafeGraphDataset:
    """安全的数据集加载器，处理模块路径问题"""

    def __init__(self, root_path: str):
        self.root_path = root_path
        self.processed_dir = os.path.join(root_path, "processed")
        self.pt_path = os.path.join(self.processed_dir, "dataset.pt")

        # 尝试加载数据集
        self.data_list = self._load_dataset()

    def _load_dataset(self) -> List:
        """安全加载数据集"""
        if not os.path.exists(self.pt_path):
            print(f"Dataset file not found: {self.pt_path}")
            print("Attempting to regenerate dataset...")
            return self._regenerate_dataset()

        try:
            # 尝试正常加载
            data, slices = torch.load(self.pt_path, map_location='cpu', weights_only=False)
            print(f"Successfully loaded dataset from {self.pt_path}")
            return self._convert_to_list(data, slices)

        except (ModuleNotFoundError, AttributeError) as e:
            print(f"Failed to load dataset due to module path issue: {e}")
            print("Attempting to regenerate dataset...")
            return self._regenerate_dataset()

    def _regenerate_dataset(self) -> List:
        """重新生成数据集"""
        try:
            # 删除旧的processed文件
            if os.path.exists(self.processed_dir):
                import shutil
                print(f"Warning: removing processed dataset directory: {self.processed_dir}")
                shutil.rmtree(self.processed_dir)
                print("Removed old processed directory")

            # 重新创建数据集
            dataset = GraphDataset(self.root_path)
            return [dataset[i] for i in range(len(dataset))]

        except Exception as e:
            print(f"Failed to regenerate dataset: {e}")
            return []

    def _convert_to_list(self, data, slices) -> List:
        """将批处理数据转换为列表格式"""
        data_list = []
        keys = slices.keys()

        for i in range(len(slices[list(keys)[0]]) - 1):
            sample = {}
            for key in keys:
                start_idx = slices[key][i].item()
                end_idx = slices[key][i + 1].item()

                if hasattr(data, key):
                    attr = getattr(data, key)
                    if attr is not None:
                        sample[key] = attr[start_idx:end_idx]

            # 转换为PyG Data对象格式
            from torch_geometric.data import Data
            data_obj = Data(**sample)
            data_list.append(data_obj)

        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]


class LSTMTrajPredictor:
    """
    LSTM轨迹预测器，适配GNN_Trans数据格式
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)

        # 初始化模型
        self.model = LSTMTrajectoryPredictor(
            input_dim=2,
            hidden_dim=128,
            num_layers=2,
            output_dim=2,
            dropout=0.1,
            sequence_length=20,
            prediction_length=60
        ).to(self.device)

        # 如果提供了模型路径，加载预训练模型
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"Loaded pretrained model from {model_path}")

        self.model.eval()

    def extract_ego_history_from_data(self, data) -> np.ndarray:
        """
        从GNN_Trans数据格式提取ego历史轨迹

        Args:
            data: PyG数据对象

        Returns:
            ego_history: ego历史轨迹 [sequence_length, 2]
        """
        try:
            if hasattr(data, 'ego_history') and data.ego_history.numel() > 0:
                ego_hist = data.ego_history.cpu().numpy()

                # 处理不同的ego_history格式
                if ego_hist.ndim == 3 and ego_hist.shape[1] >= 20:
                    # [batch_size, timesteps, 2] 格式
                    return ego_hist[0, :20, :2]  # 取前20个时间步
                elif ego_hist.ndim == 2:
                    if ego_hist.shape[0] >= 20:
                        # [timesteps, 2] 格式
                        return ego_hist[:20, :2]
                    else:
                        # 数据不足，用零填充
                        padded = np.zeros((20, 2))
                        padded[:len(ego_hist)] = ego_hist[:, :2]
                        return padded
                elif ego_hist.ndim == 1:
                    # 展平格式 [timesteps*2]
                    if len(ego_hist) >= 40:
                        return ego_hist[:40].reshape(20, 2)
                    else:
                        padded = np.zeros(40)
                        padded[:len(ego_hist)] = ego_hist
                        return padded.reshape(20, 2)

            # 备用方案：使用ego_last_pos
            if hasattr(data, 'ego_last_pos') and data.ego_last_pos.numel() > 0:
                last_pos = data.ego_last_pos.cpu().numpy()[:2]
                # 创建简单的历史轨迹（假设静止）
                history = np.tile(last_pos, (20, 1))
                return history

            # 默认：零轨迹
            return np.zeros((20, 2))

        except Exception as e:
            print(f"Warning: Failed to extract ego history: {e}")
            return np.zeros((20, 2))

    def predict_trajectory(self, data, prediction_steps: int = 60) -> np.ndarray:
        """
        使用LSTM预测轨迹

        Args:
            data: PyG数据对象
            prediction_steps: 预测步数

        Returns:
            trajectory: 预测轨迹 [steps, 2]
        """
        if prediction_steps > self.model.prediction_length:
            raise ValueError(
                f"prediction_steps ({prediction_steps}) exceeds model.prediction_length "
                f"({self.model.prediction_length})"
            )

        with torch.no_grad():
            # 提取ego历史轨迹
            ego_history = self.extract_ego_history_from_data(data)

            # Keep the zero-prepend convention to match train_lstm.py preprocessing.
            ego_increments = np.diff(ego_history, axis=0, prepend=ego_history[:1])

            # 转换为张量并添加batch维度
            history_tensor = torch.FloatTensor(ego_increments).unsqueeze(0).to(self.device)

            # LSTM预测
            predicted_increments = self.model(history_tensor)[0, :prediction_steps, :]
            predicted_increments = predicted_increments.cpu().numpy()

            # 将增量转换为绝对位置
            trajectory = np.zeros((prediction_steps, 2))
            current_pos = ego_history[-1].copy()  # 从最后一个历史位置开始

            for i in range(prediction_steps):
                current_pos += predicted_increments[i]
                trajectory[i] = current_pos.copy()

        return trajectory


def calculate_metrics(pred_trajectory: np.ndarray, true_trajectory: np.ndarray) -> Tuple[float, float]:
    """
    计算ADE和FDE指标

    Args:
        pred_trajectory: 预测轨迹 [steps, 2]
        true_trajectory: 真实轨迹 [steps, 2]

    Returns:
        ade: 平均位移误差
        fde: 最终位移误差
    """
    if pred_trajectory.ndim != 2 or true_trajectory.ndim != 2 or pred_trajectory.shape[1] != 2 or true_trajectory.shape[1] != 2:
        raise ValueError(
            f"Expected trajectories with shape [steps, 2], got {pred_trajectory.shape} and {true_trajectory.shape}"
        )

    min_steps = min(len(pred_trajectory), len(true_trajectory))
    if min_steps == 0:
        raise ValueError("Cannot calculate metrics on empty trajectories")

    pred_clipped = pred_trajectory[:min_steps]
    true_clipped = true_trajectory[:min_steps]
    distances = np.linalg.norm(pred_clipped - true_clipped, axis=1)
    ade = np.mean(distances)
    fde = distances[-1]
    return ade, fde


def evaluate_lstm_on_dataset(dataset_path: str,
                            num_samples: int = 100,
                            batch_size: int = 1,
                            model_path: Optional[str] = None) -> Dict:
    """
    在数据集上评估LSTM预测器

    Args:
        dataset_path: 数据集路径
        num_samples: 评估样本数
        batch_size: 批次大小
        model_path: 预训练模型路径

    Returns:
        evaluation_results: 评估结果字典
    """
    print(f"Loading dataset from: {dataset_path}")

    # 使用安全的数据集加载器
    try:
        full_dataset = SafeGraphDataset(dataset_path)
        print(f"Dataset loaded with {len(full_dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        backup_paths = [
            os.path.join(config.PROCESSED_DATA_BY_TYPE_DIR, "processed_high_magnitude_speed"),
            os.path.join(config.PROCESSED_DATA_BY_TYPE_DIR, "processed_near_multiple_vehicles")
        ]

        for backup_path in backup_paths:
            if os.path.exists(backup_path):
                print(f"Trying backup path: {backup_path}")
                try:
                    full_dataset = SafeGraphDataset(backup_path)
                    if len(full_dataset) > 0:
                        print(f"Successfully loaded {len(full_dataset)} samples from backup")
                        break
                except Exception as backup_error:
                    print(f"Backup path failed: {backup_error}")
                    continue
        else:
            print("All dataset paths failed. Creating dummy results.")
            return create_dummy_results()

    if len(full_dataset) == 0:
        print("No valid samples found. Creating dummy results.")
        return create_dummy_results()

    # 选择评估样本
    if num_samples < len(full_dataset):
        indices = np.random.choice(len(full_dataset), min(num_samples, len(full_dataset)), replace=False)
        eval_dataset = Subset(full_dataset, indices)
    else:
        eval_dataset = full_dataset

    if batch_size != 1:
        print(f"Info: overriding batch_size={batch_size} to 1 for sample-wise LSTM evaluation")
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # 初始化LSTM预测器
    lstm_predictor = LSTMTrajPredictor(model_path=model_path)

    # 评估指标
    ade_list = []
    fde_list = []

    print("Starting LSTM evaluation...")

    for batch_idx, data in enumerate(tqdm(eval_loader, desc="Evaluating LSTM")):
        try:
            sample_data = data

            # 检查数据完整性
            if not hasattr(sample_data, 'y') or sample_data.y.numel() == 0:
                print(f"Skipping batch {batch_idx}: missing target data")
                continue

            # 处理目标数据
            true_increments = sample_data.y
            if true_increments.dim() == 1:
                if true_increments.shape[0] == 120:  # 60 steps * 2 coords
                    true_increments = true_increments.view(60, 2)
                else:
                    print(f"Unexpected target shape: {true_increments.shape}")
                    continue
            elif true_increments.dim() == 2 and true_increments.shape[0] == 60:
                pass  # Already correct format
            else:
                print(f"Unexpected target shape: {true_increments.shape}")
                continue

            true_increments = true_increments.numpy()

            # LSTM预测
            pred_trajectory = lstm_predictor.predict_trajectory(sample_data, prediction_steps=60)

            # 获取参考位置用于轨迹恢复
            ego_history = lstm_predictor.extract_ego_history_from_data(sample_data)
            ego_pos = ego_history[-1] if len(ego_history) > 0 else np.array([0.0, 0.0])

            # 将真实增量转换为绝对轨迹
            true_trajectory = np.zeros((60, 2))
            current_pos = ego_pos.copy()
            for step in range(60):
                current_pos += true_increments[step]
                true_trajectory[step] = current_pos.copy()

            # 计算指标
            ade, fde = calculate_metrics(pred_trajectory, true_trajectory)
            ade_list.append(ade)
            fde_list.append(fde)

            # 限制评估样本数以节省时间
            if len(ade_list) >= num_samples:
                break

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue

    if not ade_list:
        print("No successful evaluations. Creating dummy results.")
        return create_dummy_results()

    # 计算统计结果
    results = {
        'num_samples': len(ade_list),
        'mean_ade': np.mean(ade_list),
        'std_ade': np.std(ade_list),
        'mean_fde': np.mean(fde_list),
        'std_fde': np.std(fde_list),
        'median_ade': np.median(ade_list),
        'median_fde': np.median(fde_list),
        'min_ade': np.min(ade_list),
        'max_ade': np.max(ade_list),
        'min_fde': np.min(fde_list),
        'max_fde': np.max(fde_list),
        'ade_list': ade_list,
        'fde_list': fde_list
    }

    return results


def create_dummy_results() -> Dict:
    """创建虚拟结果用于演示"""
    print("WARNING: returning dummy LSTM evaluation results. These metrics are not from a real model run.")
    np.random.seed(42)
    dummy_ade = np.random.normal(12.0, 4.0, 50)  # LSTM通常比IDM表现更好
    dummy_fde = np.random.normal(20.0, 6.0, 50)

    return {
        'is_dummy': True,
        'num_samples': 50,
        'mean_ade': np.mean(dummy_ade),
        'std_ade': np.std(dummy_ade),
        'mean_fde': np.mean(dummy_fde),
        'std_fde': np.std(dummy_fde),
        'median_ade': np.median(dummy_ade),
        'median_fde': np.median(dummy_fde),
        'min_ade': np.min(dummy_ade),
        'max_ade': np.max(dummy_ade),
        'min_fde': np.min(dummy_fde),
        'max_fde': np.max(dummy_fde),
        'ade_list': dummy_ade.tolist(),
        'fde_list': dummy_fde.tolist()
    }


def visualize_results(results: Dict, save_path: str = None):
    """
    可视化评估结果

    Args:
        results: 评估结果
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ADE分布
    ax1.hist(results['ade_list'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax1.axvline(results['mean_ade'], color='red', linestyle='--', label=f'Mean: {results["mean_ade"]:.2f}m')
    ax1.axvline(results['median_ade'], color='green', linestyle='--', label=f'Median: {results["median_ade"]:.2f}m')
    ax1.set_xlabel('ADE (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('LSTM - Average Displacement Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FDE分布
    ax2.hist(results['fde_list'], bins=30, alpha=0.7, color='darkgreen', edgecolor='black')
    ax2.axvline(results['mean_fde'], color='red', linestyle='--', label=f'Mean: {results["mean_fde"]:.2f}m')
    ax2.axvline(results['median_fde'], color='green', linestyle='--', label=f'Median: {results["median_fde"]:.2f}m')
    ax2.set_xlabel('FDE (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('LSTM - Final Displacement Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results visualization saved to: {save_path}")
        plt.close(fig)
        return

    plt.show()


def main():
    """主函数"""
    print("LSTM Trajectory Predictor for GNN_Trans Data Format")
    print("=" * 60)

    # 配置参数
    dataset_path = config.TRAIN_PATH
    num_samples = 3000  # 增加样本数量
    batch_size = 1

    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples to evaluate: {num_samples}")
    print("Note: Using untrained LSTM model - results will be poor without training")

    # 运行评估
    results = evaluate_lstm_on_dataset(dataset_path, num_samples, batch_size)

    # 打印结果
    print("\n" + "=" * 60)
    print("LSTM Evaluation Results:")
    print("=" * 60)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Mean ADE: {results['mean_ade']:.3f} ± {results['std_ade']:.3f} m")
    print(f"Mean FDE: {results['mean_fde']:.3f} ± {results['std_fde']:.3f} m")
    print(f"Median ADE: {results['median_ade']:.3f} m")
    print(f"Median FDE: {results['median_fde']:.3f} m")
    print("=" * 60)

    # 保存结果
    results_dir = config.LSTM_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    # 保存数值结果
    results_file = os.path.join(results_dir, "lstm_evaluation_results.json")
    with open(results_file, 'w') as f:
        json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    print(f"Results saved to: {results_file}")

    # 可视化结果
    viz_file = os.path.join(results_dir, "lstm_evaluation_plot.png")
    visualize_results(results, viz_file)


if __name__ == "__main__":
    main()
