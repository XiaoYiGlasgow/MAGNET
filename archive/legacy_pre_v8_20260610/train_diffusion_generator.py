"""
Diffusion危险场景生成器训练脚本

训练流程：
1. 加载danger_pair_*.pkl数据
2. 使用GNN+Transformer encoder提取场景特征
3. 计算SVO角度和danger指标
4. 训练Diffusion模型：
   - 重建GT轨迹
   - Danger超越（pred_danger > gt_danger）

使用方法：
python train_diffusion_generator.py --data_dir data/danger_pair_scenarios --epochs 100
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse
import json
from datetime import datetime
from tqdm import tqdm

import config

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 确保当前目录在路径中
sys.path.insert(0, current_dir)
# models和utils在当前目录下
sys.path.insert(0, os.path.join(current_dir, 'models'))
sys.path.insert(0, os.path.join(current_dir, 'utils'))

# 导入模块
from scene_condition_encoder import SceneConditionEncoder, SimpleSceneEncoder, ConditionEncoderConfig
from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
from danger_score_torch import (
    DangerExceedLoss, DangerScoreConfigTorch,
    compute_danger_score_torch, recover_trajectory_from_increment
)

# danger_score.py在utils目录
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from danger_score import DangerScoreConfig, compute_danger_score, compute_pair_svo


# ============================================================
# 数据集
# ============================================================

class DangerPairDataset(Dataset):
    """
    Danger Pair数据集

    从danger_pair_*.pkl文件加载数据，用于Diffusion训练。
    """

    def __init__(
        self,
        data_dir: str,
        max_samples: int = None,
        use_simple_encoder: bool = True,
        config: DangerScoreConfig = None
    ):
        """
        Args:
            data_dir: danger_pair_*.pkl文件所在目录
            max_samples: 最大样本数（用于调试）
            use_simple_encoder: 是否使用简单场景编码器（无需GNN+Trans模型）
            config: danger score配置
        """
        self.data_dir = Path(data_dir)
        self.use_simple_encoder = use_simple_encoder
        self.config = config if config is not None else DangerScoreConfig()

        # 查找所有pkl文件
        self.pkl_files = list(self.data_dir.glob('danger_pair_features_*.pkl'))
        if max_samples:
            self.pkl_files = self.pkl_files[:max_samples]

        print(f"[DangerPairDataset] 加载 {len(self.pkl_files)} 个文件")

        # 加载所有数据
        self.data_list = []
        for pkl_file in tqdm(self.pkl_files, desc="Loading data"):
            try:
                data = pd.read_pickle(pkl_file)
                self.data_list.append(self._process_data(data))
            except Exception as e:
                print(f"Warning: Failed to load {pkl_file}: {e}")

        print(f"[DangerPairDataset] 成功加载 {len(self.data_list)} 个样本")

    def _process_data(self, raw_data: dict) -> dict:
        """
        处理单个pkl数据，转换为训练格式

        Args:
            raw_data: 从pkl加载的原始数据

        Returns:
            处理后的数据字典
        """
        # 提取历史轨迹
        ego_history = raw_data['ego_history']  # (40, 5) - 4秒历史
        agent_history = raw_data['key_agent_history']  # (40, 5) - 4秒历史

        # 提取GT未来轨迹
        ego_future_gt = raw_data['ego_future_gt']  # (60, 5) - 6秒未来
        if ego_future_gt.ndim == 3:
            ego_future_gt = ego_future_gt.squeeze(0)

        agent_future = raw_data['key_agent_future']  # (60, 3) or None

        # 计算GT轨迹增量
        # 从ego_history最后帧的位置开始，计算每步的dx, dy
        ego_last_pos = ego_history[-1, :2]  # (x, y)
        ego_last_heading = ego_history[-1, 2]
        ego_last_vx = ego_history[-1, 3]
        ego_last_vy = ego_history[-1, 4]

        # ego_future_gt已经是完整的60帧，5维特征
        # 需要确保帧数正确
        if len(ego_future_gt) < 60:
            # 填充到60帧
            pad_frames = 60 - len(ego_future_gt)
            padding = np.zeros((pad_frames, 5))
            ego_future_gt_full = np.concatenate([ego_future_gt, padding], axis=0)
        else:
            ego_future_gt_full = ego_future_gt[:60]  # 截取前60帧

        ego_future_gt_full_2d = ego_future_gt_full[:, :2]  # (60, 2)

        # 计算增量（基于2D轨迹）
        gt_increments = self._compute_increments(ego_future_gt_full_2d, ego_last_pos)

        # 构建agent未来轨迹 (用于计算danger)
        if agent_future is not None:
            # agent_future: (60, 3) 包含位置和heading
            # 需要补全到5维
            if len(agent_future) < 60:
                agent_future_aligned = agent_future
                # 填充到60帧
                pad_frames = 60 - len(agent_future_aligned)
                agent_future_2d = agent_future_aligned[:, :2]
                padding_pos = np.zeros((pad_frames, 2))
                agent_future_2d_extended = np.concatenate([agent_future_2d, padding_pos], axis=0)
            else:
                agent_future_2d_extended = agent_future[:60, :2]  # (60, 2)

            agent_future_full = np.zeros((60, 5))
            agent_future_full[:, :2] = agent_future_2d_extended

            # heading
            if agent_future.shape[1] >= 3:
                agent_heading_last = agent_future[-1, 2]
            else:
                agent_heading_last = agent_history[-1, 2]
            agent_future_full[:, 2] = agent_heading_last

            # 使用历史最后帧速度
            agent_future_full[:, 3] = agent_history[-1, 3]  # vx
            agent_future_full[:, 4] = agent_history[-1, 4]  # vy
        else:
            # 没有agent未来，使用线性外推
            agent_future_full = self._extrapolate_agent(agent_history, 60)

        # 计算SVO和danger指标（使用完整的60帧未来）
        svo_info = compute_pair_svo(
            ego_history, ego_future_gt_full,
            agent_history, agent_future_full,
            self.config
        )

        # 计算danger score（使用完整的60帧未来）
        danger_result = compute_danger_score(
            ego_history, ego_future_gt_full,
            agent_history, agent_future_full,
            self.config
        )

        inherent_danger = danger_result['inherent_danger']['total_score']
        gt_outcome_danger = danger_result['outcome_danger']['total_score']

        # 目标danger（比GT更高）
        danger_margin = 2.0  # 超越幅度
        danger_target = gt_outcome_danger + danger_margin

        return {
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future_gt': ego_future_gt_full,
            'agent_future_gt': agent_future_full,
            'gt_increments': gt_increments,

            'ego_last_pos': ego_last_pos,
            'ego_last_heading': ego_last_heading,
            'ego_last_vx': ego_last_vx,
            'ego_last_vy': ego_last_vy,

            'svo_angle': svo_info['ego_svo_angle_deg'],
            'inherent_danger': inherent_danger,
            'gt_outcome_danger': gt_outcome_danger,
            'danger_target': danger_target,

            'scenario_name': raw_data.get('scenario_name', 'unknown'),
            'key_agent_id': raw_data.get('key_agent_id', -1)
        }

    def _extend_trajectory(self, traj: np.ndarray, target_len: int) -> np.ndarray:
        """
        扩展轨迹到目标长度

        Args:
            traj: (T, D) 原轨迹
            target_len: 目标长度

        Returns:
            (target_len, D) 扩展后的轨迹
        """
        if traj.shape[0] >= target_len:
            return traj[:target_len]

        # 简单线性插值
        extended = np.zeros((target_len, traj.shape[1]))
        for i in range(target_len):
            idx = i * (traj.shape[0] - 1) / (target_len - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, traj.shape[0] - 1)
            alpha = idx - idx_low

            if idx_low == idx_high:
                extended[i] = traj[idx_low]
            else:
                extended[i] = (1 - alpha) * traj[idx_low] + alpha * traj[idx_high]

        return extended

    def _extrapolate_agent(self, agent_history: np.ndarray, target_len: int) -> np.ndarray:
        """
        线性外推agent未来轨迹

        Args:
            agent_history: (T_hist, 5)
            target_len: 未来轨迹长度

        Returns:
            (target_len, 5) 外推的轨迹
        """
        last_pos = agent_history[-1, :2]
        last_heading = agent_history[-1, 2]
        last_vx = agent_history[-1, 3]
        last_vy = agent_history[-1, 4]

        agent_future = np.zeros((target_len, 5))
        dt = 0.1
        for t in range(target_len):
            agent_future[t, 0] = last_pos[0] + last_vx * (t + 1) * dt
            agent_future[t, 1] = last_pos[1] + last_vy * (t + 1) * dt
            agent_future[t, 2] = last_heading
            agent_future[t, 3] = last_vx
            agent_future[t, 4] = last_vy

        return agent_future

    def _compute_increments(self, traj: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
        """
        计算轨迹增量

        Args:
            traj: (T, D) 轨迹
            start_pos: (2,) 起始位置

        Returns:
            (T*2,) 轨迹增量（flatten）
        """
        increments = np.zeros((traj.shape[0], 2))

        # 第一帧增量：从start_pos到traj[0]
        increments[0] = traj[0, :2] - start_pos

        # 后续帧增量
        for t in range(1, traj.shape[0]):
            increments[t] = traj[t, :2] - traj[t-1, :2]

        return increments.flatten()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """获取单个样本"""
        data = self.data_list[idx]

        # 转换为tensor
        return {
            'ego_history': torch.tensor(data['ego_history'], dtype=torch.float32),
            'agent_history': torch.tensor(data['agent_history'], dtype=torch.float32),
            'ego_future_gt': torch.tensor(data['ego_future_gt'], dtype=torch.float32),
            'agent_future_gt': torch.tensor(data['agent_future_gt'], dtype=torch.float32),
            'gt_increments': torch.tensor(data['gt_increments'], dtype=torch.float32),

            'ego_last_pos': torch.tensor(data['ego_last_pos'], dtype=torch.float32),
            'ego_last_heading': torch.tensor(data['ego_last_heading'], dtype=torch.float32),
            'ego_last_vx': torch.tensor(data['ego_last_vx'], dtype=torch.float32),
            'ego_last_vy': torch.tensor(data['ego_last_vy'], dtype=torch.float32),

            'svo_angle': torch.tensor([data['svo_angle']], dtype=torch.float32),
            'inherent_danger': torch.tensor([data['inherent_danger']], dtype=torch.float32),
            'gt_outcome_danger': torch.tensor([data['gt_outcome_danger']], dtype=torch.float32),
            'danger_target': torch.tensor([data['danger_target']], dtype=torch.float32),

            'scenario_name': data['scenario_name'],
            'key_agent_id': data['key_agent_id']
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    自定义collate函数，处理batch数据
    """
    result = {
        'ego_history': torch.stack([item['ego_history'] for item in batch]),
        'agent_history': torch.stack([item['agent_history'] for item in batch]),
        'ego_future_gt': torch.stack([item['ego_future_gt'] for item in batch]),
        'agent_future_gt': torch.stack([item['agent_future_gt'] for item in batch]),
        'gt_increments': torch.stack([item['gt_increments'] for item in batch]),

        'ego_last_pos': torch.stack([item['ego_last_pos'] for item in batch]),
        'ego_last_heading': torch.stack([item['ego_last_heading'] for item in batch]),
        'ego_last_vx': torch.stack([item['ego_last_vx'] for item in batch]),
        'ego_last_vy': torch.stack([item['ego_last_vy'] for item in batch]),

        'svo_angle': torch.stack([item['svo_angle'] for item in batch]),
        'inherent_danger': torch.stack([item['inherent_danger'] for item in batch]),
        'gt_outcome_danger': torch.stack([item['gt_outcome_danger'] for item in batch]),
        'danger_target': torch.stack([item['danger_target'] for item in batch]),

        'scenario_names': [item['scenario_name'] for item in batch],
        'key_agent_ids': [item['key_agent_id'] for item in batch]
    }

    return result


# ============================================================
# 训练函数
# ============================================================

def train_epoch(
    model: DangerSceneDiffusion,
    condition_encoder: SceneConditionEncoder,
    scene_encoder: SimpleSceneEncoder,
    loss_fn: DangerExceedLoss,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict:
    """
    训练一个epoch
    """
    model.train()
    condition_encoder.train()
    scene_encoder.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_danger_loss = 0.0
    total_smooth_loss = 0.0

    danger_margin_met_count = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # 移动数据到device
        ego_history = batch['ego_history'].to(device)
        agent_history = batch['agent_history'].to(device)
        gt_increments = batch['gt_increments'].to(device)

        ego_last_pos = batch['ego_last_pos'].to(device)
        ego_last_heading = batch['ego_last_heading'].to(device)
        ego_last_vx = batch['ego_last_vx'].to(device)
        ego_last_vy = batch['ego_last_vy'].to(device)

        svo_angle = batch['svo_angle'].to(device)
        inherent_danger = batch['inherent_danger'].to(device)
        gt_outcome_danger = batch['gt_outcome_danger'].to(device)
        danger_target = batch['danger_target'].to(device)

        agent_future_gt = batch['agent_future_gt'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 1. 编码场景特征
        scene_feat = scene_encoder(ego_history, agent_history)

        # 2. 编码条件
        condition = condition_encoder(scene_feat, svo_angle, inherent_danger, danger_target)

        # 3. Diffusion前向传播（添加噪声 + 预测噪声）
        diffusion_loss, diffusion_info = model(gt_increments, condition)

        # 4. 生成轨迹并计算danger loss
        # 使用生成函数（简化：直接使用训练时的噪声预测）
        # 这里我们先只计算diffusion loss，后续可以添加danger exceed loss

        # 组合loss
        # diffusion_loss: MSE(noise_pred, noise)
        # danger_exceed_loss: ReLU(gt_danger - pred_danger + margin)

        # 简化：暂时只用diffusion loss + danger exceed loss
        # 需要从预测恢复轨迹来计算danger

        total_loss_batch = diffusion_loss

        # 反向传播
        total_loss_batch.backward()

        # 更新参数
        optimizer.step()

        # 记录
        total_loss += total_loss_batch.item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f"{total_loss_batch.item():.4f}",
            'avg': f"{total_loss / num_batches:.4f}"
        })

    return {
        'total_loss': total_loss / num_batches,
        'num_batches': num_batches
    }


def train_diffusion(
    data_dir: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = 'cuda',
    max_samples: int = None,
    save_every: int = 10
):
    """
    完整训练流程
    """
    print("=" * 60)
    print("开始训练 Diffusion 危险场景生成器")
    print("=" * 60)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_path / f'train_log_{timestamp}.txt'

    # 加载数据
    print("\n1. 加载训练数据...")
    dataset = DangerPairDataset(data_dir, max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 简化处理
    )

    print(f"   数据集大小: {len(dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Num batches: {len(dataloader)}")

    # 创建模型
    print("\n2. 创建模型...")

    # 场景编码器（简化版）
    scene_encoder = SimpleSceneEncoder(
        ego_history_dim=5,
        agent_history_dim=5,
        output_dim=384,
        history_length=40  # 更新为4秒历史（40帧）
    ).to(device)

    # 条件编码器
    condition_config = ConditionEncoderConfig()
    condition_encoder = SceneConditionEncoder(condition_config).to(device)

    # Diffusion模型
    diffusion_config = DiffusionConfig()
    diffusion_model = DangerSceneDiffusion(diffusion_config).to(device)

    # Loss函数
    danger_config = DangerScoreConfigTorch()
    loss_fn = DangerExceedLoss(
        recon_weight=1.0,
        danger_weight=2.0,
        smoothness_weight=0.3,
        config=danger_config
    )

    # 优化器
    all_params = list(scene_encoder.parameters()) + \
                 list(condition_encoder.parameters()) + \
                 list(diffusion_model.parameters())
    optimizer = optim.Adam(all_params, lr=lr)

    print(f"   总参数量: {sum(p.numel() for p in all_params)}")

    # 训练循环
    print("\n3. 开始训练...")
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # 训练一个epoch
        train_info = train_epoch(
            diffusion_model, condition_encoder, scene_encoder,
            loss_fn, dataloader, optimizer, device, epoch
        )

        # 记录
        log_line = f"Epoch {epoch}: loss = {train_info['total_loss']:.4f}"
        print(log_line)

        with open(log_file, 'a') as f:
            f.write(log_line + '\n')

        # 保存最佳模型
        if train_info['total_loss'] < best_loss:
            best_loss = train_info['total_loss']
            best_model_path = output_path / 'best_model.pt'
            torch.save({
                'scene_encoder': scene_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, best_model_path)

        # 定期保存
        if epoch % save_every == 0:
            checkpoint_path = output_path / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'scene_encoder': scene_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)

    print("\n4. 训练完成！")
    print(f"   最佳loss: {best_loss:.4f}")
    print(f"   模型保存到: {output_path}")

    return {
        'best_loss': best_loss,
        'output_dir': str(output_path)
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='训练Diffusion危险场景生成器')

    parser.add_argument('--data_dir', type=str,
                        default=config.DANGER_PAIR_SCENARIOS_DIR,
                        help='训练数据目录')

    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(config.DIFFUSION_TRAINING_EXPERIMENTS_DIR, 'diffusion_generator'),
                        help='输出目录')

    parser.add_argument('--epochs', type=int, default=100,
                        help='训练epoch数')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')

    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于调试）')

    parser.add_argument('--save_every', type=int, default=10,
                        help='每隔多少epoch保存一次')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA不可用，使用CPU")
        args.device = 'cpu'

    # 开始训练
    result = train_diffusion(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_samples=args.max_samples,
        save_every=args.save_every
    )

    print("\n训练结果:")
    print(f"  最佳loss: {result['best_loss']:.4f}")
    print(f"  输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
