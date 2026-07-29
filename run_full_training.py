"""
完整训练脚本 - Diffusion危险场景生成器

可以在脚本内直接调整所有参数，无需命令行参数。

使用方法:
1. 修改下方 EXPERIMENT_CONFIG 中的参数
2. 运行: python run_full_training.py

参数说明:
- 数据参数: 数据目录、最大样本数
- 模型参数: 各模块配置
- 训练参数: epochs, batch_size, lr
- Loss参数: recon/danger/smooth权重, danger_margin
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import json
import config

# ============================================================
# 实验配置 - 在这里调整所有参数
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置 - 直接修改这里的参数"""

    # ========== 数据配置 ==========
    processed_data_dir: str = config.PROCESSED_DATA_DIR
    danger_pair_dir: str = config.DANGER_PAIR_SCENARIOS_DIR
    max_samples: int = None  # None表示使用全部数据，或设置如100进行调试

    # ========== 训练配置 ==========
    epochs: int = 50
    batch_size: int = 8  # 建议4-8，GNN内存占用较大
    learning_rate: float = 1e-4
    device: str = 'cuda'  # 'cuda' 或 'cpu'

    # ========== Loss权重配置 ==========
    # Loss公式: total = recon_weight * recon + danger_weight * danger_exceed
    recon_weight: float = 1.0      # 重建loss权重（学习基本轨迹）
    danger_weight: float = 2.0     # danger超越loss权重（生成更危险轨迹）
    danger_margin: float = 0.5     # pred_danger必须超过gt_danger多少才算达标
    proximity_weight: float = 0.0  # 几何趋近loss权重（引导两车路径在空间上会合）
    proximity_threshold: float = 8.0  # m，两车最近距离超过此值则惩罚

    # ========== Diffusion配置 ==========
    num_diffusion_steps: int = 100  # 去噪步数（越多质量越好但慢）
    diffusion_beta_schedule: str = 'linear'  # 'linear' 或 'cosine'

    # ========== 模型配置 ==========
    scene_feat_dim: int = 384      # GNN+Trans输出维度
    condition_dim: int = 256       # 条件向量维度
    denoiser_hidden_dim: int = 256
    denoiser_num_heads: int = 8
    denoiser_num_layers: int = 4

    # ========== 保存配置 ==========
    output_dir: str = config.DIFFUSION_TRAINING_EXPERIMENTS_DIR
    experiment_name: str = "experiment_001"  # 实验名称，用于区分不同配置
    save_every: int = 10           # 每隔多少epoch保存checkpoint
    init_checkpoint: str = None    # 从已有 checkpoint 初始化（fine-tune 用）

    # ========== 时间窗口配置 ==========
    num_past_poses: int = 40       # 4秒历史
    num_future_poses: int = 60     # 6秒未来

    def to_dict(self) -> dict:
        """转换为字典，用于保存配置"""
        return {
            'processed_data_dir': self.processed_data_dir,
            'danger_pair_dir': self.danger_pair_dir,
            'max_samples': self.max_samples,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'recon_weight': self.recon_weight,
            'danger_weight': self.danger_weight,
            'danger_margin': self.danger_margin,
            'num_diffusion_steps': self.num_diffusion_steps,
            'diffusion_beta_schedule': self.diffusion_beta_schedule,
            'scene_feat_dim': self.scene_feat_dim,
            'condition_dim': self.condition_dim,
            'denoiser_hidden_dim': self.denoiser_hidden_dim,
            'denoiser_num_heads': self.denoiser_num_heads,
            'denoiser_num_layers': self.denoiser_num_layers,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'save_every': self.save_every,
        }


# ============================================================
# 预定义实验配置 - 选择一个使用
# ============================================================

EXPERIMENTS = {
    # 实验1: 默认配置
    'default': ExperimentConfig(
        experiment_name='default',
        epochs=50,
        batch_size=8,
        danger_weight=2.0,
        danger_margin=0.5,
    ),

    # 实验2: 快速调试（少量数据）
    'quick_debug': ExperimentConfig(
        experiment_name='quick_debug',
        epochs=10,
        batch_size=4,
        max_samples=50,
        danger_weight=1.0,
        danger_margin=0.2,
    ),

    # 实验3: 强danger超越
    'strong_danger': ExperimentConfig(
        experiment_name='strong_danger',
        epochs=100,
        batch_size=8,
        danger_weight=5.0,  # 更强调danger超越
        danger_margin=1.0,  # 更高的超越目标
        learning_rate=5e-5,  # 更小的lr避免不稳定
    ),

    # 实验4: 稳定训练（降低danger权重）
    'stable': ExperimentConfig(
        experiment_name='stable',
        epochs=100,
        batch_size=8,
        danger_weight=1.0,  # 降低danger权重
        danger_margin=0.3,
        learning_rate=1e-4,
    ),

    # 实验5: 全量数据训练
    'full_training': ExperimentConfig(
        experiment_name='full_training',
        epochs=100,
        batch_size=8,
        max_samples=None,  # 使用全部564个danger_pair
        danger_weight=2.0,
        danger_margin=0.5,
    ),
}

# ============================================================
# 选择当前使用的实验配置
# ============================================================

# 修改这里来选择不同的实验配置
CURRENT_EXPERIMENT = 'default'

# 或者直接创建自定义配置
CUSTOM_CONFIG = ExperimentConfig(
    experiment_name='v3_18_more_data',
    epochs=100,
    batch_size=4,
    max_samples=None,
    danger_weight=2.0,
    danger_margin=0.5,
    proximity_weight=2.0,
    proximity_threshold=8.0,
    init_checkpoint=None,
    processed_data_dir=config.DIFFUSION_PROCESSED_DATA_DIR,  # unused with new dataset
    danger_pair_dir=config.NEW_DANGER_PAIR_DIR,
)


# ============================================================
# 获取最终配置
# ============================================================

def get_config() -> ExperimentConfig:
    """
    获取训练配置

    可以:
    1. 使用预定义的EXPERIMENTS['default']
    2. 使用自定义的CUSTOM_CONFIG
    3. 直接返回新的ExperimentConfig()
    """
    # 方式1: 使用预定义实验
    # return EXPERIMENTS[CURRENT_EXPERIMENT]

    # 方式2: 使用自定义配置
    return CUSTOM_CONFIG

    # 方式3: 直接创建新配置
    # return ExperimentConfig(epochs=50, danger_weight=3.0, ...)


# ============================================================
# 导入模块
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'models'))
sys.path.insert(0, os.path.join(current_dir, 'utils'))

# 添加models目录
models_path = '/mnt/local/Xiao_Yi/nuplan-devkit/models'
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# 导入模块
from scene_condition_encoder import SceneConditionEncoder, ConditionEncoderConfig
from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
from danger_score_torch import (
    DangerExceedLoss, DangerScoreConfigTorch,
    compute_danger_score_torch, recover_trajectory_from_increment,
)
from danger_score import DangerScoreConfig, compute_danger_score, compute_pair_svo
from dataset import GraphData
from diffusion_danger_model.dataset import DangerPairDataset

# 导入TransformerPredictor
try:
    from transformer_predictor import TransformerPredictor
except ImportError as e:
    print(f"[ERROR] TransformerPredictor导入失败: {e}")
    sys.exit(1)


# ============================================================
# 数据集类（从train_diffusion_gnn_trans.py复用）
# ============================================================

class IntegratedDiffusionDataset(Dataset):
    """整合数据集"""

    def __init__(self, processed_data_dir, danger_pair_dir, max_samples=None, danger_config=None):
        self.processed_data_dir = Path(processed_data_dir)
        self.danger_pair_dir = Path(danger_pair_dir)
        self.danger_config = danger_config if danger_config else DangerScoreConfig()

        self.danger_pair_files = list(self.danger_pair_dir.glob('danger_pair_features_*.pkl'))
        if max_samples:
            self.danger_pair_files = self.danger_pair_files[:max_samples]

        print(f"[Dataset] 找到 {len(self.danger_pair_files)} 个danger_pair文件")

        self.data_list = []
        failed_count = 0

        for pkl_file in tqdm(self.danger_pair_files, desc="Loading"):
            try:
                data = self._load_and_process_data(pkl_file)
                if data is not None:
                    self.data_list.append(data)
            except Exception as e:
                failed_count += 1

        print(f"[Dataset] 成功加载 {len(self.data_list)} 个样本，失败 {failed_count} 个")

    def _load_and_process_data(self, danger_pair_file):
        scenario_name = danger_pair_file.stem.replace('danger_pair_features_', '').split('_agent')[0]
        processed_file = self.processed_data_dir / f"features_{scenario_name}.pkl"

        if not processed_file.exists():
            return None

        danger_pair_data = pd.read_pickle(danger_pair_file)
        processed_data = pd.read_pickle(processed_file)

        polyline_features = processed_data.iloc[0]['POLYLINE_FEATURES']
        traj_id_to_indices = processed_data.iloc[0]['TRAJ_ID_TO_INDICES']

        graph_data = self._build_graph_data(polyline_features, traj_id_to_indices)

        ego_history = danger_pair_data['ego_history']
        agent_history = danger_pair_data['key_agent_history']
        ego_future_gt = danger_pair_data['ego_future_gt']

        ego_last_pos     = ego_history[-1, :2]
        ego_last_heading = ego_history[-1, 2]
        ego_last_vx      = ego_history[-1, 3]
        ego_last_vy      = ego_history[-1, 4]

        # (Δx,Δy) increments: 60 steps × 2 = 120 dims (avoids frozen-velocity IK bug)
        gt_increments_ego = self._compute_increments(ego_future_gt, ego_last_pos)

        agent_future = danger_pair_data.get('key_agent_future')
        if agent_future is not None:
            agent_future_full = self._process_agent_future(agent_future, agent_history)
        else:
            agent_future_full = self._extrapolate_agent(agent_history, 60)

        agent_last_pos = agent_history[-1, :2]

        # Joint (Δx,Δy) increments: [ego_120 | agent_120] = [240]
        gt_increments_agent = self._compute_increments(agent_future_full, agent_last_pos)
        joint_increments = np.concatenate([gt_increments_ego, gt_increments_agent])

        # Fix frozen velocity fields before danger score computation
        ego_future_gt    = self._fix_velocity_fields(ego_future_gt,    ego_last_pos)
        agent_future_full = self._fix_velocity_fields(agent_future_full, agent_last_pos)

        svo_info = compute_pair_svo(ego_history, ego_future_gt, agent_history, agent_future_full, self.danger_config)
        danger_result = compute_danger_score(ego_history, ego_future_gt, agent_history, agent_future_full, self.danger_config)

        return {
            'graph_data':       graph_data,
            'ego_history':      ego_history,
            'agent_history':    agent_history,
            'ego_future_gt':    ego_future_gt,
            'agent_future_gt':  agent_future_full,
            'gt_increments':    gt_increments_ego,    # ego (Δx,Δy) increments (120,)
            'joint_increments': joint_increments,      # ego+agent (Δx,Δy) increments (240,)
            'ego_last_pos':     ego_last_pos,
            'ego_last_heading': ego_last_heading,
            'ego_last_vx':      ego_last_vx,
            'ego_last_vy':      ego_last_vy,
            'svo_angle':        svo_info['ego_svo_angle_deg'],
            'inherent_danger':  danger_result['inherent_danger']['total_score'],
            'gt_outcome_danger': danger_result['outcome_danger']['total_score'],
            'danger_target':    danger_result['outcome_danger']['total_score'] + 2.0,
            'scenario_name':    scenario_name,
        }

    def _build_graph_data(self, polyline_features, traj_id_to_indices):
        x = torch.tensor(polyline_features[:, :9], dtype=torch.float32)
        num_nodes = x.size(0)

        positions = polyline_features[:, :2]
        distance_threshold = 30.0
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(positions[i] - positions[j]) <= distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])

        if len(edges) > 0:
            gnn_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        ego_history_tensor = torch.zeros(40, 2, dtype=torch.float32)

        return GraphData(x=x, gnn_x=x.clone(), gnn_edge_index=gnn_edge_index, ego_history=ego_history_tensor)

    def _fix_velocity_fields(self, traj_5col: np.ndarray, last_pos: np.ndarray,
                              dt: float = 0.1) -> np.ndarray:
        """Recompute vx, vy from position differences.

        PKL files store vx,vy frozen at the last history value.
        Frame 0 of future = same position as last history → backward diff gives 0.
        Use forward diff for frame 0, backward diff for frames 1+.
        """
        fixed = traj_5col.copy()
        positions = traj_5col[:, :2]
        T = len(positions)
        # Frame 0: forward difference (position[1] - position[0])
        if T > 1:
            fixed[0, 3] = (positions[1, 0] - positions[0, 0]) / dt
            fixed[0, 4] = (positions[1, 1] - positions[0, 1]) / dt
        # Frames 1+: backward difference (position[t] - position[t-1])
        for t in range(1, T):
            fixed[t, 3] = (positions[t, 0] - positions[t - 1, 0]) / dt
            fixed[t, 4] = (positions[t, 1] - positions[t - 1, 1]) / dt
        return fixed

    def _compute_increments(self, traj, start_pos):
        increments = np.zeros((traj.shape[0], 2))
        increments[0] = traj[0, :2] - start_pos
        for t in range(1, traj.shape[0]):
            increments[t] = traj[t, :2] - traj[t-1, :2]
        return increments.flatten()

    def _compute_cumulative_kinematic(self, traj_5col):
        """Convert [60,5] trajectory to cumulative kinematic (v_norm, θ_abs) [120].

        Stores absolute speed (normalised by 20 m/s) and heading per frame instead
        of increments.  For straight driving at 15 m/s the output is [0.75, θ, 0.75, θ, ...]
        which is a non-zero SDEdit anchor — unlike incremental space where GT ≈ 0.
        """
        speeds = np.sqrt(traj_5col[:, 3] ** 2 + traj_5col[:, 4] ** 2)  # [60]
        headings = traj_5col[:, 2]                                        # [60]
        v_norm = speeds / 20.0                                            # normalise to ~[0, 2]
        return np.stack([v_norm, headings], axis=1).flatten().astype(np.float32)  # [120]

    def _process_agent_future(self, agent_future, agent_history):
        if len(agent_future) < 60:
            agent_future_2d = agent_future[:, :2]
            padding = np.zeros((60 - len(agent_future), 2))
            agent_future_2d = np.concatenate([agent_future_2d, padding])
        else:
            agent_future_2d = agent_future[:60, :2]

        agent_future_full = np.zeros((60, 5))
        agent_future_full[:, :2] = agent_future_2d
        agent_future_full[:, 2] = agent_history[-1, 2]
        agent_future_full[:, 3] = agent_history[-1, 3]
        agent_future_full[:, 4] = agent_history[-1, 4]
        return agent_future_full

    def _extrapolate_agent(self, agent_history, target_len):
        last_pos = agent_history[-1, :2]
        last_vx = agent_history[-1, 3]
        last_vy = agent_history[-1, 4]

        agent_future = np.zeros((target_len, 5))
        dt = 0.1
        for t in range(target_len):
            agent_future[t, 0] = last_pos[0] + last_vx * (t + 1) * dt
            agent_future[t, 1] = last_pos[1] + last_vy * (t + 1) * dt
            agent_future[t, 2] = agent_history[-1, 2]
            agent_future[t, 3] = last_vx
            agent_future[t, 4] = last_vy
        return agent_future

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return {
            'graph_data': data['graph_data'],
            'ego_history': torch.tensor(data['ego_history'], dtype=torch.float32),
            'agent_history': torch.tensor(data['agent_history'], dtype=torch.float32),
            'ego_future_gt': torch.tensor(data['ego_future_gt'], dtype=torch.float32),
            'agent_future_gt': torch.tensor(data['agent_future_gt'], dtype=torch.float32),
            'gt_increments': torch.tensor(data['gt_increments'], dtype=torch.float32),
            'joint_increments': torch.tensor(data['joint_increments'], dtype=torch.float32),
            'ego_last_pos': torch.tensor(data['ego_last_pos'], dtype=torch.float32),
            'ego_last_heading': torch.tensor(data['ego_last_heading'], dtype=torch.float32),
            'ego_last_vx': torch.tensor(data['ego_last_vx'], dtype=torch.float32),
            'ego_last_vy': torch.tensor(data['ego_last_vy'], dtype=torch.float32),
            'svo_angle': torch.tensor([data['svo_angle']], dtype=torch.float32),
            'inherent_danger': torch.tensor([data['inherent_danger']], dtype=torch.float32),
            'gt_outcome_danger': torch.tensor([data['gt_outcome_danger']], dtype=torch.float32),
            'danger_target': torch.tensor([data['danger_target']], dtype=torch.float32),
            'scenario_name': data['scenario_name'],
        }


def collate_fn(batch):
    from torch_geometric.data import Batch

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32)

    batched_graph = Batch.from_data_list([s['graph_data'] for s in batch])

    return {
        'batched_graph':   batched_graph,
        'ego_history':     torch.stack([_t(s['ego_history'])     for s in batch]),
        'agent_history':   torch.stack([_t(s['agent_history'])   for s in batch]),
        'ego_future_gt':   torch.stack([_t(s['ego_future_gt'])   for s in batch]),
        'agent_future_gt': torch.stack([_t(s['agent_future_gt']) for s in batch]),
        'gt_increments':   torch.stack([_t(s['gt_increments'])   for s in batch]),
        'joint_increments':torch.stack([_t(s['joint_increments'])for s in batch]),
        'ego_last_pos':    torch.stack([_t(s['ego_last_pos'])    for s in batch]),
        'ego_last_heading':torch.tensor([s['ego_last_heading'] for s in batch], dtype=torch.float32),
        'ego_last_vx':     torch.tensor([s['ego_last_vx']      for s in batch], dtype=torch.float32),
        'ego_last_vy':     torch.tensor([s['ego_last_vy']      for s in batch], dtype=torch.float32),
        'svo_angle':         torch.tensor([[s['svo_angle']]         for s in batch], dtype=torch.float32),
        'inherent_danger':   torch.tensor([[s['inherent_danger']]   for s in batch], dtype=torch.float32),
        'gt_outcome_danger': torch.tensor([[s['gt_outcome_danger']] for s in batch], dtype=torch.float32),
        'danger_target':     torch.tensor([[s['danger_target']]     for s in batch], dtype=torch.float32),
        'scenario_names':  [s['scenario_name'] for s in batch],
    }


# ============================================================
# 训练函数
# ============================================================

def train_epoch(diffusion_model, gnn_trans_encoder, condition_encoder, loss_fn,
                danger_config_torch, dataloader, optimizer, device, epoch):
    """训练一个epoch"""

    diffusion_model.train()
    gnn_trans_encoder.train()
    condition_encoder.train()

    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'danger_exceed_loss': 0.0,
        'direction_reversal_loss': 0.0,
        'jerk_loss': 0.0,
        'proximity_loss': 0.0,
        'min_dist_sum': 0.0,
        'margin_met_count': 0,
        'pred_danger_sum': 0.0,
        'gt_danger_sum': 0.0,
        'num_batches': 0,
    }

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # 移动数据到device
        batched_graph = batch['batched_graph'].to(device)
        gt_increments = batch['gt_increments'].to(device)
        joint_increments = batch['joint_increments'].to(device)
        ego_history = batch['ego_history'].to(device)
        agent_history = batch['agent_history'].to(device)
        ego_last_pos = batch['ego_last_pos'].to(device)
        ego_last_heading = batch['ego_last_heading'].to(device)
        ego_last_vx = batch['ego_last_vx'].to(device)
        ego_last_vy = batch['ego_last_vy'].to(device)
        svo_angle = batch['svo_angle'].to(device)
        inherent_danger = batch['inherent_danger'].to(device)
        danger_target = batch['danger_target'].to(device)
        gt_outcome_danger = batch['gt_outcome_danger'].to(device)

        optimizer.zero_grad()

        # 1. GNN+Transformer编码
        scene_feat = gnn_trans_encoder.get_encoder_features(batched_graph)

        # 2. 条件编码
        condition = condition_encoder(scene_feat, svo_angle, inherent_danger, danger_target)

        # 3. Diffusion前向传播（对联合240维增量加噪）
        batch_size = joint_increments.size(0)
        t = torch.randint(0, diffusion_model.config.num_diffusion_steps, (batch_size,), device=device)
        x_t, noise = diffusion_model.add_noise(joint_increments, t)
        noise_pred = diffusion_model.denoiser(x_t, condition, t)

        # 4. 恢复x_0（240维：ego_120 + agent_120）
        alpha_bar_t = diffusion_model.alpha_bars[t]
        x_0_pred = (x_t - (1 - alpha_bar_t).sqrt().unsqueeze(-1) * noise_pred) / \
                   alpha_bar_t.sqrt().unsqueeze(-1)
        # nan_to_num 处理 NaN/Inf（高噪声步除零保护）；然后 out-of-place clamp 到物理范围
        # 防止 600 m/s 天文速度导致 proximity_loss 爆炸（3600m → proximity=3592 → 梯度 NaN）
        # 注意：必须用 out-of-place (scalar clamp)，in-place 会触发 autograd "modified inplace" 错误
        x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=4.0, neginf=-4.0)
        x_0_pred = x_0_pred.clamp(min=-4.0, max=4.0)   # out-of-place: 新tensor，不破坏计算图

        # 5. 计算完整Loss（loss_fn内部拆分ego/agent并计算联合危险度）
        total_loss, loss_dict = loss_fn(
            x_0_pred, joint_increments,
            ego_history, agent_history,
            ego_last_pos, ego_last_heading, ego_last_vx, ego_last_vy,
            gt_outcome_danger.squeeze(-1),
            svo_angle,
            danger_target,
        )

        # 6. 替换 x_0-空间重建loss为 ε-空间重建loss（标准DDPM目标）
        # x_0-空间 MSE 在 t≈99 时被 1/√α̅_t ≈ 14× 放大，梯度震荡导致denoiser无法收敛
        # ε-空间 MSE 对所有 t 尺度一致（ε ~ N(0,1)），给denoiser稳定的学习信号
        eps_recon_loss = F.mse_loss(noise_pred, noise)
        total_loss = (total_loss
                      - loss_fn.recon_weight * loss_dict['recon_loss']
                      + loss_fn.recon_weight * eps_recon_loss)
        loss_dict['recon_loss'] = eps_recon_loss

        # 反向传播：若 loss 仍为 NaN/Inf（极端情况），跳过该 batch 保护权重
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            optimizer.zero_grad()
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(gnn_trans_encoder.parameters()) +
            list(condition_encoder.parameters()) +
            list(diffusion_model.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        # 记录指标
        metrics['total_loss'] += total_loss.item()
        metrics['recon_loss'] += loss_dict['recon_loss'].item()
        metrics['danger_exceed_loss'] += loss_dict['danger_exceed_loss'].item()
        metrics['direction_reversal_loss'] += loss_dict['direction_reversal_loss'].item()
        metrics['jerk_loss'] += loss_dict['jerk_loss'].item()
        metrics['proximity_loss'] += loss_dict['proximity_loss'].item()
        metrics['min_dist_sum'] += loss_dict['min_dist_mean']
        metrics['margin_met_count'] += loss_dict['danger_margin_met']
        metrics['pred_danger_sum'] += loss_dict['pred_danger']
        metrics['gt_danger_sum'] += loss_dict['gt_danger']
        metrics['num_batches'] += 1

        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.3f}",
            'recon': f"{loss_dict['recon_loss'].item():.2f}",
            'danger': f"{loss_dict['danger_exceed_loss'].item():.2f}",
            'met': f"{loss_dict['danger_margin_met']:.0%}"
        })

    # 计算平均值
    n = metrics['num_batches']
    return {
        'total_loss': metrics['total_loss'] / n,
        'recon_loss': metrics['recon_loss'] / n,
        'danger_exceed_loss': metrics['danger_exceed_loss'] / n,
        'direction_reversal_loss': metrics['direction_reversal_loss'] / n,
        'jerk_loss': metrics['jerk_loss'] / n,
        'proximity_loss': metrics['proximity_loss'] / n,
        'min_dist_mean': metrics['min_dist_sum'] / n,
        'margin_met_rate': metrics['margin_met_count'] / n,
        'pred_danger_mean': metrics['pred_danger_sum'] / n,
        'gt_danger_mean': metrics['gt_danger_sum'] / n,
    }


# ============================================================
# 主训练函数
# ============================================================

def run_training(config: ExperimentConfig):
    """
    运行完整训练流程
    """
    print("=" * 70)
    print("Diffusion危险场景生成器 - 完整训练")
    print("=" * 70)
    print(f"实验名称: {config.experiment_name}")

    # 创建输出目录
    output_path = Path(config.output_dir) / config.experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存配置
    config_file = output_path / f'config_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"配置已保存: {config_file}")

    # 检查设备
    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA不可用，使用CPU")
        device = 'cpu'

    # 加载数据
    print("\n" + "=" * 70)
    print("1. 加载训练数据")
    print("=" * 70)

    dataset = DangerPairDataset(
        danger_pair_dir=config.danger_pair_dir,
        max_samples=config.max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num batches: {len(dataloader)}")

    # 创建模型
    print("\n" + "=" * 70)
    print("2. 创建模型")
    print("=" * 70)

    # GNN+Transformer
    gnn_trans_encoder = TransformerPredictor().to(device)

    # Condition Encoder
    condition_config = ConditionEncoderConfig(
        scene_feat_dim=config.scene_feat_dim,
        condition_dim=config.condition_dim
    )
    condition_encoder = SceneConditionEncoder(condition_config).to(device)

    # Diffusion
    diffusion_config = DiffusionConfig(
        num_diffusion_steps=config.num_diffusion_steps,
        beta_schedule=config.diffusion_beta_schedule,
        denoiser_hidden_dim=config.denoiser_hidden_dim,
        denoiser_num_heads=config.denoiser_num_heads,
        denoiser_num_layers=config.denoiser_num_layers,
    )
    diffusion_model = DangerSceneDiffusion(diffusion_config).to(device)

    # 从 checkpoint 初始化（fine-tune 模式）
    if config.init_checkpoint:
        ckpt_path = Path(config.init_checkpoint) if Path(config.init_checkpoint).is_absolute() else Path(config.output_dir) / config.init_checkpoint
        print(f"\n[Fine-tune] 从 checkpoint 加载: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        gnn_trans_encoder.load_state_dict(ckpt['gnn_trans_encoder'])
        condition_encoder.load_state_dict(ckpt['condition_encoder'])
        diffusion_model.load_state_dict(ckpt['diffusion_model'])
        print(f"[Fine-tune] 三个模块加载成功，来自 epoch={ckpt.get('epoch', '?')}")

    # Danger Loss
    danger_config_torch = DangerScoreConfigTorch()
    loss_fn = DangerExceedLoss(
        recon_weight=config.recon_weight,
        danger_weight=config.danger_weight,
        danger_margin=config.danger_margin,
        proximity_weight=config.proximity_weight,
        proximity_threshold=config.proximity_threshold,
        upper_bound_weight=1.0,
        upper_bound_ratio=2.0,
        jerk_weight=1e-4,
        config=danger_config_torch
    )

    # 统计参数
    total_params = sum(p.numel() for p in gnn_trans_encoder.parameters()) + \
                   sum(p.numel() for p in condition_encoder.parameters()) + \
                   sum(p.numel() for p in diffusion_model.parameters())
    print(f"总参数量: {total_params}")

    # Loss权重
    print(f"\nLoss权重配置:")
    print(f"  recon_weight = {config.recon_weight}")
    print(f"  danger_weight = {config.danger_weight}")
    print(f"  danger_margin = {config.danger_margin}")

    # 优化器
    optimizer = optim.Adam(
        list(gnn_trans_encoder.parameters()) +
        list(condition_encoder.parameters()) +
        list(diffusion_model.parameters()),
        lr=config.learning_rate
    )

    # 训练记录
    history = []
    best_margin_met = -1.0  # 按 margin_met_rate 保存最佳模型，直接对齐评估目标

    # 开始训练
    print("\n" + "=" * 70)
    print("3. 开始训练")
    print("=" * 70)

    for epoch in range(1, config.epochs + 1):
        epoch_metrics = train_epoch(
            diffusion_model, gnn_trans_encoder, condition_encoder,
            loss_fn, danger_config_torch,
            dataloader, optimizer, device, epoch
        )

        # 记录
        history.append({
            'epoch': epoch,
            **epoch_metrics
        })

        # 打印
        print(f"Epoch {epoch}: "
              f"total={epoch_metrics['total_loss']:.3f}, "
              f"recon={epoch_metrics['recon_loss']:.3f}, "
              f"danger={epoch_metrics['danger_exceed_loss']:.3f}, "
              f"dir_rev={epoch_metrics['direction_reversal_loss']:.3f}, "
              f"jerk={epoch_metrics['jerk_loss']:.4f}, "
              f"prox={epoch_metrics['proximity_loss']:.3f}, "
              f"min_dist={epoch_metrics['min_dist_mean']:.1f}m, "
              f"margin_met={epoch_metrics['margin_met_rate']:.1%}, "
              f"pred_danger={epoch_metrics['pred_danger_mean']:.2f}, "
              f"gt_danger={epoch_metrics['gt_danger_mean']:.2f}")

        # 保存最佳模型（按 margin_met_rate，直接对齐评估目标）
        if epoch_metrics['margin_met_rate'] > best_margin_met:
            best_margin_met = epoch_metrics['margin_met_rate']
            print(f"  [Best] 新最优 margin_met={best_margin_met:.1%}（Epoch {epoch}），保存 best_model.pt")
            torch.save({
                'gnn_trans_encoder': gnn_trans_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'metrics': epoch_metrics,
            }, output_path / 'best_model.pt')

        # 定期保存checkpoint
        if epoch % config.save_every == 0:
            torch.save({
                'gnn_trans_encoder': gnn_trans_encoder.state_dict(),
                'condition_encoder': condition_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, output_path / f'checkpoint_epoch_{epoch}.pt')

    # 保存训练历史
    history_file = output_path / f'training_history_{timestamp}.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # 生成训练报告
    print("\n" + "=" * 70)
    print("4. 训练完成 - 生成报告")
    print("=" * 70)

    generate_training_report(config, history, output_path, timestamp)

    print(f"\n输出目录: {output_path}")
    print(f"最佳Margin Met: {best_margin_met:.1%}")

    return {
        'best_margin_met': best_margin_met,
        'output_dir': str(output_path),
        'history': history,
    }


def generate_training_report(config, history, output_path, timestamp):
    """生成训练报告"""

    report = f"""# Diffusion危险场景生成器 - 训练报告

**实验名称**: {config.experiment_name}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 实验配置

| 参数 | 值 |
|------|-----|
| 数据目录 | {config.danger_pair_dir} |
| 最大样本数 | {config.max_samples or '全部'} |
| Epochs | {config.epochs} |
| Batch size | {config.batch_size} |
| Learning rate | {config.learning_rate} |
| Device | {config.device} |

### Loss配置

| 参数 | 值 | 说明 |
|------|-----|------|
| recon_weight | {config.recon_weight} | 重建loss权重 |
| danger_weight | {config.danger_weight} | danger超越权重 |
| proximity_weight | {config.proximity_weight} | 接近度权重 |
| danger_margin | {config.danger_margin} | danger超越阈值 |

---

## 2. 训练过程

### Loss变化趋势

| Epoch | Total | Recon | Danger | DirRev | Jerk | Margin Met |
|-------|-------|-------|--------|--------|------|------------|
"""

    for h in history:
        report += f"| {h['epoch']} | {h['total_loss']:.3f} | {h['recon_loss']:.3f} | {h['danger_exceed_loss']:.3f} | {h['direction_reversal_loss']:.3f} | {h['jerk_loss']:.4f} | {h['margin_met_rate']:.1%} |\n"

    # 统计
    final = history[-1]
    best = min(history, key=lambda x: x['total_loss'])

    report += f"""
### 最终结果

| 指标 | 最终Epoch | 最佳Epoch |
|------|-----------|-----------|
| Total Loss | {final['total_loss']:.4f} | {best['total_loss']:.4f} (Epoch {best['epoch']}) |
| Recon Loss | {final['recon_loss']:.4f} | {best['recon_loss']:.4f} |
| Danger Loss | {final['danger_exceed_loss']:.4f} | {best['danger_exceed_loss']:.4f} |
| Margin Met Rate | {final['margin_met_rate']:.1%} | {best['margin_met_rate']:.1%} |

---

## 3. Danger分析

| 指标 | 最终值 |
|------|--------|
| Pred Danger Mean | {final['pred_danger_mean']:.2f} |
| GT Danger Mean | {final['gt_danger_mean']:.2f} |
| Danger超越比例 | {final['margin_met_rate']:.1%} |

---

## 4. 输出文件

- 配置文件: config_{timestamp}.json
- 训练历史: training_history_{timestamp}.json
- 最佳模型: best_model.pt
- Checkpoints: checkpoint_epoch_*.pt

---

**报告生成完毕**
"""

    report_file = output_path / f'training_report_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"报告已保存: {report_file}")


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""

    # 获取配置
    config = get_config()

    # 打印配置摘要
    print("\n当前配置:")
    print(f"  epochs: {config.epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  max_samples: {config.max_samples or '全部'}")
    print(f"  danger_weight: {config.danger_weight}")
    print(f"  danger_margin: {config.danger_margin}")
    print(f"  experiment_name: {config.experiment_name}")
    print()

    # 运行训练
    result = run_training(config)

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳Margin Met: {result['best_margin_met']:.1%}")
    print(f"输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
