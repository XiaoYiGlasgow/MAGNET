"""
危险场景生成脚本

使用训练好的Diffusion模型生成比GT更危险的场景。

生成流程：
1. 加载历史轨迹数据
2. 编码场景特征和条件
3. Diffusion生成新轨迹
4. 计算生成轨迹的danger
5. 保存为新的训练样本

使用方法：
python generate_danger_scenes.py --model_path experiments/danger_generation/training_runs/full_training/best_model.pt \
                                 --data_dir data/danger_pair_scenarios \
                                 --output_dir experiments/danger_generation/generated_scenes
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from typing import Dict, List, Optional
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
# shared 包在 src/ 下（danger_diffusion 依赖 shared.utils.svo_torch 等）
sys.path.insert(0, os.path.join(current_dir, 'src'))

# 导入模块
from scene_condition_encoder import SceneConditionEncoder, SimpleSceneEncoder, ConditionEncoderConfig
from danger_diffusion import DangerSceneDiffusion, DiffusionConfig
from danger_score_torch import (
    compute_danger_score_torch, recover_trajectory_from_increment,
    xy_displacement_decode_torch, DangerScoreConfigTorch
)

# danger_score.py在utils目录
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from danger_score import DangerScoreConfig, compute_danger_score, compute_pair_svo

from idm_predictor import IDMPredictor

# TransformerPredictor 在 nuplan-devkit/models 目录
_models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
if _models_path not in sys.path:
    sys.path.insert(0, _models_path)
try:
    from transformer_predictor import TransformerPredictor
    from torch_geometric.data import Batch
    _HAS_GNN_ENCODER = True
except ImportError:
    _HAS_GNN_ENCODER = False


# ============================================================
# 生成器类
# ============================================================

class DangerSceneGenerator:
    """
    危险场景生成器

    使用训练好的Diffusion模型生成比GT更危险的轨迹。
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_simple_encoder: bool = True,
        processed_data_dir: str = None,
    ):
        """
        Args:
            model_path: 训练好的模型文件路径
            device: 设备
            use_simple_encoder: 是否使用简单场景编码器（当无对应 features pkl 时的 fallback）
            processed_data_dir: features_{scenario}.pkl 所在目录（用于 GNNTransEncoder）
        """
        self.device = device
        self.use_simple_encoder = use_simple_encoder
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else None

        # 加载模型
        self._load_models(model_path)

        # Danger配置
        self.danger_config_np = DangerScoreConfig()
        self.danger_config_torch = DangerScoreConfigTorch()

        # IDM预测器（为背景agent生成反应轨迹）
        self.idm_predictor = IDMPredictor()

        print(f"[DangerSceneGenerator] 初始化完成")
        print(f"  - Model path: {model_path}")
        print(f"  - Device: {device}")

    def _load_models(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # 判断 checkpoint 类型：新版 gnn_trans_encoder 或旧版 scene_encoder
        if 'gnn_trans_encoder' in checkpoint and _HAS_GNN_ENCODER:
            self.scene_encoder = TransformerPredictor()
            self.scene_encoder.load_state_dict(checkpoint['gnn_trans_encoder'])
            self._use_gnn_encoder = True
        else:
            self.scene_encoder = SimpleSceneEncoder(
                ego_history_dim=5,
                agent_history_dim=5,
                output_dim=384,
                history_length=40,
            )
            scene_key = 'scene_encoder' if 'scene_encoder' in checkpoint else 'gnn_trans_encoder'
            self.scene_encoder.load_state_dict(checkpoint[scene_key])
            self._use_gnn_encoder = False

        self.scene_encoder.to(self.device)
        self.scene_encoder.eval()

        # 条件编码器
        condition_config = ConditionEncoderConfig()
        self.condition_encoder = SceneConditionEncoder(condition_config)
        self.condition_encoder.load_state_dict(checkpoint['condition_encoder'])
        self.condition_encoder.to(self.device)
        self.condition_encoder.eval()

        # Diffusion模型
        diffusion_config = DiffusionConfig()
        self.diffusion_model = DangerSceneDiffusion(diffusion_config)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.diffusion_model.to(self.device)
        self.diffusion_model.eval()

        print(f"  - 成功加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")

    def generate_from_pkl(
        self,
        pkl_path: str,
        svo_angle_override: Optional[float] = None,
        danger_target_override: Optional[float] = None,
        danger_margin: float = 2.0,
        num_samples: int = 1,
        use_sdedit: bool = True,
        sdedit_k: int = 20,
        sdedit_guidance_scale: float = 0.1,
        smooth_sigma: float = 1.0,
    ) -> Dict:
        """
        从单个pkl文件生成危险场景

        Args:
            pkl_path: pkl文件路径
            svo_angle_override: 强制指定的SVO角度（用于测试）
            danger_target_override: 强制指定的danger目标
            danger_margin: danger超越幅度（如果没有override）
            num_samples: 生成样本数

        Returns:
            生成结果字典
        """
        # 加载原始数据
        raw_data = pd.read_pickle(pkl_path)

        # 处理数据
        processed = self._process_raw_data(raw_data)

        # 编码场景特征
        scene_feat = self._encode_scene(processed)

        # 设置条件
        svo_angle = svo_angle_override if svo_angle_override is not None else processed['svo_angle']
        inherent_danger = processed['inherent_danger']
        gt_danger = processed['gt_outcome_danger']
        danger_target = danger_target_override if danger_target_override is not None else gt_danger + danger_margin

        # 编码条件（附带 key agent 末态运动学信息）
        agent_state = processed['agent_history'][-1, 2:5]  # (heading, vx, vy)
        condition = self._encode_condition(
            scene_feat, svo_angle, inherent_danger, danger_target, agent_state=agent_state
        )

        # 生成联合轨迹（240维 = 120 ego + 120 key_agent）
        if use_sdedit:
            # SDEdit: start from noisy GT, denoise with danger guidance
            gt_joint_inc = torch.tensor(
                processed['joint_gt_inc'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, 240]

            ego_hist_t   = torch.tensor(processed['ego_history'],   dtype=torch.float32, device=self.device).unsqueeze(0)
            agent_hist_t = torch.tensor(processed['agent_history'], dtype=torch.float32, device=self.device).unsqueeze(0)
            ego_lp_t     = torch.tensor(processed['ego_last_pos'],  dtype=torch.float32, device=self.device).unsqueeze(0)
            agent_lp_t   = torch.tensor(processed['agent_last_pos'],dtype=torch.float32, device=self.device).unsqueeze(0)

            # Repeat for num_samples
            gt_joint_inc  = gt_joint_inc.expand(num_samples, -1)
            condition_exp = condition.expand(num_samples, -1)
            ego_hist_t    = ego_hist_t.expand(num_samples, -1, -1)
            agent_hist_t  = agent_hist_t.expand(num_samples, -1, -1)
            ego_lp_t      = ego_lp_t.expand(num_samples, -1)
            agent_lp_t    = agent_lp_t.expand(num_samples, -1)

            generated_increments = self._sdedit_with_guidance(
                gt_joint_inc, condition_exp,
                ego_hist_t, agent_hist_t,
                ego_lp_t, agent_lp_t,
                k=sdedit_k,
                guidance_scale=sdedit_guidance_scale,
            )  # [num_samples, 240]
        else:
            generated_increments = self.diffusion_model.generate(condition, num_samples=num_samples)

        # 统一为 2D: [N, 240]
        if generated_increments.dim() == 1:
            generated_increments = generated_increments.unsqueeze(0)

        # 拆分 ego 和 key_agent 增量
        ego_increments   = generated_increments[:, :120]   # [N, 120]
        agent_increments = generated_increments[:, 120:]   # [N, 120]

        # 恢复 ego 轨迹 [N, 60, 5]
        ego_trajectories = self._recover_trajectories(
            ego_increments, processed['ego_last_pos'],
            processed['ego_last_heading'], processed['ego_last_vx'],
            processed['ego_last_vy']
        )

        # 恢复 key_agent 轨迹 [N, 60, 5]
        agent_trajectories = self._recover_trajectories(
            agent_increments, processed['agent_last_pos'],
            processed['agent_last_heading'], processed['agent_last_vx'],
            processed['agent_last_vy']
        )

        # 计算联合 danger（两条生成轨迹的交互危险）
        generated_danger_results = self._compute_generated_danger_joint(
            ego_trajectories, agent_trajectories,
            processed['ego_history'], processed['agent_history']
        )

        # 结果汇总
        results = []
        for i in range(ego_trajectories.size(0)):
            ego_traj_np   = ego_trajectories[i].detach().cpu().numpy()    # [60, 5]
            agent_traj_np = agent_trajectories[i].detach().cpu().numpy()  # [60, 5]

            danger_val = generated_danger_results['outcome_danger'][i].item()

            # Gaussian smoothing on vx/vy channels (danger already computed above, unaffected)
            if smooth_sigma > 0:
                ego_traj_np = ego_traj_np.copy()
                agent_traj_np = agent_traj_np.copy()
                ego_traj_np[:, 3] = gaussian_filter1d(ego_traj_np[:, 3], sigma=smooth_sigma)
                ego_traj_np[:, 4] = gaussian_filter1d(ego_traj_np[:, 4], sigma=smooth_sigma)
                agent_traj_np[:, 3] = gaussian_filter1d(agent_traj_np[:, 3], sigma=smooth_sigma)
                agent_traj_np[:, 4] = gaussian_filter1d(agent_traj_np[:, 4], sigma=smooth_sigma)

            # IDM 预测背景 agent — 使用生成的危险轨迹作为前车（单向依赖，无循环）
            bg_futures = self._predict_boundary_agents(
                processed['boundary_agents'],
                ego_traj_np, agent_traj_np
            )

            result = {
                'sample_idx':           i,
                'generated_increments': generated_increments[i].detach().cpu().numpy(),
                'generated_trajectory': ego_traj_np,       # ego 生成轨迹（向后兼容）
                'agent_trajectory':     agent_traj_np,     # key_agent 生成轨迹
                'generated_danger':     danger_val,
                'gt_danger':            gt_danger,
                'danger_exceeded':      danger_val > gt_danger,

                'svo_angle_used':    svo_angle,
                'danger_target_used': danger_target,

                'scenario_name':  processed['scenario_name'],
                'key_agent_id':   processed['key_agent_id'],

                'background_agent_futures': bg_futures,    # {agent_id: [60,2]}
            }
            results.append(result)

        return {
            'samples': results,
            'condition_info': {
                'svo_angle': svo_angle,
                'inherent_danger': inherent_danger,
                'danger_target': danger_target,
                'gt_danger': gt_danger
            }
        }

    def _process_raw_data(self, raw_data: dict) -> dict:
        """处理原始pkl数据"""
        ego_history = raw_data['ego_history']    # (40或41, 5)
        agent_history = raw_data['key_agent_history']  # (40或41, 5)

        # 构建场景图（与训练 DangerPairDataset._build_graph_data 完全一致）
        graph_data = None
        if self._use_gnn_encoder:
            boundary_agents = raw_data.get('boundary_agents', [])
            graph_data = self._build_graph_from_history(
                ego_history, agent_history, boundary_agents
            )

        ego_future_gt = raw_data['ego_future_gt']  # (60, 5) - 6秒未来
        if ego_future_gt.ndim == 3:
            ego_future_gt = ego_future_gt.squeeze(0)

        agent_future = raw_data['key_agent_future']  # (60, 3) or None

        # ego_future_gt已经是完整的60帧，5维特征
        if len(ego_future_gt) < 60:
            # 填充到60帧
            pad_frames = 60 - len(ego_future_gt)
            padding = np.zeros((pad_frames, 5))
            ego_future_gt_full = np.concatenate([ego_future_gt, padding], axis=0)
        else:
            ego_future_gt_full = ego_future_gt[:60]  # 截取前60帧

        ego_future_gt_2d = ego_future_gt_full[:, :2]  # (60, 2)

        ego_last_pos = ego_history[-1, :2]
        ego_last_heading = float(ego_history[-1, 2])
        ego_last_vx = float(ego_history[-1, 3])
        ego_last_vy = float(ego_history[-1, 4])

        agent_last_pos     = agent_history[-1, :2]
        agent_last_heading = float(agent_history[-1, 2])
        agent_last_vx      = float(agent_history[-1, 3])
        agent_last_vy      = float(agent_history[-1, 4])

        # 构建agent未来轨迹 (60, 5)
        if agent_future is not None:
            agent_future_full = np.zeros((60, 5))
            if len(agent_future) < 60:
                agent_future_2d = agent_future[:, :2]
                pad_frames = 60 - len(agent_future_2d)
                padding = np.zeros((pad_frames, 2))
                agent_future_2d_extended = np.concatenate([agent_future_2d, padding], axis=0)
            else:
                agent_future_2d_extended = agent_future[:60, :2]
            agent_future_full[:, :2] = agent_future_2d_extended

            # heading
            if agent_future.shape[1] >= 3:
                agent_heading_last = agent_future[-1, 2]
            else:
                agent_heading_last = agent_history[-1, 2]
            agent_future_full[:, 2] = agent_heading_last

            # 速度使用历史最后帧
            agent_future_full[:, 3] = agent_history[-1, 3]
            agent_future_full[:, 4] = agent_history[-1, 4]
        else:
            # 没有agent未来，使用线性外推
            agent_future_full = self._extend_trajectory_agent(agent_history, None, 60)

        # ego_future_gt_full may be (60,2) Δx,Δy increments; decode to (60,5) for SVO/danger
        if ego_future_gt_full.shape[1] == 2:
            dt = 0.1
            ego_fut_pos = np.cumsum(ego_future_gt_2d, axis=0) + ego_last_pos[np.newaxis]  # (60,2)
            headings = np.arctan2(ego_future_gt_2d[:, 1], ego_future_gt_2d[:, 0])
            vx_arr = ego_future_gt_2d[:, 0] / dt
            vy_arr = ego_future_gt_2d[:, 1] / dt
            ego_future_for_svo = np.stack(
                [ego_fut_pos[:, 0], ego_fut_pos[:, 1], headings, vx_arr, vy_arr], axis=1
            )  # (60, 5)
        else:
            ego_future_for_svo = ego_future_gt_full  # already (60, 5)

        # 计算SVO和danger（使用完整的60帧未来）
        svo_info = compute_pair_svo(
            ego_history, ego_future_for_svo,
            agent_history, agent_future_full,
            self.danger_config_np
        )

        danger_result = compute_danger_score(
            ego_history, ego_future_for_svo,
            agent_history, agent_future_full,
            self.danger_config_np
        )

        # GT joint increments for SDEdit: ego (Δx,Δy) flat + agent (Δx,Δy) flat → (240,)
        # ego_future_gt_2d is (60, 2) Δx,Δy in new pkl; positions[:,:2] in old pkl
        ego_gt_inc_flat = ego_future_gt_2d.flatten()[:120]  # (120,)
        agent_fut_pos = agent_future_full[:, :2]             # (60, 2) absolute positions
        agent_gt_inc = np.diff(
            np.vstack([agent_last_pos[np.newaxis], agent_fut_pos]), axis=0
        )                                                    # (60, 2)
        joint_gt_inc = np.concatenate([ego_gt_inc_flat, agent_gt_inc.flatten()])  # (240,)

        return {
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future_gt': ego_future_gt_full,
            'agent_future_gt': agent_future_full,

            'ego_last_pos':     ego_last_pos,
            'ego_last_heading': ego_last_heading,
            'ego_last_vx':      ego_last_vx,
            'ego_last_vy':      ego_last_vy,

            'agent_last_pos':     agent_last_pos,
            'agent_last_heading': agent_last_heading,
            'agent_last_vx':      agent_last_vx,
            'agent_last_vy':      agent_last_vy,

            'boundary_agents': raw_data.get('boundary_agents', []),
            'graph_data':      graph_data,

            'svo_angle': svo_info['ego_svo_angle_deg'],
            'inherent_danger': danger_result['inherent_danger']['total_score'],
            'gt_outcome_danger': danger_result['outcome_danger']['total_score'],

            'joint_gt_inc': joint_gt_inc,   # (240,) for SDEdit

            'scenario_name': raw_data.get('scenario_name', 'unknown'),
            'key_agent_id': raw_data.get('key_agent_id', -1)
        }

    def _extend_trajectory(self, traj: np.ndarray, target_len: int) -> np.ndarray:
        """扩展轨迹"""
        if traj.shape[0] >= target_len:
            return traj[:target_len]

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

    def _extend_trajectory_agent(
        self,
        agent_history: np.ndarray,
        agent_future: Optional[np.ndarray],
        target_len: int
    ) -> np.ndarray:
        """扩展agent轨迹"""
        agent_future_full = np.zeros((target_len, 5))

        if agent_future is not None:
            agent_future_full[:, :2] = self._extend_trajectory(agent_future[:, :2], target_len)
            if agent_future.shape[1] >= 3:
                agent_future_full[:, 2] = self._extend_trajectory(agent_future[:, 2:3], target_len).flatten()
        else:
            # 线性外推
            for t in range(target_len):
                dt = (t + 1) * 0.1
                agent_future_full[t, 0] = agent_history[-1, 0] + agent_history[-1, 3] * dt
                agent_future_full[t, 1] = agent_history[-1, 1] + agent_history[-1, 4] * dt
                agent_future_full[t, 2] = agent_history[-1, 2]

        # 速度使用历史最后帧
        agent_future_full[:, 3] = agent_history[-1, 3]
        agent_future_full[:, 4] = agent_history[-1, 4]

        return agent_future_full

    def _encode_scene(self, processed: dict) -> torch.Tensor:
        """编码场景特征"""
        with torch.no_grad():
            if self._use_gnn_encoder and processed.get('graph_data') is not None:
                graph = processed['graph_data'].to(self.device)
                scene_feat = self.scene_encoder.get_encoder_features(graph)
            else:
                ego_history   = torch.tensor(processed['ego_history'],   dtype=torch.float32).unsqueeze(0).to(self.device)
                agent_history = torch.tensor(processed['agent_history'], dtype=torch.float32).unsqueeze(0).to(self.device)
                scene_feat = self.scene_encoder(ego_history, agent_history)
        return scene_feat

    def _build_graph_from_history(
        self,
        ego_history: np.ndarray,
        agent_history: np.ndarray,
        boundary_agents: list,
    ):
        """Build GraphData from histories, matching DangerPairDataset._build_graph_data exactly."""
        from dataset import GraphData

        def _traj_to_polyline_features(traj: np.ndarray) -> np.ndarray:
            T = len(traj)
            feat = np.zeros((T, 9), dtype=np.float32)
            feat[:, :5] = traj
            prev = np.vstack([traj[:1], traj[:-1]])
            feat[:, 5] = traj[:, 0] - prev[:, 0]
            feat[:, 6] = traj[:, 1] - prev[:, 1]
            feat[:, 7] = traj[:, 2] - prev[:, 2]
            feat[:, 8] = 1.0
            return feat

        node_feats = [
            _traj_to_polyline_features(ego_history)[-1],
            _traj_to_polyline_features(agent_history)[-1],
        ]
        for ba in boundary_agents[:10]:
            hist = ba.get('agent_data')
            if hist is not None and len(hist) > 0:
                node_feats.append(_traj_to_polyline_features(np.array(hist))[-1])

        x = torch.tensor(np.stack(node_feats, axis=0), dtype=torch.float32)
        N = x.size(0)
        edges = [[i, j] for i in range(N) for j in range(N) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        ego_history_tensor = torch.tensor(ego_history[:, :2], dtype=torch.float32)
        return GraphData(x=x, gnn_x=x.clone(), gnn_edge_index=edge_index,
                         ego_history=ego_history_tensor)

    def _build_graph_data(self, polyline_features, traj_id_to_indices):
        """从 polyline 特征构建 PyG GraphData（与训练保持一致）。"""
        from dataset import GraphData
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
        if edges:
            gnn_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        ego_history_tensor = torch.zeros(40, 2, dtype=torch.float32)
        return GraphData(x=x, gnn_x=x.clone(), gnn_edge_index=gnn_edge_index, ego_history=ego_history_tensor)

    def _encode_condition(
        self,
        scene_feat: torch.Tensor,
        svo_angle: float,
        inherent_danger: float,
        danger_target: float,
        agent_state: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """编码条件向量"""
        svo_tensor = torch.tensor([[svo_angle]], dtype=torch.float32).to(self.device)
        inherent_tensor = torch.tensor([[inherent_danger]], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([[danger_target]], dtype=torch.float32).to(self.device)

        agent_state_tensor = None
        if agent_state is not None:
            agent_state_tensor = torch.tensor(
                agent_state.reshape(1, -1), dtype=torch.float32
            ).to(self.device)  # [1, 3]

        with torch.no_grad():
            condition = self.condition_encoder(
                scene_feat, svo_tensor, inherent_tensor, target_tensor,
                agent_state=agent_state_tensor
            )

        return condition

    def _recover_trajectories(
        self,
        increments: torch.Tensor,
        ego_last_pos: np.ndarray,
        ego_last_heading: float,
        ego_last_vx: float,
        ego_last_vy: float
    ) -> torch.Tensor:
        """恢复轨迹"""
        # 处理多样本情况
        if increments.dim() == 2 and increments.size(0) > 1:
            batch_size = increments.size(0)
            ego_last_pos_tensor = torch.tensor(ego_last_pos, dtype=torch.float32).to(self.device).unsqueeze(0).repeat(batch_size, 1)
            ego_last_heading_tensor = torch.tensor(ego_last_heading, dtype=torch.float32).to(self.device).unsqueeze(0).repeat(batch_size)
            ego_last_vx_tensor = torch.tensor(ego_last_vx, dtype=torch.float32).to(self.device).unsqueeze(0).repeat(batch_size)
            ego_last_vy_tensor = torch.tensor(ego_last_vy, dtype=torch.float32).to(self.device).unsqueeze(0).repeat(batch_size)
        else:
            ego_last_pos_tensor = torch.tensor(ego_last_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
            ego_last_heading_tensor = torch.tensor(ego_last_heading, dtype=torch.float32).unsqueeze(0).to(self.device)
            ego_last_vx_tensor = torch.tensor(ego_last_vx, dtype=torch.float32).unsqueeze(0).to(self.device)
            ego_last_vy_tensor = torch.tensor(ego_last_vy, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            trajectories = recover_trajectory_from_increment(
                increments, ego_last_pos_tensor, ego_last_heading_tensor,
                ego_last_vx_tensor, ego_last_vy_tensor, dt=0.1
            )

        return trajectories

    def _sdedit_with_guidance(
        self,
        gt_joint_inc: torch.Tensor,        # [B, 240] GT joint (Δx,Δy) increments
        condition: torch.Tensor,           # [B, 256]
        ego_history_t: torch.Tensor,       # [B, T, 5]
        agent_history_t: torch.Tensor,     # [B, T, 5]
        ego_last_pos_t: torch.Tensor,      # [B, 2]
        agent_last_pos_t: torch.Tensor,    # [B, 2]
        k: int = 30,
        guidance_scale: float = 0.1,
        max_grad_norm: float = 1.0,
    ) -> torch.Tensor:
        """SDEdit + danger gradient guidance for (Δx,Δy) parameterisation.

        Steps:
          1. Add noise up to step k (partial forward process, preserves GT structure).
          2. Denoise from k → 0; at each step inject ∇_x_t danger as guidance.

        Returns:
            [B, 240] generated joint increments
        """
        model = self.diffusion_model
        device = gt_joint_inc.device
        B = gt_joint_inc.size(0)

        k = min(k, model.config.num_diffusion_steps - 1)

        # ── 1. Partial forward noise ──────────────────────────────────────
        alpha_bar_k = model.alpha_bars[k].to(device)
        eps = torch.randn_like(gt_joint_inc)
        x_t = (alpha_bar_k.sqrt() * gt_joint_inc
                + (1.0 - alpha_bar_k).sqrt() * eps).detach()

        # ── 2. Guided denoising k → 0 ────────────────────────────────────
        for t in reversed(range(k)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            alpha_bar_t = model.alpha_bars[t].to(device)

            # Track gradient through x_t for danger guidance
            x_t = x_t.detach().requires_grad_(True)

            with torch.enable_grad():
                noise_pred = model.denoiser(x_t, condition, t_tensor)
                x_0_pred = (
                    (x_t - (1.0 - alpha_bar_t).sqrt() * noise_pred)
                    / alpha_bar_t.sqrt().clamp(min=1e-8)
                )
                x_0_pred = torch.nan_to_num(x_0_pred, nan=0.0, posinf=2.0, neginf=-2.0)

                # Decode predicted x_0 → trajectories for danger computation
                ego_pred_inc   = x_0_pred[:, :120]
                agent_pred_inc = x_0_pred[:, 120:]
                ego_fut   = xy_displacement_decode_torch(ego_pred_inc,   ego_last_pos_t)
                agent_fut = xy_displacement_decode_torch(agent_pred_inc, agent_last_pos_t)

                danger_result = compute_danger_score_torch(
                    ego_history_t, ego_fut,
                    agent_history_t, agent_fut,
                    self.danger_config_torch,
                )
                danger = danger_result['outcome_danger'].mean()

                grad = torch.autograd.grad(danger, x_t)[0]

            # Clip gradient norm
            grad_norm = grad.norm()
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / (grad_norm + 1e-8))

            # Normal DDPM posterior step
            with torch.no_grad():
                mean = (model.posterior_mean_coef1[t] * x_0_pred.detach()
                        + model.posterior_mean_coef2[t] * x_t.detach())
                if t > 0:
                    sigma = model.posterior_variance[t].sqrt()
                    x_next = mean + sigma * torch.randn_like(x_t)
                else:
                    x_next = mean

                # Inject danger guidance
                x_t = x_next + guidance_scale * grad.detach()

        return x_t.detach()

    def _compute_generated_danger(
        self,
        generated_trajectories: torch.Tensor,
        ego_history: np.ndarray,
        agent_history: np.ndarray,
        agent_future_gt: np.ndarray
    ) -> Dict:
        """计算生成轨迹的danger"""
        # 处理多样本情况
        if generated_trajectories.dim() == 3 and generated_trajectories.size(0) > 1:
            batch_size = generated_trajectories.size(0)
            ego_history_tensor = torch.tensor(ego_history, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            agent_history_tensor = torch.tensor(agent_history, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
            agent_future_tensor = torch.tensor(agent_future_gt, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device)
        else:
            ego_history_tensor = torch.tensor(ego_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            agent_history_tensor = torch.tensor(agent_history, dtype=torch.float32).unsqueeze(0).to(self.device)
            agent_future_tensor = torch.tensor(agent_future_gt, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 只使用前30帧（与GT对齐）
        generated_future = generated_trajectories[:, :30]

        with torch.no_grad():
            danger_result = compute_danger_score_torch(
                ego_history_tensor, generated_future,
                agent_history_tensor, agent_future_tensor[:, :30],
                self.danger_config_torch
            )

        return danger_result

    def _compute_generated_danger_joint(
        self,
        ego_trajectories: torch.Tensor,    # [N, 60, 5]
        agent_trajectories: torch.Tensor,  # [N, 60, 5]
        ego_history: np.ndarray,
        agent_history: np.ndarray,
    ) -> Dict:
        """计算联合生成轨迹的 danger（ego+agent 双车交互危险）。"""
        N = ego_trajectories.size(0)

        ego_hist   = torch.tensor(ego_history,   dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1).to(self.device)
        agent_hist = torch.tensor(agent_history, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1).to(self.device)

        ego_future   = ego_trajectories[:, :30]    # [N, 30, 5]
        agent_future = agent_trajectories[:, :30]  # [N, 30, 5]

        with torch.no_grad():
            danger_result = compute_danger_score_torch(
                ego_hist, ego_future,
                agent_hist, agent_future,
                self.danger_config_torch
            )

        return danger_result

    def _predict_boundary_agents(
        self,
        boundary_agents: List[Dict],
        ego_traj_np: np.ndarray,    # [60, 5]
        agent_traj_np: np.ndarray,  # [60, 5]
    ) -> Dict[int, np.ndarray]:
        """
        用 IDM 为每个背景 agent 预测未来轨迹，前车来自生成的危险双车轨迹。

        Returns:
            {agent_id: np.ndarray [60, 2]}  全局绝对坐标
        """
        if not boundary_agents:
            return {}

        dynamic_vehicles = [ego_traj_np, agent_traj_np]

        futures: Dict[int, np.ndarray] = {}
        for ba in boundary_agents:
            agent_id   = ba.get('agent_id', -1)
            agent_data = ba.get('agent_data')
            if agent_data is None or not isinstance(agent_data, np.ndarray):
                continue
            future = self.idm_predictor.predict_for_background_agent(
                agent_history=agent_data,
                lane_graph_data=None,        # 无车道图时退化为直线 IDM
                dynamic_vehicles=dynamic_vehicles,
            )
            futures[agent_id] = future       # [60, 2]

        return futures


# ============================================================
# 批量生成
# ============================================================

def batch_generate(
    generator: DangerSceneGenerator,
    data_dir: str,
    output_dir: str,
    svo_angle_range: tuple = (-45, 135),
    danger_margin_range: tuple = (1.0, 5.0),
    num_samples_per_scene: int = 3,
    max_scenes: int = None
):
    """
    批量生成危险场景

    Args:
        generator: 生成器实例
        data_dir: 输入数据目录
        output_dir: 输出目录
        svo_angle_range: SVO角度范围（用于随机采样）
        danger_margin_range: Danger超越幅度范围
        num_samples_per_scene: 每个场景生成的样本数
        max_scenes: 最大场景数（用于调试）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找pkl文件
    pkl_files = list(Path(data_dir).glob('danger_pair_features_*.pkl'))
    if max_scenes:
        pkl_files = pkl_files[:max_scenes]

    print(f"[Batch Generate] 处理 {len(pkl_files)} 个场景")

    generated_count = 0
    danger_exceeded_count = 0

    for pkl_file in tqdm(pkl_files, desc="Generating"):
        try:
            # 随机采样SVO角度和danger margin
            svo_angle = np.random.uniform(*svo_angle_range)

            # 加载原始数据获取GT danger
            raw_data = pd.read_pickle(pkl_file)
            processed = generator._process_raw_data(raw_data)
            gt_danger = processed['gt_outcome_danger']

            # 随机danger margin
            danger_margin = np.random.uniform(*danger_margin_range)
            danger_target = gt_danger + danger_margin

            # 生成
            result = generator.generate_from_pkl(
                str(pkl_file),
                svo_angle_override=svo_angle,
                danger_target_override=danger_target,
                num_samples=num_samples_per_scene
            )

            # 统计
            for sample in result['samples']:
                generated_count += 1
                if sample['danger_exceeded']:
                    danger_exceeded_count += 1

            # 保存生成结果
            save_generated_sample(result, output_path, pkl_file.stem)

        except Exception as e:
            print(f"Warning: Failed to generate from {pkl_file}: {e}")

    print(f"\n[Batch Generate] 完成")
    print(f"  生成样本数: {generated_count}")
    print(f"  Danger超越数: {danger_exceeded_count}")
    print(f"  超越比例: {danger_exceeded_count / generated_count:.2%}")

    return {
        'generated_count': generated_count,
        'danger_exceeded_count': danger_exceeded_count,
        'output_dir': str(output_path)
    }


def save_generated_sample(result: Dict, output_path: Path, original_name: str):
    """
    保存生成样本

    Args:
        result: 生成结果
        output_path: 输出目录
        original_name: 原始文件名
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for i, sample in enumerate(result['samples']):
        # 保存轨迹数据
        save_data = {
            'generated_increments': sample['generated_increments'],
            'generated_trajectory': sample['generated_trajectory'],
            'agent_trajectory': sample.get('agent_trajectory'),
            'background_agent_futures': sample.get('background_agent_futures', {}),
            'generated_danger': sample['generated_danger'],
            'gt_danger': sample['gt_danger'],
            'danger_exceeded': sample['danger_exceeded'],

            'svo_angle_used': sample['svo_angle_used'],
            'danger_target_used': sample['danger_target_used'],

            'scenario_name': sample['scenario_name'],
            'key_agent_id': sample['key_agent_id'],
            'generation_timestamp': timestamp
        }

        save_file = output_path / f'generated_{original_name}_sample{i}_{timestamp}.pkl'
        pd.to_pickle(save_data, save_file)


# ============================================================
# 单场景生成测试
# ============================================================

def test_single_generation(
    model_path: str,
    pkl_path: str,
    device: str = 'cuda',
    svo_angles: List[float] = [-30, 0, 45, 90],
    danger_margin: float = 3.0
):
    """
    测试单个场景的生成

    Args:
        model_path: 模型路径
        pkl_path: 数据路径
        device: 设备
        svo_angles: 测试的SVO角度列表
        danger_margin: danger超越幅度
    """
    print("=" * 60)
    print("单场景生成测试")
    print("=" * 60)

    # 创建生成器
    generator = DangerSceneGenerator(model_path, device)

    # 加载原始数据
    raw_data = pd.read_pickle(pkl_path)
    processed = generator._process_raw_data(raw_data)

    print(f"\n场景信息:")
    print(f"  Scenario: {processed['scenario_name']}")
    print(f"  Key Agent ID: {processed['key_agent_id']}")
    print(f"  GT Danger: {processed['gt_outcome_danger']:.2f}")
    print(f"  Original SVO: {processed['svo_angle']:.1f}°")

    print(f"\n测试不同SVO角度的生成:")
    print("-" * 60)

    results_table = []

    for svo_angle in svo_angles:
        danger_target = processed['gt_outcome_danger'] + danger_margin

        result = generator.generate_from_pkl(
            pkl_path,
            svo_angle_override=svo_angle,
            danger_target_override=danger_target,
            num_samples=1
        )

        sample = result['samples'][0]

        results_table.append({
            'svo_angle': svo_angle,
            'danger_target': danger_target,
            'generated_danger': sample['generated_danger'],
            'gt_danger': sample['gt_danger'],
            'exceeded': sample['danger_exceeded']
        })

        print(f"  SVO={svo_angle:6.1f}°: Generated={sample['generated_danger']:6.2f}, "
              f"GT={sample['gt_danger']:6.2f}, Exceeded={sample['danger_exceeded']}")

    print("-" * 60)

    return results_table


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='生成危险场景')

    parser.add_argument('--model_path', type=str,
                        default=os.path.join(config.DIFFUSION_TRAINING_EXPERIMENTS_DIR, 'full_training', 'best_model.pt'),
                        help='训练好的模型路径')

    parser.add_argument('--data_dir', type=str,
                        default=config.DANGER_PAIR_SCENARIOS_DIR,
                        help='输入数据目录')

    parser.add_argument('--output_dir', type=str,
                        default=config.GENERATED_DANGER_SCENES_DIR,
                        help='输出目录')

    parser.add_argument('--mode', type=str, default='batch',
                        choices=['single', 'batch'],
                        help='生成模式 (single=单场景测试, batch=批量生成)')

    parser.add_argument('--pkl_path', type=str, default=None,
                        help='单场景测试时使用的pkl文件路径')

    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    parser.add_argument('--svo_angles', type=float, nargs='+',
                        default=[-30, 0, 45, 90],
                        help='测试的SVO角度列表')

    parser.add_argument('--danger_margin', type=float, default=3.0,
                        help='Danger超越幅度')

    parser.add_argument('--num_samples', type=int, default=3,
                        help='每个场景生成的样本数')

    parser.add_argument('--max_scenes', type=int, default=None,
                        help='最大场景数（用于调试）')

    args = parser.parse_args()

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA不可用，使用CPU")
        args.device = 'cpu'

    if args.mode == 'single':
        # 单场景测试
        if args.pkl_path is None:
            # 使用第一个pkl文件
            pkl_files = list(Path(args.data_dir).glob('danger_pair_features_*.pkl'))
            if pkl_files:
                args.pkl_path = str(pkl_files[0])
            else:
                print("Error: 未找到pkl文件")
                return

        results = test_single_generation(
            args.model_path, args.pkl_path, args.device,
            args.svo_angles, args.danger_margin
        )

    else:
        # 批量生成
        generator = DangerSceneGenerator(args.model_path, args.device)

        result = batch_generate(
            generator, args.data_dir, args.output_dir,
            svo_angle_range=(-45, 135),
            danger_margin_range=(1.0, 5.0),
            num_samples_per_scene=args.num_samples,
            max_scenes=args.max_scenes
        )

        print("\n批量生成完成:")
        print(f"  生成样本数: {result['generated_count']}")
        print(f"  Danger超越数: {result['danger_exceeded_count']}")
        print(f"  输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
