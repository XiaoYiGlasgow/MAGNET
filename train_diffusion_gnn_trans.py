"""
整合版Diffusion危险场景生成器训练脚本

使用GNN+Transformer作为场景编码器（替代SimpleSceneEncoder）

架构：
1. TransformerPredictor (GNN+Transformer Encoder) -> 384维场景特征
2. SceneConditionEncoder -> 256维条件向量
3. DangerSceneDiffusion -> 生成的轨迹

数据来源：
- processed_data/features_*.pkl: POLYLINE_FEATURES用于GNN+Transformer输入
- danger_pair_scenarios/danger_pair_*.pkl: GT轨迹和危险指标用于Diffusion训练

使用方法：
python train_diffusion_gnn_trans.py --epochs 50 --batch_size 4
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
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import config

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'models'))
sys.path.insert(0, os.path.join(current_dir, 'utils'))
sys.path.insert(0, os.path.join(current_dir, 'src'))

# 添加models目录（TransformerPredictor）
models_path = '/mnt/local/Xiao_Yi/nuplan-devkit/models'
if models_path not in sys.path:
    sys.path.insert(0, models_path)

# 导入模块
from scene_condition_encoder import SceneConditionEncoder, ConditionEncoderConfig, AgentPredictionHead
from danger_diffusion import DangerSceneDiffusion, DiffusionConfig, LaneEncoder
from dataset import GraphData

# 导入TransformerPredictor
try:
    from transformer_predictor import TransformerPredictor
    print("[SUCCESS] TransformerPredictor导入成功")
except ImportError as e:
    print(f"[ERROR] TransformerPredictor导入失败: {e}")
    sys.exit(1)

# 导入danger_score（用于计算SVO和danger）
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from danger_score import DangerScoreConfig, compute_danger_score, compute_pair_svo

# 导入danger_score_torch（用于训练时的danger loss计算）
from danger_score_torch import (
    DangerExceedLoss, DangerScoreConfigTorch,
)


# ============================================================
# 配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    processed_data_dir: str = config.DIFFUSION_PROCESSED_DATA_DIR
    danger_pair_dir: str = config.NEW_DANGER_PAIR_DIR
    output_dir: str = config.DIFFUSION_TRAINING_GNN_TRANS_DIR

    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-4
    device: str = 'cuda'

    max_samples: int = None
    save_every: int = 10

    # 早停：对每 epoch loss 取近 early_stop_window 个的滑动均值（抵抗 diffusion 单 epoch 噪声），
    # 连续 early_stop_patience 个 epoch 滑动均值无改善(> min_delta)则停。patience<=0 关闭（跑满 epochs）。
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_window: int = 5

    # 时间窗口配置（需要匹配数据处理时的配置）
    num_past_poses: int = 40  # 4秒历史
    num_future_poses: int = 60  # 6秒未来

    # [v6.0] 六C：运动学参数化重训练
    use_kinematic_v6_0: bool = False  # True = 用 (Δv,Δθ) 空间训练（需与 eval 侧解码对齐）

    # Map-conditioned DDPM：lane conditioning（方案 B）
    use_map_conditioning: bool = True  # False = baseline 消融（无地图条件）

    # v9：排除 agent 未来不满 6s 的样本（修复 target 污染，默认 off = v8 行为）
    exclude_truncated_agent: bool = False
    # v9：用 agent_future[0] 重构 agent present（修 A1 时间方向 bug，默认 off）
    fix_agent_alignment: bool = False
    # v9 阶段2 B1：condition 加入 key agent 的 xy 位置（默认 off = 阶段1 行为）
    use_agent_xy: bool = False

    # v10 CFG：属性 classifier-free guidance。danger_dropout_enable=True 让 encoder 创建
    # null-danger token；danger_dropout 是训练时逐样本把 danger 置 null 的概率（典型 0.1~0.2）。
    # Phase 0 证 v9 danger 条件被忽略（conditioning collapse），靠 dropout 强制模型学到
    # cond/uncond 真差异，推理 generate_cfg 放大。默认 off = v9 行为（不影响老路径）。
    danger_dropout_enable: bool = False
    danger_dropout: float = 0.0


# ============================================================
# DDP 辅助
# ============================================================

class _GNNWrapper(nn.Module):
    """将 TransformerPredictor 包裹成标准 forward()，使 DDP 可正常 hook。

    Returns (scene_feat [B,384], ego_pred [B,120]):
      - scene_feat: 编码器特征，供 SceneConditionEncoder 和 AgentPredictionHead 使用
      - ego_pred:   GNN+Trans 预训练的 ego 未来轨迹预测（Δx/Δy，60步）
    编码器只跑一次：get_encoder_features() 得到 384-dim，再接 feature_fusion + output_head。
    """
    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(self, batched_graph):
        scene_feat = self.predictor.get_encoder_features(batched_graph)   # [B, 384]
        fused      = self.predictor.feature_fusion(scene_feat)            # [B, 128]
        ego_pred   = self.predictor.output_head(fused)                    # [B, 120]
        return scene_feat, ego_pred


# ============================================================
# 数据集工具函数（模块级，供并行 worker 调用）
# ============================================================

_CACHE_VERSION = 4  # 修改数据处理逻辑时递增，自动使旧缓存失效（v4: 边惰性重算，cache 不存 gnn_edge_index）

# Lane polyline extraction
# 注意：POLYLINE_FEATURES col-9 是 subgraph_id（非 type），须经 SUBGRAPH_ID_TO_TYPE 查表
# 才能定位 lane 行；每场景 lane 行固定 900（60 lanes × 15 segs），含全零 padding
MAX_LANE_SEGS    = 900   # padding/容量上限，与 features pkl 固定结构对齐
SUBGRAPH_ID_COL  = 9     # subgraph_id column index in POLYLINE_FEATURES


def _build_gnn_edge_index(positions: np.ndarray) -> torch.Tensor:
    """radius-30m 无向边（照搬原 _build_graph_data_fn 逻辑，逐字节一致）。

    惰性重算用：cache 不再存 gnn_edge_index（单样本实测 ~84MB），改在 __getitem__ 里
    用缓存的节点位置 positions=graph.x[:, :2] 现算。这样全量样本可放 RAM/磁盘。
    """
    num_nodes = positions.shape[0]
    if num_nodes <= 1:
        return torch.tensor([[0], [0]], dtype=torch.long)
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 2)
    dists = np.linalg.norm(diff, axis=-1)                  # (N, N)
    i_idx, j_idx = np.where(
        (dists <= 30.0) & (np.arange(num_nodes)[:, None] < np.arange(num_nodes)[None, :])
    )
    if len(i_idx) == 0:
        return torch.tensor([[0], [0]], dtype=torch.long)
    edges = np.concatenate([
        np.stack([i_idx, j_idx], axis=1),
        np.stack([j_idx, i_idx], axis=1),
    ])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _build_graph_data_fn(polyline_features, traj_id_to_indices, lane_id_to_indices, subgraph_id_to_type):
    """构建 GraphData（**不含 gnn_edge_index**，节省 cache/RAM）。

    gnn_edge_index 由节点位置确定性重建 → 不入 data_list/cache（单样本 84MB→~150KB）。
    在 __getitem__ 里用 _build_gnn_edge_index(graph.x[:, :2]) 现算（与旧实现一致）。
    """
    x = torch.tensor(polyline_features[:, :9], dtype=torch.float32)

    gnn_x = x.clone()

    if 0 in traj_id_to_indices:
        start, end = traj_id_to_indices[0]
        ego_poly = polyline_features[start:end + 1, :2]
        if len(ego_poly) >= 40:
            ego_hist = torch.tensor(ego_poly[:40], dtype=torch.float32)
        else:
            pad = np.zeros((40 - len(ego_poly), 2))
            ego_hist = torch.tensor(np.concatenate([ego_poly, pad]), dtype=torch.float32)
    else:
        ego_hist = torch.zeros(40, 2, dtype=torch.float32)

    # gnn_edge_index 留空：由 __getitem__ 用 _build_gnn_edge_index 现算（不入 cache）
    return GraphData(x=x, gnn_x=gnn_x, gnn_edge_index=None, ego_history=ego_hist)


def _compute_increments_fn(traj: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
    increments = np.zeros((traj.shape[0], 2))
    increments[0] = traj[0, :2] - start_pos
    increments[1:] = traj[1:, :2] - traj[:-1, :2]
    return increments.flatten()


def _offroad_fraction_np(xy: np.ndarray, lane_segs: np.ndarray, half_width: float) -> float:
    """轨迹点落在所有车道 half_width 之外的比例；无有效车道返回 nan。

    ⚠️ 口径与 benchmark_filter._offroad_fraction 一致（投影到最近车道线段、min-over-segs
    不罚变道）。改这里记得同步那边。
    """
    valid = np.abs(lane_segs).sum(axis=-1) > 0
    segs = lane_segs[valid]
    if len(segs) == 0 or len(xy) == 0:
        return float("nan")
    p = xy[:, None, :]; a = segs[None, :, :2]; b = segs[None, :, 2:]
    ab = b - a; ap = p - a
    t = np.clip((ap * ab).sum(-1) / (ab * ab).sum(-1).clip(min=1e-8), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return float((np.linalg.norm(p - proj, axis=-1).min(axis=1) > half_width).mean())


def _is_offroad_keyagent(ego_fut_xy: np.ndarray, agent_fut_xy: np.ndarray, lane_segs: np.ndarray,
                         half_width: float = 3.0, max_agent_offroad: float = 0.3,
                         ego_on_thr: float = 0.2) -> bool:
    """数据层离道 key agent 判据（与 benchmark 种子过滤口径一致）：
    ego GT 在道（本地车道覆盖可信）但 agent GT 大段离道 → True（应 skip，模型不学离道车）。
    无车道覆盖(nan) 或 ego 也离道（覆盖差/不可信）时 → False（放行，不赌）。
    """
    lane = np.asarray(lane_segs)
    ego_off = _offroad_fraction_np(np.asarray(ego_fut_xy)[:, :2], lane, half_width)
    agent_off = _offroad_fraction_np(np.asarray(agent_fut_xy)[:, :2], lane, half_width)
    if np.isnan(ego_off) or np.isnan(agent_off):
        return False
    return (ego_off < ego_on_thr) and (agent_off > max_agent_offroad)


def _process_agent_future_fn(agent_future: np.ndarray, agent_history: np.ndarray) -> np.ndarray:
    n = len(agent_future)
    af2d = agent_future[:60, :2] if n >= 60 else np.concatenate([agent_future[:, :2], np.zeros((60 - n, 2))])
    out = np.zeros((60, 5))
    out[:, :2] = af2d
    out[:, 2] = agent_history[-1, 2]
    out[:, 3] = agent_history[-1, 3]
    out[:, 4] = agent_history[-1, 4]
    return out


def _extrapolate_agent_fn(agent_history: np.ndarray, target_len: int) -> np.ndarray:
    lp, lh, lvx, lvy = agent_history[-1, :2], agent_history[-1, 2], agent_history[-1, 3], agent_history[-1, 4]
    t = np.arange(1, target_len + 1) * 0.1
    out = np.zeros((target_len, 5))
    out[:, 0] = lp[0] + lvx * t
    out[:, 1] = lp[1] + lvy * t
    out[:, 2] = lh
    out[:, 3] = lvx
    out[:, 4] = lvy
    return out


def _agent_future_truncated(af_raw) -> bool:
    """原始 key_agent_future 尾部是否含 (0,0) padding（agent 未来不足 6s）。

    danger_pair pkl 里 key_agent_future 为 (60,5)，agent 提前驶离/丢 track 时
    尾部帧被补 (0,0)（绝对坐标原点，物理上不可能停在 ego 当前位置）。任一尾部
    (0,0) 帧即视为截断。无 agent 未来也视为不可用。
    """
    if af_raw is None:
        return True
    af = np.asarray(af_raw)
    if af.ndim != 2 or af.shape[0] < 60:
        return True
    xy = af[:60, :2]
    nonzero = np.where(np.abs(xy).sum(axis=1) != 0)[0]
    if len(nonzero) == 0:
        return True
    trailing_zeros = 60 - 1 - int(nonzero[-1])
    return trailing_zeros > 0


def _realign_agent_present(agent_history: np.ndarray, agent_future_full: np.ndarray) -> np.ndarray:
    """A1 修复：源预处理里 agent history 取自 present 之后（`get_tracked_objects_at_iteration(0..19)`
    时间方向 bug），故 `agent_history[-1]` 实为 present+1.9s。但 `agent_future[0]` 才是真正的
    present（与 ego 对齐）。用 future[0] + 首段速度重构 agent present 帧，使 agent 与 ego 时间
    对齐、`agent_inc[0]=0`（与 ego 对称），并让 condition 的 agent present 状态取自正确时刻。
    返回 copy，不改原数组。"""
    af = np.asarray(agent_future_full)
    new = np.array(agent_history, dtype=np.float32).copy()
    d = af[1, :2] - af[0, :2]
    new[-1, 0] = af[0, 0]
    new[-1, 1] = af[0, 1]
    new[-1, 2] = float(np.arctan2(d[1], d[0]))
    new[-1, 3] = d[0] / 0.1
    new[-1, 4] = d[1] / 0.1
    return new


def _load_single_sample(args: tuple) -> Optional[dict]:
    """并行 worker：加载并处理单个 danger_pair 样本。

    args 第 3 个元素（可选）exclude_truncated：True 时跳过 agent 未来不满 6s 的样本，
    避免被 padding 污染的 agent target 进入 joint_increments（recon_loss 无 mask）。
    args 第 4 个元素（可选）fix_agent_alignment：True 时用 agent_future[0] 重构 agent present
    （修 A1 时间方向 bug，详见 _realign_agent_present）。
    """
    danger_pair_file_str, processed_data_dir_str = args[0], args[1]
    exclude_truncated = args[2] if len(args) > 2 else False
    fix_agent_alignment = args[3] if len(args) > 3 else False
    danger_pair_file = Path(danger_pair_file_str)
    processed_data_dir = Path(processed_data_dir_str)

    stem = danger_pair_file.stem
    if stem.startswith('danger_pair_features_'):
        scenario_name = stem.replace('danger_pair_features_', '').split('_agent')[0]
    else:
        scenario_name = stem.replace('danger_pair_', '').split('_agent')[0]

    processed_file = processed_data_dir / f"features_{scenario_name}.pkl"
    if not processed_file.exists():
        return None

    try:
        danger_pair_data = pd.read_pickle(danger_pair_file)
        processed_data = pd.read_pickle(processed_file)

        pf = processed_data.iloc[0]['POLYLINE_FEATURES']
        graph_data = _build_graph_data_fn(
            pf,
            processed_data.iloc[0]['TRAJ_ID_TO_INDICES'],
            processed_data.iloc[0].get('LANE_ID_TO_INDICES', {}),
            processed_data.iloc[0].get('SUBGRAPH_ID_TO_TYPE', {}),
        )

        # Extract lane centerline segments, ego-centric.
        # col-9 是 subgraph_id，需经 SUBGRAPH_ID_TO_TYPE 查表筛 type=='lane' 的行。
        # col 0:4 = [x_start, y_start, x_end, y_end]（相邻行首尾相连）。
        _sid_to_type = processed_data.iloc[0].get('SUBGRAPH_ID_TO_TYPE', {})
        _lane_out = np.zeros((MAX_LANE_SEGS, 4), dtype=np.float32)
        _sids = pf[:, SUBGRAPH_ID_COL].astype(int)
        _lane_mask = np.array([_sid_to_type.get(s, '') == 'lane' for s in _sids], dtype=bool)
        _lane_segs = pf[_lane_mask, :4].astype(np.float32)
        _n = min(len(_lane_segs), MAX_LANE_SEGS)
        if _n > 0:
            _lane_out[:_n] = _lane_segs[:_n]
        lane_polylines = _lane_out

        ego_history = danger_pair_data['ego_history']
        agent_history = danger_pair_data['key_agent_history']
        ego_future_gt = np.array(danger_pair_data['ego_future_gt'])

        # 补全 ego_future_gt 为 5 列
        if ego_future_gt.ndim == 2 and ego_future_gt.shape[1] == 2:
            ego_xy = np.vstack([ego_history[-1:, :2], ego_future_gt[:, :2]])
            dxy = np.diff(ego_xy, axis=0)
            tmp = np.zeros((ego_future_gt.shape[0], 5))
            tmp[:, :2] = ego_future_gt
            tmp[:, 2] = np.arctan2(dxy[:, 1], dxy[:, 0])
            tmp[:, 3] = dxy[:, 0] / 0.1
            tmp[:, 4] = dxy[:, 1] / 0.1
            ego_future_gt = tmp

        # 对向来车过滤
        rh = abs(float(agent_history[-1, 2])) % (2 * np.pi)
        if rh > np.pi:
            rh = 2 * np.pi - rh
        if rh > np.deg2rad(150.0) and float(agent_history[-1, 0]) > 3.0:
            return None

        ego_last_pos = ego_history[-1, :2]
        gt_increments = _compute_increments_fn(ego_future_gt[:, :2], ego_last_pos)

        af_raw = danger_pair_data.get('key_agent_future')
        # 排除 agent 未来不满 6s 的样本（target 污染源），与 _is_offroad_keyagent 同款 skip
        if exclude_truncated and _agent_future_truncated(af_raw):
            return None
        agent_future_full = (
            _process_agent_future_fn(af_raw, agent_history)
            if af_raw is not None
            else _extrapolate_agent_fn(agent_history, 60)
        )

        # A1 修复：用 agent_future[0] 重构 agent present（修源预处理 agent history 时间方向 bug）
        if fix_agent_alignment and af_raw is not None:
            agent_history = _realign_agent_present(agent_history, agent_future_full)

        # 数据层离道 key agent 过滤：ego 在道但 agent GT 离道 → skip（模型不学离道车）
        if _is_offroad_keyagent(ego_future_gt, agent_future_full, lane_polylines):
            return None

        # fix_agent_alignment 时 agent_history[-1]=future[0] → inc[0]=0（与 ego 对称）
        agent_inc_flat = _compute_increments_fn(agent_future_full[:, :2], agent_history[-1, :2])
        joint_increments = np.concatenate([gt_increments, agent_inc_flat])

        danger_config = DangerScoreConfig()
        svo_info = compute_pair_svo(ego_history, ego_future_gt, agent_history, agent_future_full, danger_config)
        danger_result = compute_danger_score(ego_history, ego_future_gt, agent_history, agent_future_full, danger_config)

        gt_outcome_danger = danger_result['outcome_danger']['total_score']

        return {
            'graph_data': graph_data,
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future_gt': ego_future_gt,
            'agent_future_gt': agent_future_full,
            'gt_increments': gt_increments,
            'joint_increments': joint_increments,
            'ego_last_pos': ego_last_pos,
            'ego_last_heading': float(ego_history[-1, 2]),
            'ego_last_vx': float(ego_history[-1, 3]),
            'ego_last_vy': float(ego_history[-1, 4]),
            'svo_angle': svo_info['ego_svo_angle_deg'],
            'inherent_danger': danger_result['inherent_danger']['total_score'],
            'gt_outcome_danger': gt_outcome_danger,
            'danger_target': gt_outcome_danger + 2.0,
            'scenario_name': scenario_name,
            'key_agent_id': danger_pair_data.get('key_agent_id', -1),
            'lane_polylines': lane_polylines,
        }
    except Exception:
        return None


def _load_single_sample_kinematic_v6_0(args: tuple) -> Optional[dict]:
    """[v6.0] 六C: 同 _load_single_sample，但将 gt/joint increments 编码为 (Δv,Δθ) 空间。

    直接包装 _load_single_sample，在返回 dict 前替换增量编码。
    """
    result = _load_single_sample(args)
    if result is None:
        return None

    from src.shared.utils.kinematic_v6_0 import (
        encode_ego_kinematic_v6_0,
        encode_agent_kinematic_v6_0,
    )

    gt_kin   = encode_ego_kinematic_v6_0(result['ego_future_gt'], result['ego_history'])
    agent_kin = encode_agent_kinematic_v6_0(result['agent_future_gt'], result['agent_history'])
    result['gt_increments']    = gt_kin
    result['joint_increments'] = np.concatenate([gt_kin, agent_kin])
    return result


# ============================================================
# 数据集
# ============================================================

class IntegratedDiffusionDataset(Dataset):
    """
    整合数据集 - 同时加载processed_data和danger_pair数据

    processed_data: POLYLINE_FEATURES用于GNN+Transformer输入
    danger_pair: GT轨迹和危险指标用于Diffusion训练
    """

    def __init__(
        self,
        processed_data_dir: str,
        danger_pair_dir: str,
        max_samples: int = None,
        danger_config: DangerScoreConfig = None,
        num_workers: int = 8,
        cache_file: str = None,
        use_kinematic_v6_0: bool = False,   # [v6.0] 六C
        exclude_truncated_agent: bool = False,  # v9: 跳过 agent 未来不满 6s 的样本
        fix_agent_alignment: bool = False,  # v9: 用 future[0] 重构 agent present（修 A1）
    ):
        self.processed_data_dir = Path(processed_data_dir)
        self.danger_pair_dir = Path(danger_pair_dir)
        self.danger_config = danger_config if danger_config else DangerScoreConfig()
        self.use_kinematic_v6_0 = use_kinematic_v6_0  # [v6.0]
        self.exclude_truncated_agent = exclude_truncated_agent  # v9
        self.fix_agent_alignment = fix_agent_alignment  # v9

        self.danger_pair_files = sorted(self.danger_pair_dir.glob('danger_pair_*.pkl'))
        if max_samples:
            self.danger_pair_files = self.danger_pair_files[:max_samples]

        print(f"[IntegratedDiffusionDataset] 找到 {len(self.danger_pair_files)} 个danger_pair文件")

        # ── 缓存命中：直接加载 ──────────────────────────────────
        _suffix = ("_excltrunc" if exclude_truncated_agent else "") + \
                  ("_fixalign" if fix_agent_alignment else "")
        if cache_file:
            cache_path = Path(cache_file)
        elif max_samples:
            cache_path = self.danger_pair_dir / f"dataset_cache_{max_samples}{_suffix}.pt"
        else:
            cache_path = self.danger_pair_dir / f"dataset_cache{_suffix}.pt"
        if self._cache_is_valid(cache_path, self.danger_pair_files, max_samples):
            print(f"[IntegratedDiffusionDataset] 缓存命中，从 {cache_path.name} 加载 ...")
            cached = torch.load(cache_path, weights_only=False)
            self.data_list = cached['data_list']
            print(f"[IntegratedDiffusionDataset] 缓存加载完成，{len(self.data_list)} 个样本（跳过重新计算）")
            return

        # ── 并行构建 ────────────────────────────────────────────
        # [v6.0] 六C: kinematic encoding requires a different loader function
        load_fn = (_load_single_sample_kinematic_v6_0
                   if use_kinematic_v6_0 else _load_single_sample)
        if use_kinematic_v6_0:
            print("[IntegratedDiffusionDataset] [v6.0 六C] 使用运动学参数化编码 (Δv,Δθ)")

        n_workers = min(num_workers, len(self.danger_pair_files))
        print(f"[IntegratedDiffusionDataset] 并行加载，num_workers={n_workers} ...")
        args_list = [(str(f), str(self.processed_data_dir), self.exclude_truncated_agent,
                      self.fix_agent_alignment)
                     for f in self.danger_pair_files]

        self.data_list = []
        failed_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for data in tqdm(
                executor.map(load_fn, args_list),
                total=len(args_list),
                desc="Loading data",
            ):
                if data is not None:
                    self.data_list.append(data)
                else:
                    failed_count += 1

        print(f"[IntegratedDiffusionDataset] 成功加载 {len(self.data_list)} 个样本，跳过 {failed_count} 个")

        # ── 保存缓存 ────────────────────────────────────────────
        print(f"[IntegratedDiffusionDataset] 保存缓存到 {cache_path.name} ...")
        torch.save({
            'data_list': self.data_list,
            'version': _CACHE_VERSION,
            'num_files': len(self.danger_pair_files),
            'max_samples': max_samples,
        }, cache_path)
        print(f"[IntegratedDiffusionDataset] 缓存保存完成")

    @staticmethod
    def _cache_is_valid(cache_path: Path, danger_pair_files: list, max_samples) -> bool:
        if not cache_path.exists():
            return False
        try:
            meta = torch.load(cache_path, weights_only=False)
            if meta.get('version') != _CACHE_VERSION:
                print(f"[IntegratedDiffusionDataset] 缓存版本不匹配，重新构建")
                return False
            if meta.get('num_files') != len(danger_pair_files):
                return False
            if meta.get('max_samples') != max_samples:
                return False
        except Exception:
            return False
        cache_mtime = cache_path.stat().st_mtime
        for f in danger_pair_files:
            if f.stat().st_mtime > cache_mtime:
                print(f"[IntegratedDiffusionDataset] 源文件已更新，重新构建缓存")
                return False
        return True

    def _load_and_process_data(self, danger_pair_file: Path) -> Optional[dict]:
        """
        加载并处理单个样本

        同时读取processed_data和danger_pair数据
        """
        # 从danger_pair文件名提取scenario名称（兼容两种命名格式）
        # danger_pair_features_{name}_agent{id}.pkl 或 danger_pair_{name}_agent{id}.pkl
        stem = danger_pair_file.stem
        if stem.startswith('danger_pair_features_'):
            scenario_name = stem.replace('danger_pair_features_', '').split('_agent')[0]
        else:
            scenario_name = stem.replace('danger_pair_', '').split('_agent')[0]
        processed_file = self.processed_data_dir / f"features_{scenario_name}.pkl"

        # 检查processed_data文件是否存在
        if not processed_file.exists():
            return None

        # 加载两个数据文件
        danger_pair_data = pd.read_pickle(danger_pair_file)
        processed_data = pd.read_pickle(processed_file)

        # 提取processed_data中的POLYLINE_FEATURES
        polyline_features = processed_data.iloc[0]['POLYLINE_FEATURES']  # (N, 12)
        traj_id_to_indices = processed_data.iloc[0]['TRAJ_ID_TO_INDICES']
        lane_id_to_indices = processed_data.iloc[0].get('LANE_ID_TO_INDICES', {})
        subgraph_id_to_type = processed_data.iloc[0].get('SUBGRAPH_ID_TO_TYPE', {})

        # 构建图数据
        graph_data = self._build_graph_data(
            polyline_features,
            traj_id_to_indices,
            lane_id_to_indices,
            subgraph_id_to_type
        )

        # 提取danger_pair中的轨迹数据
        ego_history = danger_pair_data['ego_history']  # (40, 5)
        agent_history = danger_pair_data['key_agent_history']  # (40, 5)
        ego_future_gt = np.array(danger_pair_data['ego_future_gt'])  # may be (60,2) or (60,5)

        # 补全ego_future_gt为5列（部分旧版pkl只存了xy）
        # 用轨迹差分推导 heading/vx/vy，比直接用历史末帧常数值更准确
        if ego_future_gt.ndim == 2 and ego_future_gt.shape[1] == 2:
            # 拼接历史末帧 xy 作为差分起点
            ego_xy = np.vstack([ego_history[-1:, :2], ego_future_gt[:, :2]])  # (61, 2)
            dxy    = np.diff(ego_xy, axis=0)                                  # (60, 2)
            headings = np.arctan2(dxy[:, 1], dxy[:, 0])                      # (60,)
            vx = dxy[:, 0] / 0.1                                              # dt=0.1s
            vy = dxy[:, 1] / 0.1
            tmp = np.zeros((ego_future_gt.shape[0], 5))
            tmp[:, :2] = ego_future_gt
            tmp[:, 2]  = headings
            tmp[:, 3]  = vx
            tmp[:, 4]  = vy
            ego_future_gt = tmp

        # 对向车道过滤：ego-centric frame中ego heading≈0，对向来车heading≈π
        agent_heading_last = float(agent_history[-1, 2])
        rel_heading_abs = abs(agent_heading_last) % (2 * np.pi)
        if rel_heading_abs > np.pi:
            rel_heading_abs = 2 * np.pi - rel_heading_abs
        agent_x = float(agent_history[-1, 0])
        if rel_heading_abs > np.deg2rad(150.0) and agent_x > 3.0:
            return None  # skip opposite-lane oncoming agent

        # 计算GT轨迹增量（ego, 120-dim）
        ego_last_pos = ego_history[-1, :2]
        ego_future_2d = ego_future_gt[:, :2]  # (60, 2)
        gt_increments = self._compute_increments(ego_future_2d, ego_last_pos)

        # 处理agent未来轨迹
        agent_future = danger_pair_data.get('key_agent_future')
        if agent_future is not None:
            agent_future_full = self._process_agent_future(agent_future, agent_history)
        else:
            agent_future_full = self._extrapolate_agent(agent_history, 60)

        # 计算joint增量（240-dim = ego_120 ‖ agent_120）
        agent_last_pos = agent_history[-1, :2]
        agent_inc_flat = self._compute_increments(agent_future_full[:, :2], agent_last_pos)
        joint_increments = np.concatenate([gt_increments, agent_inc_flat])  # (240,)

        # 计算SVO和danger指标
        svo_info = compute_pair_svo(
            ego_history, ego_future_gt,
            agent_history, agent_future_full,
            self.danger_config
        )

        danger_result = compute_danger_score(
            ego_history, ego_future_gt,
            agent_history, agent_future_full,
            self.danger_config
        )

        inherent_danger = danger_result['inherent_danger']['total_score']
        gt_outcome_danger = danger_result['outcome_danger']['total_score']

        # 目标danger（比GT更高）
        danger_margin = 2.0
        danger_target = gt_outcome_danger + danger_margin

        return {
            # 图数据（用于GNN+Transformer）
            'graph_data': graph_data,

            # 轨迹数据
            'ego_history': ego_history,
            'agent_history': agent_history,
            'ego_future_gt': ego_future_gt,
            'agent_future_gt': agent_future_full,
            'gt_increments': gt_increments,        # (120,) ego-only, kept for compat
            'joint_increments': joint_increments,  # (240,) ego+agent, used as diffusion x_0

            # 最后帧状态
            'ego_last_pos': ego_history[-1, :2],
            'ego_last_heading': ego_history[-1, 2],
            'ego_last_vx': ego_history[-1, 3],
            'ego_last_vy': ego_history[-1, 4],

            # danger指标
            'svo_angle': svo_info['ego_svo_angle_deg'],
            'inherent_danger': inherent_danger,
            'gt_outcome_danger': gt_outcome_danger,
            'danger_target': danger_target,

            # 元数据
            'scenario_name': scenario_name,
            'key_agent_id': danger_pair_data.get('key_agent_id', -1)
        }

    def _build_graph_data(
        self,
        polyline_features: np.ndarray,
        traj_id_to_indices: dict,
        lane_id_to_indices: dict,
        subgraph_id_to_type: dict
    ) -> GraphData:
        """
        构建GraphData对象用于TransformerPredictor

        Args:
            polyline_features: (N, 12) 特征向量
            traj_id_to_indices: 轨迹ID到索引映射
            lane_id_to_indices: 车道ID到索引映射
            subgraph_id_to_type: 子图ID到类型映射

        Returns:
            GraphData对象
        """
        # 创建节点特征 (N, 9) - 取前9维用于TransformerPredictor
        x = torch.tensor(polyline_features[:, :9], dtype=torch.float32)

        # 创建边索引（使用简单的全连接图）
        num_nodes = x.size(0)

        # GNN边索引：基于距离的连接
        if num_nodes > 1:
            # 计算节点位置
            positions = polyline_features[:, :2]  # (N, 2) x, y

            # 创建距离阈值内的边
            distance_threshold = 30.0
            edges = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= distance_threshold:
                        edges.append([i, j])
                        edges.append([j, i])  # 双向边

            if len(edges) > 0:
                gnn_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                # 至少创建一个自环
                gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        else:
            gnn_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # GNN节点特征
        gnn_x = x.clone()

        # 创建ego_history (40, 2) - 取位置信息
        # 从ID=0的轨迹提取ego历史
        if 0 in traj_id_to_indices:
            start, end = traj_id_to_indices[0]
            ego_polyline = polyline_features[start:end+1]
            # 提取位置 (x, y)
            ego_history_positions = ego_polyline[:, :2]
            # 确保是40帧
            if len(ego_history_positions) >= 40:
                ego_history_tensor = torch.tensor(ego_history_positions[:40], dtype=torch.float32)
            else:
                # 填充到40帧
                padding = np.zeros((40 - len(ego_history_positions), 2))
                ego_history_full = np.concatenate([ego_history_positions, padding])
                ego_history_tensor = torch.tensor(ego_history_full, dtype=torch.float32)
        else:
            ego_history_tensor = torch.zeros(40, 2, dtype=torch.float32)

        # 创建GraphData对象
        graph_data = GraphData(
            x=x,
            gnn_x=gnn_x,
            gnn_edge_index=gnn_edge_index,
            ego_history=ego_history_tensor
        )

        return graph_data

    def _compute_increments(self, traj: np.ndarray, start_pos: np.ndarray) -> np.ndarray:
        """计算轨迹增量"""
        increments = np.zeros((traj.shape[0], 2))
        increments[0] = traj[0, :2] - start_pos
        for t in range(1, traj.shape[0]):
            increments[t] = traj[t, :2] - traj[t-1, :2]
        return increments.flatten()

    def _process_agent_future(self, agent_future: np.ndarray, agent_history: np.ndarray) -> np.ndarray:
        """处理agent未来轨迹"""
        if len(agent_future) < 60:
            agent_future_2d = agent_future[:, :2]
            padding = np.zeros((60 - len(agent_future), 2))
            agent_future_2d = np.concatenate([agent_future_2d, padding])
        else:
            agent_future_2d = agent_future[:60, :2]

        agent_future_full = np.zeros((60, 5))
        agent_future_full[:, :2] = agent_future_2d
        agent_future_full[:, 2] = agent_history[-1, 2]  # heading
        agent_future_full[:, 3] = agent_history[-1, 3]  # vx
        agent_future_full[:, 4] = agent_history[-1, 4]  # vy
        return agent_future_full

    def _extrapolate_agent(self, agent_history: np.ndarray, target_len: int) -> np.ndarray:
        """线性外推agent轨迹"""
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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """获取单个样本"""
        data = self.data_list[idx]

        # 惰性重建 radius-30 边：cache 不存边（省 RAM/磁盘），用缓存节点位置现算，构造 fresh GraphData
        # （不写回 data_list，用完即弃，避免 84MB 边表在 RAM 里累积）
        _gd = data['graph_data']
        _edge_index = _build_gnn_edge_index(_gd.x[:, :2].numpy())
        _graph = GraphData(x=_gd.x, gnn_x=_gd.gnn_x, gnn_edge_index=_edge_index, ego_history=_gd.ego_history)

        return {
            'graph_data': _graph,
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
            'key_agent_id': data['key_agent_id'],
            'lane_polylines': torch.tensor(data['lane_polylines'], dtype=torch.float32),
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    自定义collate函数

    需要处理GraphData对象的批处理
    """
    from torch_geometric.data import Batch

    # 批处理GraphData对象
    graph_data_list = [item['graph_data'] for item in batch]
    batched_graph = Batch.from_data_list(graph_data_list)

    result = {
        'batched_graph': batched_graph,

        'ego_history': torch.stack([item['ego_history'] for item in batch]),
        'agent_history': torch.stack([item['agent_history'] for item in batch]),
        'ego_future_gt': torch.stack([item['ego_future_gt'] for item in batch]),
        'agent_future_gt': torch.stack([item['agent_future_gt'] for item in batch]),
        'gt_increments': torch.stack([item['gt_increments'] for item in batch]),
        'joint_increments': torch.stack([item['joint_increments'] for item in batch]),

        'ego_last_pos': torch.stack([item['ego_last_pos'] for item in batch]),
        'ego_last_heading': torch.stack([item['ego_last_heading'] for item in batch]),
        'ego_last_vx': torch.stack([item['ego_last_vx'] for item in batch]),
        'ego_last_vy': torch.stack([item['ego_last_vy'] for item in batch]),

        'svo_angle': torch.stack([item['svo_angle'] for item in batch]),
        'inherent_danger': torch.stack([item['inherent_danger'] for item in batch]),
        'gt_outcome_danger': torch.stack([item['gt_outcome_danger'] for item in batch]),
        'danger_target': torch.stack([item['danger_target'] for item in batch]),

        'scenario_names': [item['scenario_name'] for item in batch],
        'key_agent_ids': [item['key_agent_id'] for item in batch],
        'lane_polylines': torch.stack([item['lane_polylines'] for item in batch]),
    }

    return result


# ============================================================
# 训练函数
# ============================================================

def _compute_batch_loss(
    batch: dict,
    diffusion_model: DangerSceneDiffusion,
    gnn_trans_encoder: nn.Module,
    condition_encoder: SceneConditionEncoder,
    agent_pred_head: AgentPredictionHead,
    lane_encoder: nn.Module,
    loss_fn: DangerExceedLoss,
    device: str,
    t_generator: Optional[torch.Generator] = None,
    danger_dropout: float = 0.0,
):
    """单 batch 前向 + loss（不含 zero_grad/backward/step）。

    供 train_epoch（训练）与正式脚本的 val pass 共用，确保两侧前向口径完全一致。
    返回 (total_loss_batch, loss_dict, aux_loss)，total_loss_batch 已含 aux。
    t_generator 非 None 时用固定 generator 采样扩散时间步（val 复现，跨 epoch 可比）。
    """
    batched_graph = batch['batched_graph'].to(device)
    gt_increments = batch['joint_increments'].to(device)  # x_0 [B, 240]: ego_120 || agent_120

    # 轨迹数据（用于danger计算）
    ego_history = batch['ego_history'].to(device)  # [batch, 40, 5]
    agent_history = batch['agent_history'].to(device)  # [batch, 40, 5]
    agent_future_gt = batch['agent_future_gt'].to(device)  # [batch, 60, 5]

    # 最后帧状态（用于恢复轨迹）
    ego_last_pos = batch['ego_last_pos'].to(device)  # [batch, 2]
    ego_last_heading = batch['ego_last_heading'].to(device)  # [batch]
    ego_last_vx = batch['ego_last_vx'].to(device)  # [batch]
    ego_last_vy = batch['ego_last_vy'].to(device)  # [batch]

    # 条件编码输入
    svo_angle = batch['svo_angle'].to(device)  # [batch, 1]
    inherent_danger = batch['inherent_danger'].to(device)  # [batch, 1]
    danger_target = batch['danger_target'].to(device)  # [batch, 1]
    gt_outcome_danger = batch['gt_outcome_danger'].to(device)  # [batch, 1]
    lane_polylines = batch['lane_polylines'].to(device)  # [batch, 200, 4]

    # 1. GNN+Transformer编码场景特征，同时得到预训练的 ego 预测
    scene_feat, ego_pred = gnn_trans_encoder(batched_graph)

    # 1.5. AgentPredictionHead：用 scene_feat + agent 最后状态预测 agent 基线轨迹
    agent_last_state = agent_history[:, -1, :]                            # [B, 5]
    agent_pred = agent_pred_head(scene_feat, agent_last_state)            # [B, 120]
    pred_baseline = torch.cat([ego_pred, agent_pred], dim=-1)             # [B, 240]
    aux_loss = 0.3 * F.mse_loss(pred_baseline, gt_increments)

    # 纯 DDPM：用 GT outcome danger 作为 conditioning target（固定，不动态优化）
    danger_target_static = gt_outcome_danger  # [batch, 1]

    # 2. 条件编码 -> (batch, 256)
    # B1: 从 condition_encoder 读 use_agent_xy（权威，且与其 Linear 维度一致；兼容 DDP）。
    _enc_mod = condition_encoder.module if hasattr(condition_encoder, 'module') else condition_encoder
    _use_xy = getattr(_enc_mod, 'use_agent_xy', False)
    # use_agent_xy=True 传 [x,y,heading,vx,vy]（0:5），否则只传 [heading,vx,vy]（2:5）
    agent_state = agent_history[:, -1, 0:5] if _use_xy else agent_history[:, -1, 2:5]
    # v10 CFG：danger_dropout 仅训练态生效（encoder 内按 self.training 二次门控；val 传 0）
    condition = condition_encoder(scene_feat, svo_angle, inherent_danger, danger_target_static,
                                  agent_state=agent_state, danger_dropout=danger_dropout)

    # Map conditioning: fuse lane features into condition
    if diffusion_model.config.use_map_conditioning and lane_encoder is not None:
        condition = condition + lane_encoder(lane_polylines)

    # 3. Diffusion前向传播：添加噪声并预测
    batch_size = gt_increments.size(0)
    num_diffusion_steps = diffusion_model.config.num_diffusion_steps

    # 随机采样时间步（val 传入固定 generator 以跨 epoch 可比）
    t = torch.randint(0, num_diffusion_steps, (batch_size,), device=device, generator=t_generator)

    # 添加噪声
    x_t, noise = diffusion_model.add_noise(gt_increments, t)

    # 预测噪声
    noise_pred = diffusion_model.denoiser(x_t, condition, t)

    # 4. 从噪声预测恢复x_0：x_0 = (x_t - sqrt(1-α̅_t) * noise_pred) / sqrt(α̅_t)
    alpha_bar_t = diffusion_model.alpha_bars[t]
    x_0_pred = (x_t - (1 - alpha_bar_t).sqrt().unsqueeze(-1) * noise_pred) / \
               alpha_bar_t.sqrt().unsqueeze(-1)

    # 5. 计算 Loss（纯 DDPM：recon + jerk/direction 正则化）
    total_loss_batch, loss_dict = loss_fn(
        x_0_pred, gt_increments,
        ego_history, agent_history,
        ego_last_pos, ego_last_heading, ego_last_vx, ego_last_vy,
        gt_outcome_danger.squeeze(-1),
        svo_angle,
        danger_target=danger_target_static.squeeze(-1),
    )
    total_loss_batch = total_loss_batch + aux_loss
    return total_loss_batch, loss_dict, aux_loss


def train_epoch(
    diffusion_model: DangerSceneDiffusion,
    gnn_trans_encoder: nn.Module,
    condition_encoder: SceneConditionEncoder,
    agent_pred_head: AgentPredictionHead,
    lane_encoder: nn.Module,
    loss_fn: DangerExceedLoss,
    danger_config_torch: DangerScoreConfigTorch,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    danger_dropout: float = 0.0,
) -> Dict:
    """
    训练一个epoch - 使用完整DangerExceedLoss

    Loss组成（实际计入 total 的项）:
    1. recon_loss: MSE(x_0_pred, x_0)（recon_weight=1.0）
    2. jerk_loss / direction_reversal_loss / proximity_loss / lane_deviation_loss: 轨迹正则
    3. aux_loss: MSE(pred_baseline, joint_gt)  — 监督 AgentPredictionHead
    注: danger/upper_bound 权重为 0（纯 DDPM，推理时才用 SVO 梯度引导）；
        smoothness/validity 旧占位已移除（平滑由 jerk_loss 管）。
    """
    diffusion_model.train()
    gnn_trans_encoder.train()
    condition_encoder.train()
    agent_pred_head.train()
    if lane_encoder is not None:
        lane_encoder.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    # 真正计入 total 的正则项
    total_jerk_loss = 0.0
    total_dir_rev_loss = 0.0
    total_proximity_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        # 清零梯度
        optimizer.zero_grad()

        # 单 batch 前向 + loss（与 val 共用 _compute_batch_loss，口径一致）
        total_loss_batch, loss_dict, aux_loss = _compute_batch_loss(
            batch, diffusion_model, gnn_trans_encoder, condition_encoder,
            agent_pred_head, lane_encoder, loss_fn, device,
            danger_dropout=danger_dropout,
        )

        # 反向传播
        total_loss_batch.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            list(gnn_trans_encoder.parameters()) +
            list(condition_encoder.parameters()) +
            list(diffusion_model.parameters()) +
            list(agent_pred_head.parameters()) +
            (list(lane_encoder.parameters()) if lane_encoder is not None else []),
            max_norm=1.0
        )

        optimizer.step()

        # 记录
        total_loss += total_loss_batch.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_jerk_loss += loss_dict['jerk_loss'].item()
        total_dir_rev_loss += loss_dict['direction_reversal_loss'].item()
        total_proximity_loss += loss_dict['proximity_loss'].item()
        num_batches += 1

        progress_bar.set_postfix({
            'loss': f"{total_loss_batch.item():.4f}",
            'recon': f"{loss_dict['recon_loss'].item():.3f}",
            'dir_rev': f"{loss_dict['direction_reversal_loss'].item():.3f}",
            'aux': f"{aux_loss.item():.3f}",
        })

    return {
        'total_loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'jerk_loss': total_jerk_loss / num_batches,
        'direction_reversal_loss': total_dir_rev_loss / num_batches,
        'proximity_loss': total_proximity_loss / num_batches,
        'num_batches': num_batches
    }


def build_models(config: TrainingConfig, device: str):
    """构造 5 个模型组件（行为与 train_integrated 内联构造一致，不含 DDP 包装）。

    供小批验证脚本与正式训练脚本（train_diffusion_v9.py）共用，杜绝副本漂移。
    返回 (gnn_trans_encoder, condition_encoder, diffusion_model, agent_pred_head,
          lane_encoder, diffusion_config)；lane_encoder 在 use_map_conditioning=False 时为 None。
    """
    use_map = getattr(config, 'use_map_conditioning', True)

    gnn_trans_encoder = _GNNWrapper(TransformerPredictor()).to(device)
    condition_config = ConditionEncoderConfig(
        use_agent_xy=getattr(config, 'use_agent_xy', False),
        danger_dropout_enable=getattr(config, 'danger_dropout_enable', False),  # v10 CFG
    )
    condition_encoder = SceneConditionEncoder(condition_config).to(device)
    diffusion_config = DiffusionConfig(use_map_conditioning=use_map)
    diffusion_model = DangerSceneDiffusion(diffusion_config).to(device)
    agent_pred_head = AgentPredictionHead().to(device)
    # lane_encoder 仅在 map conditioning 开启时创建，避免 DDP find_unused_parameters 崩溃
    lane_encoder = (
        LaneEncoder(out_dim=diffusion_config.lane_encoder_out_dim).to(device)
        if use_map else None
    )
    return (gnn_trans_encoder, condition_encoder, diffusion_model,
            agent_pred_head, lane_encoder, diffusion_config)


def build_loss(config: TrainingConfig, danger_config_torch) -> 'DangerExceedLoss':
    """构造 DangerExceedLoss（纯 DDPM 权重，与 train_integrated 内联一致）。

    recon=1.0、danger=0.0（纯 DDPM，不优化 danger，推理时用 SVO 梯度引导）、
    upper_bound=0.0、danger_margin=0.5。direction_reversal/jerk 为 DangerExceedLoss 内置默认。
    """
    _use_kin = getattr(config, 'use_kinematic_v6_0', False)
    return DangerExceedLoss(
        recon_weight=1.0,
        danger_weight=0.0,       # 纯 DDPM：不优化 danger，推理时用 SVO 梯度引导
        upper_bound_weight=0.0,  # 同上，避免 pred_danger/gt_danger 比值爆炸
        danger_margin=0.5,
        config=danger_config_torch,
        use_kinematic_v6_0=_use_kin,  # [v6.0] 六C
    )


def train_integrated(config: TrainingConfig):
    """完整训练流程，支持单卡和 torchrun 多卡 DDP。"""

    # ── DDP 初始化 ──────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0
    if use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = config.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA不可用，使用CPU")
            device = 'cpu'

    rank = dist.get_rank() if use_ddp else 0
    is_main = (rank == 0)

    if is_main:
        print("=" * 60)
        print("开始训练整合版 Diffusion 危险场景生成器")
        mode_str = f"DDP ({dist.get_world_size()} GPUs)" if use_ddp else "单卡"
        print(f"模式: {mode_str}  device: {device}")
        print("=" * 60)

    output_path = Path(config.output_dir)
    if is_main:
        output_path.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()  # 等 rank-0 建目录

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_path / f'train_log_{timestamp}.txt'

    # ── 数据集（只在 rank-0 构建/写缓存，其余进程等待） ───────────
    if is_main:
        print("\n1. 加载训练数据...")
    if use_ddp:
        dist.barrier()  # 确保 rank-0 先写完缓存

    dataset = IntegratedDiffusionDataset(
        config.processed_data_dir,
        config.danger_pair_dir,
        max_samples=config.max_samples,
        num_workers=8,
        use_kinematic_v6_0=getattr(config, 'use_kinematic_v6_0', False),  # [v6.0]
        exclude_truncated_agent=getattr(config, 'exclude_truncated_agent', False),  # v9
        fix_agent_alignment=getattr(config, 'fix_agent_alignment', False),  # v9
    )

    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
    else:
        sampler = None
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    if is_main:
        print(f"   数据集大小: {len(dataset)}")
        print(f"   Batch size: {config.batch_size}  Num batches: {len(dataloader)}")

    # ── 创建模型 ────────────────────────────────────────────────
    if is_main:
        print("\n2. 创建模型...")

    use_map = getattr(config, 'use_map_conditioning', True)

    (gnn_trans_encoder, condition_encoder, diffusion_model,
     agent_pred_head, lane_encoder, diffusion_config) = build_models(config, device)

    if use_ddp:
        gnn_trans_encoder        = DDP(gnn_trans_encoder,        device_ids=[local_rank], find_unused_parameters=False)
        condition_encoder        = DDP(condition_encoder,        device_ids=[local_rank], find_unused_parameters=False)
        diffusion_model.denoiser = DDP(diffusion_model.denoiser, device_ids=[local_rank], find_unused_parameters=False)
        agent_pred_head          = DDP(agent_pred_head,          device_ids=[local_rank], find_unused_parameters=False)
        if use_map:
            lane_encoder         = DDP(lane_encoder,             device_ids=[local_rank], find_unused_parameters=False)

    danger_config_torch = DangerScoreConfigTorch()
    _use_kin = getattr(config, 'use_kinematic_v6_0', False)
    loss_fn = build_loss(config, danger_config_torch)
    if _use_kin and is_main:
        print("  [v6.0 六C] DangerExceedLoss 使用运动学解码模式")

    lane_params = list(lane_encoder.parameters()) if use_map else []

    if is_main:
        total_params = sum(p.numel() for p in gnn_trans_encoder.parameters()) + \
                       sum(p.numel() for p in condition_encoder.parameters()) + \
                       sum(p.numel() for p in diffusion_model.parameters()) + \
                       sum(p.numel() for p in agent_pred_head.parameters()) + \
                       sum(p.numel() for p in lane_params)
        print(f"   总参数量: {total_params}  Loss: pure DDPM (recon=1.0, aux=0.3, smooth+jerk 正则化)")
        print(f"   Map conditioning: {diffusion_config.use_map_conditioning}")

    # ── 优化器 ──────────────────────────────────────────────────
    all_params = list(gnn_trans_encoder.parameters()) + \
                 list(condition_encoder.parameters()) + \
                 list(diffusion_model.parameters()) + \
                 list(agent_pred_head.parameters()) + \
                 lane_params
    optimizer = optim.Adam(all_params, lr=config.lr)

    # ── 辅助：提取 state_dict（剥掉 DDP 的 module. 前缀） ────────
    def _ckpt_dict(epoch, extra=None):
        gnn_w  = gnn_trans_encoder.module if use_ddp else gnn_trans_encoder
        cond   = condition_encoder.module  if use_ddp else condition_encoder
        aph    = agent_pred_head.module    if use_ddp else agent_pred_head
        diff_sd = diffusion_model.state_dict()
        if use_ddp:
            diff_sd = {k.replace('denoiser.module.', 'denoiser.'): v for k, v in diff_sd.items()}
        d = {
            'gnn_trans_encoder':   gnn_w.predictor.state_dict(),
            'condition_encoder':   cond.state_dict(),
            'diffusion_model':     diff_sd,
            'agent_prediction_head': aph.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if use_map:
            lane_e = lane_encoder.module if use_ddp else lane_encoder
            d['lane_encoder'] = lane_e.state_dict()
        if extra:
            d.update(extra)
        return d

    # ── 训练循环 ────────────────────────────────────────────────
    if is_main:
        print("\n3. 开始训练...")
    best_loss = float('inf')
    # 早停状态（滑动窗口均值 + patience）
    loss_history: List[float] = []
    best_window_mean = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        if use_ddp:
            sampler.set_epoch(epoch)  # 每 epoch 重新 shuffle

        train_info = train_epoch(
            diffusion_model, gnn_trans_encoder, condition_encoder, agent_pred_head,
            lane_encoder, loss_fn, danger_config_torch,
            dataloader, optimizer, device, epoch,
            danger_dropout=getattr(config, 'danger_dropout', 0.0),  # v10 CFG
        )

        should_stop = False
        if is_main:
            log_line = (
                f"Epoch {epoch}: total={train_info['total_loss']:.4f}, "
                f"recon={train_info['recon_loss']:.4f}, "
                f"jerk={train_info['jerk_loss']:.4f}, "
                f"dir_rev={train_info['direction_reversal_loss']:.4f}, "
                f"prox={train_info['proximity_loss']:.4f}"
            )
            print(log_line)
            with open(log_file, 'a') as f:
                f.write(log_line + '\n')

            if train_info['total_loss'] < best_loss:
                best_loss = train_info['total_loss']
                torch.save(_ckpt_dict(epoch, {'loss': best_loss}),
                           output_path / 'best_model.pt')
                print(f"   保存最佳模型: loss={best_loss:.4f}")

            if epoch % config.save_every == 0:
                torch.save(_ckpt_dict(epoch),
                           output_path / f'checkpoint_epoch_{epoch}.pt')
                print(f"   保存checkpoint: epoch={epoch}")

            # 早停：近 window 个 epoch loss 的滑动均值（抗 diffusion 单 epoch 噪声），
            # 连续 patience 个 epoch 滑动均值无改善(> min_delta)则停。best_model 始终保留全程最优。
            if config.early_stop_patience and config.early_stop_patience > 0:
                loss_history.append(train_info['total_loss'])
                w = max(1, config.early_stop_window)
                if len(loss_history) >= w:
                    window_mean = sum(loss_history[-w:]) / w
                    if window_mean < best_window_mean - config.early_stop_min_delta:
                        best_window_mean = window_mean
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= config.early_stop_patience:
                        msg = (f"   [早停] 近 {w}-epoch 滑动均值连续 {epochs_no_improve} epoch 无改善"
                               f"（best_window_mean={best_window_mean:.4f}）→ 在 epoch {epoch} 停止")
                        print(msg)
                        with open(log_file, 'a') as f:
                            f.write(msg + '\n')
                        should_stop = True

        # DDP 下广播停止决策，保证各 rank 一起退出（单进程时直接 break）
        if use_ddp:
            stop_t = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_t, src=0)
            should_stop = bool(stop_t.item())
        if should_stop:
            break

    if use_ddp:
        dist.destroy_process_group()

    if is_main:
        print("\n4. 训练完成！")
        print(f"   最佳loss: {best_loss:.4f}  模型保存到: {output_path}")

    return {'best_loss': best_loss, 'output_dir': str(output_path)}


# ============================================================
# 测试函数
# ============================================================

def test_single_sample():
    """
    测试单个样本的处理流程
    """
    print("\n" + "=" * 60)
    print("测试单个样本处理流程")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载单个样本
    processed_dir = config.PROCESSED_DATA_DIR
    danger_pair_dir = config.DANGER_PAIR_SCENARIOS_DIR

    dataset = IntegratedDiffusionDataset(processed_dir, danger_pair_dir, max_samples=2)

    if len(dataset) == 0:
        print("[ERROR] 没有成功加载任何样本")
        return False

    sample = dataset[0]
    print(f"\n样本数据:")
    print(f"  graph_data.x: {sample['graph_data'].x.shape}")
    print(f"  graph_data.gnn_edge_index: {sample['graph_data'].gnn_edge_index.shape}")
    print(f"  ego_history: {sample['ego_history'].shape}")
    print(f"  gt_increments: {sample['gt_increments'].shape}")
    print(f"  svo_angle: {sample['svo_angle'].shape}")
    print(f"  inherent_danger: {sample['inherent_danger']}")
    print(f"  gt_outcome_danger: {sample['gt_outcome_danger']}")

    # 创建模型
    print("\n创建模型...")
    gnn_trans_encoder = TransformerPredictor().to(device)
    condition_config = ConditionEncoderConfig()
    condition_encoder = SceneConditionEncoder(condition_config).to(device)
    diffusion_config = DiffusionConfig()
    diffusion_model = DangerSceneDiffusion(diffusion_config).to(device)

    # 批处理
    batch = collate_fn([sample])
    batched_graph = batch['batched_graph'].to(device)
    gt_increments = batch['gt_increments'].to(device)
    svo_angle = batch['svo_angle'].to(device)
    inherent_danger = batch['inherent_danger'].to(device)
    danger_target = batch['danger_target'].to(device)

    print(f"\n批处理后:")
    print(f"  batched_graph.x: {batched_graph.x.shape}")
    print(f"  batched_graph.batch: {batched_graph.batch.shape if hasattr(batched_graph, 'batch') else 'None'}")

    # 测试前向传播
    print("\n测试前向传播...")

    with torch.no_grad():
        # 1. GNN+Transformer编码
        scene_feat = gnn_trans_encoder.get_encoder_features(batched_graph)
        print(f"  scene_feat: {scene_feat.shape}")

        # 2. 条件编码
        agent_history_t = batch['agent_history'].to(device)  # [1, 40, 5]
        agent_state_t = agent_history_t[:, -1, 2:5]          # [1, 3]
        condition = condition_encoder(scene_feat, svo_angle, inherent_danger, danger_target,
                                      agent_state=agent_state_t)
        print(f"  condition: {condition.shape}")

        # 3. Diffusion前向传播
        loss, info = diffusion_model(gt_increments, condition)
        print(f"  diffusion_loss: {loss.item():.4f}")

        # 4. 生成测试
        generated = diffusion_model.generate(condition)
        print(f"  generated: {generated.shape}")

    print("\n" + "=" * 60)
    print("测试通过！")
    print("=" * 60)

    return True


# ============================================================
# 主函数
# ============================================================

def set_seed(seed: int = 42) -> None:
    """固定随机种子（复现）。脚本原本不设 seed → v8 起固定（不开 cudnn.deterministic 以免拖慢训练）。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description='训练整合版Diffusion危险场景生成器')

    parser.add_argument('--processed_data_dir', type=str,
                        default=config.DIFFUSION_PROCESSED_DATA_DIR,
                        help='processed_data目录')

    parser.add_argument('--danger_pair_dir', type=str,
                        default=config.NEW_DANGER_PAIR_DIR,
                        help='danger_pair目录')

    parser.add_argument('--output_dir', type=str,
                        default=config.DIFFUSION_TRAINING_GNN_TRANS_DIR,
                        help='输出目录')

    parser.add_argument('--epochs', type=int, default=50,
                        help='训练epoch数')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')

    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于调试）')

    parser.add_argument('--save_every', type=int, default=10,
                        help='每隔多少epoch保存一次')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='早停 patience（滑动均值连续无改善的 epoch 数）；<=0 关闭')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='早停最小改善阈值（滑动均值需至少降这么多才算改善）')
    parser.add_argument('--early_stop_window', type=int, default=5,
                        help='早停滑动窗口大小（对近 N 个 epoch loss 取均值抗噪）')

    parser.add_argument('--test_only', action='store_true',
                        help='只运行测试，不训练')
    parser.add_argument('--use_kinematic_v6_0', action='store_true', default=False,
                        help='[v6.0] 六C: 用 (Δv,Δθ) 运动学空间训练，物理有效性由 clamp 结构保证')
    parser.add_argument('--no_map_conditioning', action='store_true', default=False,
                        help='关闭 lane map conditioning（baseline 消融用）；默认开启')
    parser.add_argument('--exclude_truncated_agent', action='store_true', default=False,
                        help='v9: 跳过 agent 未来不满 6s（尾部 padding）的样本，修复 target 污染')
    parser.add_argument('--fix_agent_alignment', action='store_true', default=False,
                        help='v9: 用 agent_future[0] 重构 agent present，修 A1 时间方向 bug')
    parser.add_argument('--condition_agent_xy', action='store_true', default=False,
                        help='v9 阶段2 B1: condition 加入 key agent 的 xy 位置（agent_state 4→6 维）')

    args = parser.parse_args()

    set_seed(42)

    if args.test_only:
        # 只运行测试
        success = test_single_sample()
        return

    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA不可用，使用CPU")
        args.device = 'cpu'

    # 创建配置
    # [v6.0] 六C: 自动修改 output_dir 以区分 kinematic 训练结果
    out_dir = args.output_dir
    if args.use_kinematic_v6_0 and 'v6_0_kinematic' not in out_dir:
        from pathlib import Path as _P
        out_dir = str(_P(out_dir).parent / (_P(out_dir).name + '_v6_0_kinematic'))
    if args.exclude_truncated_agent and 'excltrunc' not in out_dir:
        from pathlib import Path as _P
        out_dir = str(_P(out_dir).parent / (_P(out_dir).name + '_excltrunc'))
    if args.fix_agent_alignment and 'fixalign' not in out_dir:
        from pathlib import Path as _P
        out_dir = str(_P(out_dir).parent / (_P(out_dir).name + '_fixalign'))
    if args.condition_agent_xy and 'agentxy' not in out_dir:
        from pathlib import Path as _P
        out_dir = str(_P(out_dir).parent / (_P(out_dir).name + '_agentxy'))
    train_config = TrainingConfig(
        processed_data_dir=args.processed_data_dir,
        danger_pair_dir=args.danger_pair_dir,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        max_samples=args.max_samples,
        save_every=args.save_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_window=args.early_stop_window,
        use_kinematic_v6_0=args.use_kinematic_v6_0,  # [v6.0]
        use_map_conditioning=not args.no_map_conditioning,
        exclude_truncated_agent=args.exclude_truncated_agent,  # v9
        fix_agent_alignment=args.fix_agent_alignment,  # v9
        use_agent_xy=args.condition_agent_xy,  # v9 阶段2 B1
    )

    # 开始训练
    result = train_integrated(train_config)

    print("\n训练结果:")
    print(f"  最佳loss: {result['best_loss']:.4f}")
    print(f"  输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
