"""E1 · MAGNET imitation trainer（路线B，脱离 diffusion/danger 线）。

用 E0 新契约的 GraphDataset（离线 .pt）+ MAGNET `TransformerPredictor`，做纯 imitation 训练：
  预测 60×2 局部增量轨迹，对 ground-truth 未来轨迹算加权 ADE/FDE/smoothness/direction 损失。
train/val 走 log-disjoint split（split_log_disjoint.py 产出的两个 leaf root），防同一 nuPlan log 泄漏。
存 `best_model_default.pth`（bare state_dict，对齐在线 planner 硬编码加载名）+ 富 checkpoint + loss/env 记录。

**边界**：不 import 任何 danger 代码（train_diffusion_*/get_encoder_features/danger_pairs 全禁）。
**sys.path 陷阱**：`TransformerPredictor` 的 E0 生效副本在 parent `nuplan-devkit/models`，本脚本显式 insert。

运行（conda nuplan env；base env 的 torch 是坏的）：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/train_magnet_imitation.py \
      --train_root /abs/.../train --val_root /abs/.../val \
      --epochs 100 --batch_size 8 --lr 5e-4 --device cuda:0 \
      --output_dir /abs/.../experiments/magnet_imitation_e1/run_xxx
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- sys.path: GNN_Trans（config/src.shared）+ parent models（E0 生效 transformer_predictor）+ devkit 兜底 ---
_THIS = Path(__file__).resolve()
GNN_TRANS_ROOT = _THIS.parents[2]              # scripts/training/x.py -> GNN_Trans
DEVKIT_ROOT = GNN_TRANS_ROOT.parent            # nuplan-devkit
MODELS_DIR = DEVKIT_ROOT / "models"            # 生效 transformer_predictor.py 所在
for _p in (str(GNN_TRANS_ROOT), str(MODELS_DIR), str(DEVKIT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402  GNN_Trans/config.py
from transformer_predictor import TransformerPredictor  # noqa: E402  parent models/（E0 生效副本）
from src.shared.utils.dataset import GraphDataset, get_adjacency_config, set_subgraph_agg  # noqa: E402
from perturbation import EgoPerturbation  # noqa: E402  同目录（scripts/training）
from torch_geometric.loader import DataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_magnet_imitation")

NUM_FUTURE_POSES = 60  # E0 契约：6s/60 帧未来


def set_seed(seed: int = 42) -> None:
    """设置随机种子（实验可复现规则）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment() -> Dict[str, str]:
    """记录环境信息（复现规则）。"""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
    }


class ImprovedTrajectoryLoss(nn.Module):
    """加权 ADE/FDE/smoothness/direction 损失（自 legacy 抽出，权重沿用其实际用值）。"""

    def __init__(self, ade_weight: float = 0.5, fde_weight: float = 3.0,
                 smoothness_weight: float = 0.3, direction_weight: float = 0.5) -> None:
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.smoothness_weight = smoothness_weight
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_components: bool = False):
        bs = pred.shape[0]
        pred_traj = pred.view(bs, NUM_FUTURE_POSES, 2)
        true_traj = target.view(bs, NUM_FUTURE_POSES, 2)

        disp = torch.norm(pred_traj - true_traj, dim=2)
        ade_loss = disp.mean()
        fde_loss = torch.norm(pred_traj[:, -1, :] - true_traj[:, -1, :], dim=1).mean()

        pred_d = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
        true_d = true_traj[:, 1:, :] - true_traj[:, :-1, :]
        smooth_loss = torch.norm(pred_d - true_d, dim=2).mean()

        pred_dir = pred_traj[:, -1, :] - pred_traj[:, 0, :]
        true_dir = true_traj[:, -1, :] - true_traj[:, 0, :]
        pred_u = pred_dir / torch.clamp(torch.norm(pred_dir, dim=1, keepdim=True), min=1e-8)
        true_u = true_dir / torch.clamp(torch.norm(true_dir, dim=1, keepdim=True), min=1e-8)
        direction_loss = (1 - torch.sum(pred_u * true_u, dim=1)).mean()

        total = (self.ade_weight * ade_loss + self.fde_weight * fde_loss
                 + self.smoothness_weight * smooth_loss + self.direction_weight * direction_loss)
        if return_components:
            return {"total": total.item(), "ade": ade_loss.item(), "fde": fde_loss.item(),
                    "smoothness": smooth_loss.item(), "direction": direction_loss.item()}
        return total


def resolve_monitor_name(ckpt_metric: str, multimodal: bool, joint: bool = False) -> str:
    """解析训练控制信号名（D0#19）。**纯函数，供单测直接调用。**

    :param ckpt_metric: `auto` / `val_loss` / `val_ade`。
    :param multimodal: 是否多模态格。
    :param joint: 🆕 路线 B 的 joint 格。它的 `val_loss` 同样含选模 CE 项、与误差不同向，
        且 ego 与周车混在一个损失里 ⇒ **必须**用 `val_ade`（实际取 `all_minADE`）。
    :returns: `"val_ade"` 或 `"val_loss"`。

    🔴 `auto` 的含义：**多模态用 `val_ade`、单模态用 `val_loss`**。
    单模态保持 `val_loss` 不是偷懒，是为了让既有 checkpoint **逐位可复现**
    （单模态下 val_loss 与 ADE 同向，两者选出的 epoch 实测一致，见
    `tests/test_ckpt_selection.py` 的回归门）。
    """
    if ckpt_metric not in ("auto", "val_loss", "val_ade"):
        raise ValueError(f"未知 ckpt_metric {ckpt_metric!r}")
    if ckpt_metric != "auto":
        return ckpt_metric
    return "val_ade" if (multimodal or joint) else "val_loss"


class EarlyStopping:
    """val 无改善 patience 轮后停，恢复最优权重（自 legacy 抽出）。"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        if self.counter >= self.patience:
            if self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


def move_batch_to_device(data, device: torch.device):
    """把 GraphData（含 ego_history/gnn_* 等自定义属性）整体搬到 device（自 legacy 抽出兜底）。"""
    custom = ["ego_history", "ego_last_pos", "gnn_x", "gnn_edge_index",
              "gnn_edge_attr", "gnn_x_batch",   # D0#7：不搬它会留在 CPU，池化时报 device 不一致
              "traj_id_to_indices", "cluster", "valid_len", "time_step_len",
              # 闸⑥：GraphData.to() 的 custom_attrs 白名单里没有这三个，
              # 靠 PyG 的通用 store 搬运；显式列出是兜底（漏搬会在 aux loss 处报 device 不一致）
              "gnn_is_agent", "agent_future_tgt", "agent_future_mask"]
    saved = {a: getattr(data, a) for a in custom if hasattr(data, a)}
    data = data.to(device)
    for a, v in saved.items():
        if not hasattr(data, a) or getattr(data, a) is None:
            setattr(data, a, v.to(device) if isinstance(v, torch.Tensor) else v)
    return data


def _ade_fde(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """由增量制预测/真值算 per-sample ADE/FDE（米）。

    pred/target 均为 60×2 局部逐帧增量；cumsum 得相对位置。两者共用 origin=0，相减时
    origin 自动抵消，故直接 cumsum(零起点)即可，ADE/FDE 与参考原点无关。
    """
    bs = pred.shape[0]
    pred_pos = torch.cumsum(pred.view(bs, NUM_FUTURE_POSES, 2), dim=1)
    true_pos = torch.cumsum(target.view(bs, NUM_FUTURE_POSES, 2), dim=1)
    err = torch.norm(pred_pos - true_pos, dim=2)  # (bs, 60)
    return err.mean(dim=1), err[:, -1]


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
              optimizer, device: torch.device, train: bool,
              aux_cfg: "AgentAuxConfig | None" = None) -> Tuple[float, float, float, float]:
    """跑一个 epoch（train=True 反传）；返回 (avg_loss, avg_ade, avg_fde, avg_aux)。

    :param aux_cfg: 闸⑥ 配置；None = 关闭（与基线逐字节同路径）。

    🔴 **aux loss 只加进 train 的反传，不进 `val_loss`**。
    `val_loss` 是 early stopping 与 best-checkpoint 选择的依据，也是与基线格对比的量；
    把 aux 混进去会让三格选出的 checkpoint 依据不同 ⇒ 比较的不再是同一件事。
    aux 的 val 值仍然计算并返回，仅作监控（判断"真/打乱"两条曲线何时分叉）。
    """
    model.train(train)
    total_loss, total_aux, n = 0.0, 0.0, 0
    ade_all, fde_all = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for data in loader:
            data = move_batch_to_device(data, device)
            output = model(data)                       # (B, 120)
            bs = int(data.num_graphs)
            if output.shape[0] != bs:                  # 防御：模型 batch 维异常
                output = output.repeat(bs, 1) if output.shape[0] == 1 else output[:bs]
            y = data.y.reshape(bs, -1).to(device)      # (B, 120)

            loss = criterion(output, y)
            aux_val = 0.0
            if aux_cfg is not None:
                aux_loss = aux_cfg.compute(model, data)
                aux_val = float(aux_loss.detach())
                if train:
                    loss = loss + aux_loss             # ★ 只在 train 进反传
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 🔴 记账用的 total_loss 必须是**ego 部分**，否则 train_loss 与基线不可比
            ego_loss = float(loss.detach()) - (aux_val if (train and aux_cfg is not None) else 0.0)
            total_loss += ego_loss * bs
            total_aux += aux_val * bs
            n += bs
            ade, fde = _ade_fde(output.detach(), y)
            ade_all.append(ade.cpu())
            fde_all.append(fde.cpu())

    avg_loss = total_loss / max(n, 1)
    avg_aux = total_aux / max(n, 1)
    avg_ade = float(torch.cat(ade_all).mean()) if ade_all else 0.0
    avg_fde = float(torch.cat(fde_all).mean()) if fde_all else 0.0
    return avg_loss, avg_ade, avg_fde, avg_aux


class AgentAuxConfig:
    """闸⑥ 的 aux loss 驱动器（把 batch → aux loss 这段封起来，`run_epoch` 只调 `compute`）。

    :param shuffle: True 即 `+aux(打乱)` 对照格 —— batch 内跨 agent 重排目标，
        **保容量与梯度总量、只破坏信号**。
    :param weight: aux loss 权重。
    :param seed: 打乱用的 generator seed（**CPU generator**，见 `_shuffle_perm` 的 CUDA 坑）。
    """

    def __init__(self, shuffle: bool = False, weight: float = 1.0, seed: int = 42) -> None:
        self.shuffle = shuffle
        self.weight = weight
        self.generator = torch.Generator().manual_seed(seed)

    def compute(self, model: nn.Module, data) -> torch.Tensor:
        from magnet_agent_aux import masked_aux_loss
        out = model.agent_aux_forward(data)
        if out is None:                                # 整批没有 agent 节点（空图）
            return torch.zeros((), device=next(model.parameters()).device)
        pred, _idx = out
        tgt = data.agent_future_tgt.to(pred.device)
        msk = data.agent_future_mask.to(pred.device)
        if tgt.shape[0] != pred.shape[0]:
            raise RuntimeError(
                f"闸⑥ 行数对不上：aux 预测 {pred.shape[0]} 行、目标 {tgt.shape[0]} 行 —— "
                f"batch 拼接后 `gnn_is_agent` 与 `agent_future_tgt` 失去对齐，"
                f"继续训练会产出系统性错误的监督（见 test_agent_aux_integration.py B1/B3）")
        return masked_aux_loss(pred, tgt, msk, shuffle=self.shuffle,
                               weight=self.weight, generator=self.generator)


def build_loaders(train_root: str, val_root: str, batch_size: int,
                  dataset_config: str, train_transform=None,
                  gnn_pooling_fix: bool = False,
                  agent_aux: bool = False,
                  joint_adaptive: bool = False,
                  temporal_graph: bool = False,
                  temporal_topology: str = "full",
                  oracle_gate: str = None,
                  edge_rank_loss: bool = False,
                  num_workers: int = 0) -> Tuple[DataLoader, DataLoader, int, int]:
    """构建 train/val 两个 GraphDataset（各自独立 root，log-disjoint）。

    :param train_transform: 仅施加于 **train** 的在线增广（如 EgoPerturbation）。
        val 恒为 `transform=None` —— 增广绝不能污染模型选择依据（early stopping / 超参）。
    :param agent_aux: 闸⑥ —— 换成 `AgentAuxGraphDataset`，额外挂三个 aux 字段。
        缓存须先由 `build_agent_aux_cache.py` 生成；缺缓存时**直接报错不降级**
        （静默退回无 aux 会让"打开了闸⑥"变成一句空话）。
    :param temporal_graph: R1（Q76 重构）—— 挂逐帧场景图 `tg_x/tg_edge_index/tg_frame`。
        🔴 它**不是增广**而是必需的数据构造 ⇒ **train 与 val 都要挂**，
        与上面 "val 恒为 None" 的规则不冲突（那条针对的是增广）。
        走零缓存路径（从 `data.x` 现场重建），**不重建 .pt 缓存** ——
        存 `tg_edge_index` 在 full 拓扑下要 ≈14 GB。
    """
    get_adjacency_config(dataset_config)  # 设全局，GraphDataset 据此命名 .pt 并选 edge builder

    temporal_tf = None
    if temporal_graph:
        from src.shared.utils.temporal_graph import AttachTemporalGraph
        temporal_tf = AttachTemporalGraph(topology=temporal_topology)

    def _with_temporal(base):
        """把逐帧图 transform 接在既有 transform 之后（增广先、建图后）。"""
        if temporal_tf is None:
            return base
        if base is None:
            return temporal_tf
        from torch_geometric.transforms import Compose
        return Compose([base, temporal_tf])

    if agent_aux or joint_adaptive:
        # 🆕 joint 也走 `AgentAuxGraphDataset`：它挂的三个字段正是 joint target 的**周车侧**
        #   （`gnn_is_agent` / `agent_future_tgt` / `agent_future_mask`，位置域绝对坐标）。
        #   ego 侧现算 `cumsum(data.y)`，无需缓存。缺缓存时**直接报错不降级**。
        if oracle_gate:
            # E1（Q108 §7 判据3）：注入 S3 边效用当逐边 `w`，量「拓扑即使完全正确的上界」。
            # 🔴 train 与 val **都要挂** —— 它不是增广而是模型输入的一部分；
            #    只挂 train 会让 val 与部署口径不一致，best-checkpoint 就选错了。
            from magnet_oracle_gate_dataset import OracleGateGraphDataset
            train_ds = OracleGateGraphDataset(train_root, transform=_with_temporal(train_transform),
                                              adjacency_config=dataset_config, variant=oracle_gate)
            val_ds = OracleGateGraphDataset(val_root, transform=_with_temporal(None),
                                            adjacency_config=dataset_config, variant=oracle_gate)
        elif edge_rank_loss:
            # S5：边监督项需要逐边的 `s4_U` / `s4_Usd` / `s4_stratum`（S3 标签烘的缓存）。
            # 🔴 train 与 val 都挂 —— val 侧只用于诊断读数，不进 best-checkpoint 控制信号。
            from magnet_edge_label_dataset import EdgeLabelGraphDataset
            train_ds = EdgeLabelGraphDataset(train_root, transform=_with_temporal(train_transform),
                                             adjacency_config=dataset_config)
            val_ds = EdgeLabelGraphDataset(val_root, transform=_with_temporal(None),
                                           adjacency_config=dataset_config)
        else:
            from magnet_agent_aux_dataset import AgentAuxGraphDataset
            train_ds = AgentAuxGraphDataset(train_root, transform=_with_temporal(train_transform),
                                            adjacency_config=dataset_config)
            val_ds = AgentAuxGraphDataset(val_root, transform=_with_temporal(None),
                                          adjacency_config=dataset_config)
    else:
        train_ds = GraphDataset(train_root, transform=_with_temporal(train_transform))
        val_ds = GraphDataset(val_root, transform=_with_temporal(None))   # ★ val 绝不增广
    logger.info("GraphDataset: train=%d, val=%d (adjacency=%s, train_transform=%s, temporal=%s)",
                len(train_ds), len(val_ds), dataset_config, train_transform, temporal_tf)
    # D0#7：follow_batch 产出 gnn_x_batch，供 MagnetImitationPooled 别名为 gnn_batch。
    # 不开修复时保持 None，DataLoader 行为与旧版逐字节相同。
    # 🔴 joint 必须有 `gnn_x_batch`：契约 B（`GraphData.__inc__` 对 `cluster` 返回 0 不偏移）
    #    ⇒ 没有它就无法把逐向量分组成**跨样本不串**的实体（V-12 的反证已固化这一点）。
    follow = ["gnn_x"] if (gnn_pooling_fix or joint_adaptive) else None
    if temporal_graph:
        # R1 需要 tg_x_batch 来做 (样本, 帧) 双重分组池化
        follow = (follow or []) + ["tg_x"]
    # num_workers>0 时必须一并给 persistent_workers，否则每个 epoch 重建 worker 进程；
    # 实测（bench_r1_gpu.py）R1 在 w=4 下 0.585 s/batch vs w=0 的 0.648 ⇒ 省 10%。
    dl_kw = {}
    if num_workers > 0:
        dl_kw = {"num_workers": num_workers, "persistent_workers": True}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              follow_batch=follow, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            follow_batch=follow, **dl_kw)
    return train_loader, val_loader, len(train_ds), len(val_ds)


#: 建类旋钮的互斥表：开关名 -> 与它互斥的其它开关名。
#: 🔴 **只有这一份** —— 之前 `--temporal_graph` 的检查写在模型构造处、`--multimodal` 的写在另一处，
#: 两份清单必然漂（第十六条）。而且构造发生在 `build_loaders` **之后**，
#: 打错 flag 要先等完整个数据集加载才报错。故提到 `train()` 第一行。
_ARCH_EXCLUSIVE = {
    # 🆕 路线 B（Q108/Q110 决策 D-e）：`joint_adaptive` 是**顶层分支不是旋钮** ——
    #   新结构把 ego/context/gnn 三条分支合并成一套 per-entity token，旧旋钮在它里面没有对应物。
    #   与 `bn_calibration.build_model` 的拦截清单**逐字对应**（两处不得分叉，第十六条）。
    "joint_adaptive": ("gnn_pooling_fix", "context_pooling", "ego_query_fusion",
                       "ego_shortcut_suppress", "agent_aux", "temporal_graph",
                       "multimodal", "gnn_operator", "head_norm"),
    "multimodal": ("gnn_pooling_fix", "context_pooling", "ego_query_fusion",
                   "ego_shortcut_suppress", "agent_aux", "temporal_graph", "gnn_operator",
                   "joint_adaptive"),
    "temporal_graph": ("gnn_pooling_fix", "context_pooling", "ego_query_fusion",
                       "ego_shortcut_suppress", "agent_aux", "multimodal", "gnn_operator",
                       "joint_adaptive"),
    "agent_aux": ("gnn_pooling_fix", "context_pooling", "ego_query_fusion",
                  "ego_shortcut_suppress", "temporal_graph", "multimodal", "gnn_operator",
                  "joint_adaptive"),
    # 旋钮 B：与其它建类旋钮互斥；与 --gnn_pooling_fix **不**互斥（它隐含 fix，同一语义，
    # 显式同开不构成第二变量）。
    "gnn_operator": ("context_pooling", "ego_query_fusion", "ego_shortcut_suppress",
                     "agent_aux", "temporal_graph", "multimodal", "joint_adaptive"),
}


def validate_arch_exclusivity(args: argparse.Namespace) -> None:
    """建类旋钮互斥校验 —— 必须在**加载数据之前**跑。

    每个格子的「唯一变量」是它能不能被归因的前提：同时开两个建类旋钮，
    差值里就有两个变量，事后无论怎么统计都拆不开（Q45 已经吃过一次亏）。

    :raises SystemExit: 有互斥组合被同时打开。
    """
    for knob, others in _ARCH_EXCLUSIVE.items():
        if not getattr(args, knob, False):
            continue
        on = [o for o in others if getattr(args, o, False)]
        if on:
            raise SystemExit(
                f"--{knob} 不得与 {['--' + o for o in on]} 同开："
                f"每格的唯一变量必须是一个，否则归因拆不开")


def train(args: argparse.Namespace) -> None:
    validate_arch_exclusivity(args)
    # 旋钮 A（判据 20260810）：必须在 build_loaders 之前设全局 —— GraphDataset 据此
    # 命名 .pt 缓存并在 process() 里选聚合函数；默认 mean = 旧行为与旧缓存逐位不变
    set_subgraph_agg(args.subgraph_agg)
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir).resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    env = log_environment()
    logger.info("env: %s", env)

    # 在线增广（仅 train）：seed 由实验 seed 显式派生，保证同 seed 完全可复现
    perturb = None
    if args.use_perturbation:
        perturb = EgoPerturbation(
            max_lateral=args.max_lateral, max_longitudinal=args.max_longitudinal,
            max_yaw=args.max_yaw, prob=args.perturb_prob,
            seed=(args.perturb_seed if args.perturb_seed is not None else args.seed),
        )
        logger.info("启用 perturbation 增广（仅 train）: %s", perturb)
    else:
        logger.info("perturbation 增广：关闭（默认，与旧训练命令兼容）")

    # 🔴 生效值 ≠ 原始 flag：context_pooling / ego_query_fusion 都**隐含**打开 follow_batch。
    #   记录里必须两个都留 —— 只记原始 flag 会让「C2 run 的 GNN 分支是逐样本还是整批共享」
    #   在事后无法判定（C2 命令行不带 --gnn_pooling_fix，但 GNN 分支实际是逐样本的）。
    #   闸④（Q50）同理：`MagnetEgoShortcutSuppressed` 也在 `MagnetImitationPooled` 继承链上，
    #   即使不带 --ego_query_fusion 也隐含逐样本池化 ⇒ 与 B 对照仍差两个变量（Q45 同类坑）。
    gnn_pooling_fix_effective = bool(
        args.gnn_pooling_fix or args.context_pooling or args.ego_query_fusion
        or args.ego_shortcut_suppress or args.gnn_operator or args.head_norm)
    train_loader, val_loader, n_train, n_val = build_loaders(
        args.train_root, args.val_root, args.batch_size, args.dataset_config,
        train_transform=perturb,
        # context_pooling 走 MagnetContextPooled（继承 MagnetImitationPooled），
        # 同样需要 follow_batch 产出的 gnn_x_batch，否则父类会走 mean(0).repeat 串样本。
        gnn_pooling_fix=gnn_pooling_fix_effective,
        agent_aux=args.agent_aux,
        joint_adaptive=args.joint_adaptive,
        temporal_graph=args.temporal_graph,
        temporal_topology=args.temporal_topology,
        oracle_gate=args.oracle_gate,
        edge_rank_loss=args.edge_rank_loss,
        num_workers=args.num_workers)

    # 生效副本自证（副本/sys.path 陷阱兜底）
    import inspect
    logger.info("TransformerPredictor <- %s", inspect.getfile(TransformerPredictor))
    logger.info("GraphDataset       <- %s", inspect.getfile(GraphDataset))

    # D0#7：逐样本池化修复（默认关闭 = 旧行为，父类 transformer_predictor.py 一字未改）
    # 判据0：context 分支 polyline 池化（VectorNet 式 encode-then-pool），同样默认关闭。
    #   实测 context 分支占 99.8% FLOPs（M 中位数 2036 个 segment token，O(M²) 注意力），
    #   池化到 polyline 级（P 中位数 44）后全模型 FLOPs 7986 -> 50 MFLOP（159×）。
    #   ⚠️ context_pooling 需要 cluster 且必须与 gnn_pooling_fix 一起用（子类继承关系）。
    # C2（阶段㊵–㊹）：ego-query attention 融合，替代两处 mean pooling。
    #   靶点由 G19+G16 三条独立证据锁定；🔴 **必须与非完全图（--dataset_config nearby_only）同用** ——
    #   完全图上 GCN 一层即过平滑（真实 ckpt 实测 conv1 后节点间最大差 ~1e-7），
    #   GNN 分支的信息在池化之前就已销毁，融合层面的修复救不了它（单测判据 D/E 已固化该性质）。
    # 闸④（Q50）：切断 ego 捷径。`MagnetEgoShortcutSuppressed` 是 MagnetEgoQueryFusion 的子类
    #   ⇒ **必须排在 ego_query_fusion 之前**，否则带 shortcut 的 run 会被建成父类。
    #   ⚠️ §7：闸④ 是 PlanTF `2309.10443` 已发表贡献的移植，只作对照格，不得写成本文方法。
    # 闸⑥（Q70）：agent 预测辅助监督。`MagnetAgentAux` 直接继承 `TransformerPredictor`
    #   ⇒ 与上面几个 pooled 变体**互斥**（它们各自改了池化语义，混用会同时动两个变量，Q45 同类坑）。
    #   放在最前面判断并显式拒绝组合，而不是靠 elif 顺序"碰巧"排除。
    if args.joint_adaptive:
        # 🆕 路线 B 顶层分支。互斥已由 `validate_arch_exclusivity` 在 train() 第一行统一校验。
        #    🔴 判断顺序与 `bn_calibration.build_model` 一致：joint_adaptive **排在最前**。
        from magnet_joint_adaptive import MagnetJointAdaptive
        model = MagnetJointAdaptive(num_modes=args.num_modes,
                                    n_mp_layers=args.n_mp_layers,
                                    learn_topology=not args.no_learn_topology).to(device)
        if args.scorer_init:
            # S5：用 S4 训好的 scorer 作初始化（Q110 §6 的 S4→S5 次序）。
            # 🔴 只搬 `stack.scorer.*` 这一支，其余权重保持随机初始化 —— 搬多了就
            #    不是「打开学习边权重训」而是「从 teacher 微调」，两件事的对照含义不同。
            _obj = torch.load(args.scorer_init, map_location="cpu")
            _sd = _obj.get("model_state_dict", _obj)
            _pick = {k[len("stack.scorer."):]: v for k, v in _sd.items()
                     if k.startswith("stack.scorer.")}
            if not _pick:
                raise SystemExit(f"{args.scorer_init} 里没有 stack.scorer.* 权重 —— 拒绝静默继续")
            _missing = model.stack.scorer.load_state_dict(_pick, strict=True)
            logger.info("🔑 scorer 初始化自 %s（%d 个张量，strict=True）",
                        args.scorer_init, len(_pick))
    elif args.agent_aux:
        conflict = [n for n, v in (("--gnn_pooling_fix", args.gnn_pooling_fix),
                                   ("--context_pooling", args.context_pooling),
                                   ("--ego_query_fusion", args.ego_query_fusion),
                                   ("--ego_shortcut_suppress", args.ego_shortcut_suppress)) if v]
        if conflict:
            raise SystemExit(
                f"--agent_aux 不得与 {', '.join(conflict)} 同用：闸⑥ 的对照基线是 L1 recipe "
                f"（TransformerPredictor + gnn_pooling_fix=false），叠加其它开关会让"
                f"「+aux 真 vs 基线」差两个变量，归因无从谈起。")
        from magnet_agent_aux import MagnetAgentAux
        model = MagnetAgentAux(agent_aux=True).to(device)
    elif args.head_norm:
        # 修 A（Q105）：输出头前置 LayerNorm。继承 MagnetImitationPooled ⇒ 隐含逐样本池化，
        #   与 fr_base（class=MagnetImitationPooled, gnn_pooling_fix=true）**恰差一个变量**。
        #   🔴 与上面几个旋钮互斥：它们各自动了池化/融合/算子语义，混用会同时动两个变量。
        conflict = [n for n, v in (("--context_pooling", args.context_pooling),
                                   ("--ego_query_fusion", args.ego_query_fusion),
                                   ("--ego_shortcut_suppress", args.ego_shortcut_suppress),
                                   ("--gnn_operator", args.gnn_operator),
                                   ("--agent_aux", args.agent_aux)) if v]
        if conflict:
            raise SystemExit(
                f"--head_norm 不得与 {', '.join(conflict)} 同用：修 A 的对照基线是 fr_base "
                f"（MagnetImitationPooled + gnn_pooling_fix=true），叠加其它开关会让"
                f"「+head_norm vs 基线」差两个变量，归因无从谈起。")
        from magnet_head_norm import MagnetHeadNorm
        model = MagnetHeadNorm().to(device)
    elif args.ego_shortcut_suppress:
        from magnet_ego_shortcut import MagnetEgoShortcutSuppressed
        model = MagnetEgoShortcutSuppressed(context_pooling=args.context_pooling,
                                            context_pool_mode=args.context_pool_mode,
                                            ego_query_fusion=args.ego_query_fusion,
                                            ego_query_heads=args.ego_query_heads,
                                            use_ego_history=False,
                                            ego_state_dropout=args.ego_state_dropout).to(device)
    elif args.ego_query_fusion:
        from magnet_ego_query_fusion import MagnetEgoQueryFusion
        model = MagnetEgoQueryFusion(context_pooling=args.context_pooling,
                                     context_pool_mode=args.context_pool_mode,
                                     ego_query_fusion=True,
                                     ego_query_heads=args.ego_query_heads).to(device)
    elif args.context_pooling:
        from magnet_context_pooled import MagnetContextPooled
        model = MagnetContextPooled(context_pooling=True,
                                    context_pool_mode=args.context_pool_mode).to(device)
    elif args.gnn_operator:
        # 旋钮 B（D5）：继承 MagnetImitationPooled ⇒ 隐含逐样本池化，与 K0 恰差一个变量（算子）
        from magnet_gnn_operator import MagnetGNNOperator
        model = MagnetGNNOperator(operator=args.gnn_operator).to(device)
    elif args.gnn_pooling_fix:
        from magnet_imitation_pooled import MagnetImitationPooled
        model = MagnetImitationPooled().to(device)
    else:
        model = TransformerPredictor().to(device)

    # R1（Q76 重构）：逐帧场景图 + 跨帧时序 Transformer。
    # 🔴 放在最后覆盖，且与上面几个旋钮互斥 —— 混用会让「唯一变量」这个前提失效。
    if args.temporal_graph:
        # 互斥已由 `validate_arch_exclusivity` 在 train() 第一行统一校验（_ARCH_EXCLUSIVE）
        from temporal_predictor import TemporalPredictor
        model = TemporalPredictor().to(device)

    # 多模态输出头（Q77）：与 R1 同样放在最后覆盖，且与**全部**建类旋钮互斥。
    # 🔴 与对照格 s4b_magnet_L1（class=TransformerPredictor、开关全 false）的唯一差别
    #    必须只是输出头 —— 叠任何一个旋钮都会变成两个变量。
    if args.multimodal:
        # 互斥已由 `validate_arch_exclusivity` 在 train() 第一行统一校验（_ARCH_EXCLUSIVE）
        from magnet_multimodal import MagnetImitationMultiModal
        model = MagnetImitationMultiModal(num_modes=args.num_modes).to(device)

    logger.info("model <- %s (gnn_pooling_fix=%s/eff=%s, gnn_operator=%s, "
                "context_pooling=%s/%s, ego_query_fusion=%s/%s heads, "
                "ego_shortcut_suppress=%s/dropout=%s)",
                type(model).__name__, args.gnn_pooling_fix, gnn_pooling_fix_effective,
                args.gnn_operator, args.context_pooling, args.context_pool_mode,
                args.ego_query_fusion, args.ego_query_heads,
                args.ego_shortcut_suppress, args.ego_state_dropout)
    # ── S5 边监督项 ────────────────────────────────────────────────────────────────
    edge_rank = edge_rank_val = None
    if args.edge_rank_loss:
        if not args.joint_adaptive:
            raise SystemExit("--edge_rank_loss 只在 --joint_adaptive 下有意义")
        if args.no_learn_topology:
            raise SystemExit("--edge_rank_loss 与 --no_learn_topology 冲突："
                             "固定臂 w≡1、没有 scorer 前向，边监督项无处可加")
        if args.oracle_gate:
            raise SystemExit("--edge_rank_loss 与 --oracle_gate 冲突："
                             "前者学边权、后者注入 GT 边权，同时开等于同时动两个变量")
        from magnet_joint_loss import EdgeRankLoss
        edge_rank = EdgeRankLoss(seed=args.seed, device=device)
        edge_rank_val = EdgeRankLoss(seed=args.seed + 1, device=device)
        logger.info("🆕 S5 边监督项开启（λ 首 batch 量级标定；val 侧只作诊断，不进控制信号）")

    criterion: nn.Module = ImprovedTrajectoryLoss()
    if args.joint_adaptive:
        from magnet_joint_loss import JointWTALoss
        criterion = JointWTALoss(mode_weight=args.mode_weight,
                                 scene_level=not args.joint_per_entity)
        logger.info("🆕 joint 开启：K=%d，MP 层数=%d，learn_topology=%s，WTA=%s，mode_weight=%s\n"
                    "   🔴 `val_loss` 与任何旧格**不可并列**（位置域 + 联合 + WTA，三处口径都不同）；\n"
                    "   🔴 参数量 %d（旧 MAGNET 779,850）",
                    args.num_modes, args.n_mp_layers, not args.no_learn_topology,
                    "per-entity" if args.joint_per_entity else "scene-level", args.mode_weight,
                    sum(p.numel() for p in model.parameters()))
    elif args.multimodal:
        # 🔴 判据落笔在见任何数字之前：默认 `wta` —— 它与 PlanTF 的 ego 损失**逐项对应**
        #    （`lightning_trainer.py:94-95`：smooth_l1(best_traj) + cross_entropy(prob, best_mode)），
        #    故本格可以写成「把 PlanTF 的输出形式化移植过来」，而不是我方新提的损失。
        #    `soft` 是旧路线B 留下的变体，作为探索档保留，**默认不用**（换它就是换两个变量）。
        from magnet_multimodal import MultiModalSoftLoss, MultiModalWTALoss
        criterion = (MultiModalSoftLoss(mode_weight=args.mode_weight)
                     if args.multimodal_loss == "soft"
                     else MultiModalWTALoss(mode_weight=args.mode_weight))
        logger.info("多模态开启：K=%d，loss=%s，mode_weight=%s ⇒ val_loss 与单模态基线**不可并列**"
                    "（WTA 是 K 个里取最优，天然更低）",
                    args.num_modes, args.multimodal_loss, args.mode_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.9, 0.999))
    # 🔴🔴 D0#19（08-05 查实）：**训练控制信号必须与关心的指标同向**。
    #
    # 事故：多模态（Q77）的 val_loss = WTA 最优模回归 + 选模 CE。训练中 CE 项的增长
    # 盖过回归项的下降 ⇒ **val_loss 一路上升**（1.60→2.25）而 ADE 一路下降（4.64→3.05）。
    # 当时三处控制信号全部挂在 val_loss 上，于是同时出了三件事：
    #   ① checkpoint 选择 —— best 落在 **epoch 1/1/4**，保存的是几乎没训练的模型；
    #   ② LR 调度 —— ReduceLROnPlateau 认为一直没改善，把 LR 砍了 51×/51×/35×
    #      （同期单模态基线只砍 8×/3×/12×）⇒ 多模态的训练被系统性削弱；
    #   ③ early stopping —— 本次 patience=999 侥幸没触发，否则会在 epoch ~2 停掉。
    # 代价：一次 5 小时的闭环跑在 epoch-1 模型上，结论作废。
    #
    # ⇒ 三处信号**必须取同一个、且与指标同向的量**。单模态下 val_loss 与 ADE 同向，
    #   保持原样以保证既有 checkpoint 逐位可复现；多模态下改用 **top1 val_ade**。
    monitor_name = resolve_monitor_name(args.ckpt_metric, bool(args.multimodal),
                                        joint=bool(args.joint_adaptive))
    logger.info("🔑 训练控制信号 = %s（checkpoint 选择 / LR 调度 / early stopping 三处同源；"
                "--ckpt_metric=%s 解析所得）", monitor_name, args.ckpt_metric)
    if args.multimodal and monitor_name == "val_loss":
        logger.warning("🔴 多模态用 val_loss 作控制信号：它含选模 CE 项、与 ADE 不同向，"
                       "历史上导致 best checkpoint 落在 epoch 1。除非你明确要复现旧行为，否则别这么跑")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3, min_lr=1e-6)
    stopper = EarlyStopping(patience=args.patience)

    # 闸⑥ 的 loss 驱动器（None = 关闭 ⇒ run_epoch 走与基线逐字节相同的路径）
    aux_cfg = None
    if args.agent_aux:
        aux_cfg = AgentAuxConfig(shuffle=args.agent_aux_shuffle,
                                 weight=args.agent_aux_weight, seed=args.seed)
        logger.info("闸⑥ 开启：shuffle=%s weight=%s（aux loss 只进 train 反传，不进 val_loss）",
                    args.agent_aux_shuffle, args.agent_aux_weight)

    # P5（Q110）：margin 与 funnel 余弦进训练期常规监控，每 epoch 记 p1。
    # 🔴 不是锦上添花 —— 阶段299 查实**损失里没有任何一项惩罚塌陷** ⇒ margin 是自由漂移的副产品，
    #    没人盯就会漂到零点；而「前置 LayerNorm 数学上排除塌陷」**只在 β=0 的初始化时成立**。
    p5_monitor = None
    if args.joint_adaptive:
        from magnet_joint_loss import MarginFunnelMonitor
        p5_monitor = MarginFunnelMonitor(model.active_head())
        logger.info("P5 监控已挂：每 epoch 记 margin p1 / 塌陷率 / funnel 余弦 / 模态使用率")

    history = {"train_loss": [], "val_loss": [], "val_ade": [], "val_fde": [], "lr": [],
               "train_aux": [], "val_aux": []}
    best_val = float("inf")
    best_epoch = None
    best_saved = {"val_ade": None, "val_fde": None, "val_loss": None}
    best_bare = out_dir / "models" / "best_model_default.pth"   # 对齐 planner 加载名（bare state_dict）
    best_ckpt = out_dir / "models" / "checkpoint_best.pt"       # 富 checkpoint（可恢复）

    for epoch in range(args.epochs):
        t0 = time.time()
        mm_msg = ""
        if args.joint_adaptive:
            from magnet_joint_loss import run_epoch_joint  # noqa: WPS433
            tr = run_epoch_joint(model, train_loader, criterion, optimizer, device, train=True,
                                 ego_only=args.joint_ego_only, edge_rank=edge_rank)
            va = run_epoch_joint(model, val_loader, criterion, optimizer, device, train=False,
                                 monitor=p5_monitor, ego_only=args.joint_ego_only,
                                 edge_rank=edge_rank_val)
            tr_loss, tr_aux, va_aux = tr["loss"], 0.0, 0.0
            va_loss = va["loss"]
            # 🔴 控制信号取 `all_minADE`（ego+周车合并的 minADE@K），三处同源。
            #    ego 与周车分开的读数照记进 history，但**不作控制信号** —— 换来换去会让
            #    「best checkpoint 依据什么」在事后说不清（D0#19 的教训）。
            if "all_minADE" not in va:            # 整个 epoch 没有可监督实体 ⇒ 数据/开关配错了
                raise RuntimeError("val 侧没有任何可监督实体（`all_minADE` 缺失）—— "
                                   "检查 aux 缓存与 --joint_ego_only 组合，不得静默继续")
            va_ade, va_fde = va["all_minADE"], va["all_minFDE"]
            # 🔴 跳过 `loss` —— 下面的公共段已经写过 `history["val_loss"]/["train_loss"]`，
            #    这里再写一次会让 `epochs_run = len(history["train_loss"])` 翻倍（08-25 实测 3 → 6）。
            for k, v in va.items():
                if k != "loss":
                    history.setdefault(f"val_{k}", []).append(v)
            for k in ("reg", "mode_ce", "all_minADE"):
                history.setdefault(f"train_{k}", []).append(tr.get(k, float("nan")))
            # 🔴 全部走 `g()` —— `--joint_ego_only` 下 agent/moving/static 三层**是空的**，
            #    直取 `va['agent_missRate']` 会 KeyError 把整个训练打死（08-25 实测崩过一次）。
            g = lambda k: va.get(k, float("nan"))  # noqa: E731
            mm_msg = (f" ego={g('ego_minADE'):.3f}/{g('ego_minFDE'):.3f}"
                      f" top1ego={g('ego_top1ADE'):.3f}"
                      f" mov={g('moving_minADE'):.3f} sta={g('static_minADE'):.3f}"
                      f" miss={g('agent_missRate'):.3f}"
                      f" hit={g('all_hitRate'):.3f}(rand={1.0 / args.num_modes:.3f})"
                      # 🔴 modeMax/spread 必须分层：聚合值被 86% 静止车主导（08-25 P5 探针）
                      f" modeMax(e/m/s)={g('ego_mode_usage_max'):.2f}/"
                      f"{g('moving_mode_usage_max'):.2f}/{g('static_mode_usage_max'):.2f}"
                      f" spread(e/m/s)={g('ego_mode_spread'):.2f}/"
                      f"{g('moving_mode_spread'):.2f}/{g('static_mode_spread'):.2f}"
                      f" CE={g('mode_ce'):.4f}"
                      f" margin_p1={g('margin_p1'):.3f}"
                      f" collapse={g('collapse_rate'):.3f}")
        elif args.multimodal:
            # 局部 import：`magnet_multimodal` 反过来 import 本模块的共享件，顶层会成环。
            from magnet_multimodal import run_epoch_multimodal  # noqa: WPS433
            tr = run_epoch_multimodal(model, train_loader, criterion, optimizer, device, train=True)
            va = run_epoch_multimodal(model, val_loader, criterion, optimizer, device, train=False)
            tr_loss, tr_aux, va_aux = tr["loss"], 0.0, 0.0
            va_loss, va_ade, va_fde = va["loss"], va["ade"], va["fde"]
            history.setdefault("val_min_ade", []).append(va["min_ade"])
            history.setdefault("val_min_fde", []).append(va["min_fde"])
            history.setdefault("val_hit_rate", []).append(va["hit_rate"])
            history.setdefault("train_hit_rate", []).append(tr["hit_rate"])
            # minADE/minFDE 只作监控 —— 带 oracle 成分，不得与单模态基线并列。
            # hit/gap 是**诊断**用：hit 应从 1/K=0.167 往上走、gap 应从 ~K 的水平往 1 收；
            # 两者一直不动就是选模没学会，与「候选集不好」是两回事（见 min_ade_fde docstring）。
            mm_msg = (f" minADE={va['min_ade']:.3f}/minFDE={va['min_fde']:.3f}"
                      f" hit={va['hit_rate']:.3f}(rand={1.0 / args.num_modes:.3f})"
                      f" gap={va['ade'] / max(va['min_ade'], 1e-9):.2f}x")
        else:
            tr_loss, _, _, tr_aux = run_epoch(model, train_loader, criterion, optimizer, device,
                                              train=True, aux_cfg=aux_cfg)
            va_loss, va_ade, va_fde, va_aux = run_epoch(model, val_loader, criterion, optimizer,
                                                        device, train=False, aux_cfg=aux_cfg)
        monitor = va_ade if monitor_name == "val_ade" else va_loss
        scheduler.step(monitor)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_ade"].append(va_ade)
        history["val_fde"].append(va_fde)
        history["lr"].append(lr)
        history["train_aux"].append(tr_aux)
        history["val_aux"].append(va_aux)
        aux_msg = f" aux={tr_aux:.4f}/{va_aux:.4f}" if aux_cfg is not None else ""
        logger.info("epoch %3d/%d | train=%.4f val=%.4f ADE=%.3fm FDE=%.3fm lr=%.2e%s%s (%.1fs)",
                    epoch + 1, args.epochs, tr_loss, va_loss, va_ade, va_fde, lr, aux_msg, mm_msg,
                    time.time() - t0)

        if monitor < best_val:
            best_val = monitor
            best_epoch = epoch + 1          # 1-based，写进 loss_record 供审计
            best_saved = {"val_ade": va_ade, "val_fde": va_fde, "val_loss": va_loss}
            torch.save(model.state_dict(), best_bare)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "monitor_name": monitor_name, "monitor_value": best_val,
                        "best_val_loss": va_loss,       # 该 epoch 的 val_loss（不再是被监控量）
                        "val_ade": va_ade, "val_fde": va_fde, "seed": args.seed,
                        "dataset_config": args.dataset_config, "oracle_gate": args.oracle_gate, "env": env,
                        "edge_rank_loss": bool(args.edge_rank_loss),
                        "edge_rank_lambda": (edge_rank.lam if edge_rank is not None else None),
                        "scorer_init": args.scorer_init,
                        "gnn_pooling_fix": bool(args.gnn_pooling_fix),
                        "gnn_pooling_fix_effective": gnn_pooling_fix_effective,
                        "head_norm": bool(args.head_norm),
                        "gnn_operator": args.gnn_operator,
                        "subgraph_agg": args.subgraph_agg,
                        "context_pooling": bool(args.context_pooling),
                        "context_pool_mode": args.context_pool_mode,
                        "ego_query_fusion": bool(args.ego_query_fusion),
                        "ego_query_heads": args.ego_query_heads,
                        "ego_shortcut_suppress": bool(args.ego_shortcut_suppress),
                        "ego_state_dropout": args.ego_state_dropout,
                        "agent_aux": bool(args.agent_aux),
                        "agent_aux_shuffle": bool(args.agent_aux_shuffle),
                        "agent_aux_weight": args.agent_aux_weight,
                        "multimodal": bool(args.multimodal),
                        "num_modes": args.num_modes,
                        "multimodal_loss": args.multimodal_loss,
                        "joint_adaptive": bool(args.joint_adaptive),
                        "learn_topology": (not args.no_learn_topology) if args.joint_adaptive else None,
                        "n_mp_layers": args.n_mp_layers if args.joint_adaptive else None,
                        "joint_ego_only": bool(args.joint_ego_only),
                        "joint_wta": ("per_entity" if args.joint_per_entity
                                      else "scene_level")}, best_ckpt)

        if stopper(monitor, model):
            logger.info("early stopping @ epoch %d", epoch + 1)
            break

    if p5_monitor is not None:
        p5_monitor.close()

    # 收尾产物
    torch.save(model.state_dict(), out_dir / "models" / "final_model_default.pth")
    record = {
        "n_train": n_train, "n_val": n_val, "epochs_run": len(history["train_loss"]),
        # 🔴 D0#19 语义修复（08-05）：旧版 `best_val_ade` 写的是 **min(history)**，
        # 与实际保存的 checkpoint **无关** —— 它掩盖了「best 落在 epoch 1」这个事故
        # （旧记录显示 best_val_ade=2.9736，而那个 ckpt 实测 4.6363）。
        # 现在这三个字段一律指**被保存的那个 checkpoint 的**值；曲线最小值另开字段。
        "ckpt_metric": monitor_name,
        "best_epoch": best_epoch,                       # 1-based
        "best_monitor_value": best_val,
        "best_val_loss": best_saved["val_loss"],
        "best_val_ade": best_saved["val_ade"],
        "best_val_fde": best_saved["val_fde"],
        # 曲线最小值（＝旧版 `best_val_ade` 的真实含义），保留但正名
        "min_val_ade_over_history": min(history["val_ade"]) if history["val_ade"] else None,
        "min_val_fde_over_history": min(history["val_fde"]) if history["val_fde"] else None,
        "train_root": args.train_root, "val_root": args.val_root,
        "dataset_config": args.dataset_config, "oracle_gate": args.oracle_gate,
        "seed": args.seed, "lr": args.lr,
        "batch_size": args.batch_size, "env": env, "history": history,
        "route_mode": args.route_mode,
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),   # D0#7：命令行原始 flag
        # 🔴 GNN 分支语义的**唯一可信依据**：True=逐样本 global_mean_pool，
        #    False=整批塌成一向量再 repeat（transformer_predictor.py:270-271）。
        "gnn_pooling_fix_effective": gnn_pooling_fix_effective,
        # 旋钮 B：区分 GNN 算子格（None=GCNConv 父类实现 / gatv2 / transformer）的唯一依据
        "gnn_operator": args.gnn_operator,
        # 旋钮 A：区分建图时间池化聚合格（mean=旧行为 / max / sum）的唯一依据
        "subgraph_agg": args.subgraph_agg,
        "context_pooling": bool(args.context_pooling),   # 判据0：区分池化线的唯一依据
        "context_pool_mode": args.context_pool_mode,
        "ego_query_fusion": bool(args.ego_query_fusion), # C2：区分融合算子的唯一依据
        "ego_query_heads": args.ego_query_heads,
        "ego_shortcut_suppress": bool(args.ego_shortcut_suppress),  # 闸④：区分 ego 通路的唯一依据
        "ego_state_dropout": args.ego_state_dropout,
        "agent_aux": bool(args.agent_aux),               # 闸⑥：区分三格（基线/真/打乱）的唯一依据
        "agent_aux_shuffle": bool(args.agent_aux_shuffle),
        "agent_aux_weight": args.agent_aux_weight,
        # Q77 多模态：`best_val_loss` 在本格是 WTA 损失，**与单模态格不同量纲** ⇒
        #   这三个字段是事后判断「这个 loss_record 能不能和别的并列」的唯一依据。
        "multimodal": bool(args.multimodal),
        "num_modes": args.num_modes,
        "multimodal_loss": args.multimodal_loss,
        # 🆕 路线 B（Q108/Q110/Q111）：区分 joint 格与任何旧格的唯一依据。
        #   🔴 `best_val_loss` 在本格是**位置域联合 WTA 损失**，与旧格不同量纲、不可并列。
        "joint_adaptive": bool(args.joint_adaptive),
        "learn_topology": (not args.no_learn_topology) if args.joint_adaptive else None,
        "n_mp_layers": args.n_mp_layers if args.joint_adaptive else None,
        "joint_ego_only": bool(args.joint_ego_only), "joint_wta": ("per_entity" if args.joint_per_entity else "scene_level")
                     if args.joint_adaptive else None,
        "perturbation": {
            "enabled": bool(args.use_perturbation),
            "prob": args.perturb_prob, "max_lateral": args.max_lateral,
            "max_longitudinal": args.max_longitudinal, "max_yaw": args.max_yaw,
            "seed": (args.perturb_seed if args.perturb_seed is not None else args.seed),
        },
    }
    with open(out_dir / "loss_record.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # checkpoint sidecar：planner 加载时据此校验 route_mode 一致性（阶段⑬）。
    # 放在 models/ 下与 best_model_default.pth 同目录，planner 按 dirname(ckpt)/train_meta.json 找。
    train_meta = {
        "route_mode": args.route_mode,
        "perturbation": record["perturbation"],
        "seed": args.seed,
        "dataset_config": args.dataset_config,
        "oracle_gate": args.oracle_gate,
        # S5 provenance：产物必须能自证边监督项开过、λ 标到了多少、scorer 从哪来。
        # 🔴 λ 是首个 batch 量级标定的结果（不是超参），只有落盘才可复现审计。
        "edge_rank_loss": bool(args.edge_rank_loss),
        "edge_rank_lambda": (edge_rank.lam if edge_rank is not None else None),
        "scorer_init": args.scorer_init,
        "train_root": args.train_root,
        "val_root": args.val_root,
        "best_val_loss": best_val,
        "env": env,
        # 🔴 架构块：**推理侧重建模型的唯一依据**。
        #   calibrated 产物是 bare state_dict（无 provenance），而 `ego_query_heads` 这类参数
        #   **无法从权重形状反推**（MultiheadAttention 的 in_proj_weight 形状与 heads 无关）——
        #   少了它 `load_state_dict` 仍会成功，但模型算的是另一回事，属静默错模型。
        #   `bn_calibration.build_model` 按本块重建；改字段名要两边同步。
        "architecture": {
            "class": type(model).__name__,
            "gnn_pooling_fix": bool(args.gnn_pooling_fix),
            "gnn_pooling_fix_effective": gnn_pooling_fix_effective,
            # 修 A（Q105）：`MagnetHeadNorm` 比基线多 `output_head.5.*` 且末层下标 6→7 ⇒
            #   推理侧不按本字段建类，strict load 会崩（有意，与 agent_aux/gnn_operator 同构）。
            #   `bn_calibration.ARCH_BUILD_KEYS` 已同步加入 `head_norm`。
            "head_norm": bool(args.head_norm),
            # 旋钮 B：`MagnetGNNOperator` 的 gnn_encoder.* 权重键名与父类不同 ⇒
            #   推理侧不按本字段建类，strict load 会崩（有意，与 agent_aux 同构）。
            "gnn_operator": args.gnn_operator,
            # 旋钮 A：建图侧变量（不改权重形状）。审计/在线侧必须按本字段选同款聚合，
            #   否则建出的图与训练所见不同（run_interaction_audit 已接线；planner 侧待闭环时接）
            "subgraph_agg": args.subgraph_agg,
            "context_pooling": bool(args.context_pooling),
            "context_pool_mode": args.context_pool_mode,
            "ego_query_fusion": bool(args.ego_query_fusion),
            "ego_query_heads": args.ego_query_heads,
            # 闸④：`ego_state_dropout` 与 heads 同理无法从权重形状反推（只影响训练期采样），
            #   但 `build_model` 的构造签名需要它，缺了会用默认值建出「另一个模型」的 provenance。
            "ego_shortcut_suppress": bool(args.ego_shortcut_suppress),
            "ego_state_dropout": args.ego_state_dropout,
            # 闸⑥：`MagnetAgentAux` 比父类多 `agent_future_head.*` 三组权重 ⇒
            #   推理侧不按本字段建类，strict load 会直接崩（这是有意的，见模块 docstring）。
            #   `bn_calibration.ARCH_BUILD_KEYS` 已同步加入 `agent_aux`。
            "agent_aux": bool(args.agent_aux),
            # 下面两个不影响建类，只影响训练期行为；写进来是为了让三格 provenance 自洽
            #   （`+aux(真)` 与 `+aux(打乱)` 的权重形状完全相同，事后只能靠 sidecar 区分）。
            "agent_aux_shuffle": bool(args.agent_aux_shuffle),
            "agent_aux_weight": args.agent_aux_weight,
            # R1（Q76）：`TemporalPredictor` 比父类多 frame_pos_embedding.* / temporal_transformer.*
            #   ⇒ 推理侧不按本字段建类，strict load 会崩。`bn_calibration.ARCH_BUILD_KEYS` 已同步。
            # 🔴 `temporal_topology` 不进 build_model（它是**数据侧**参数，决定每帧的边），
            #   但必须落 provenance：审计侧要用同一个拓扑重建图，否则测的是另一张图。
            "temporal_graph": bool(args.temporal_graph),
            "temporal_topology": args.temporal_topology,
            # Q77：`MagnetImitationMultiModal` 比父类多 traj_head.* / mode_head.*，
            #   且 `num_modes` 决定 traj_head 末层 out_features ⇒ 建错就 strict load 崩（有意）。
            #   `bn_calibration.ARCH_BUILD_KEYS` 已同步加入两者。
            "multimodal": bool(args.multimodal),
            "num_modes": args.num_modes,
            # 不影响建类，只影响训练期损失；写进来是为了让 wta/soft 两格 provenance 自洽
            #   （两者权重形状完全相同，事后只能靠 sidecar 区分）。
            "multimodal_loss": args.multimodal_loss,
            # 🆕 路线 B（Q108/Q110/D-e）：`MagnetJointAdaptive` **不在** TransformerPredictor 家族里，
            #   权重键名与所有旧类都不同 ⇒ 少了这三个字段，推理侧会退回父类并 strict load 崩。
            #   🔴 `bn_calibration.ARCH_BUILD_KEYS` 已同步加入三者，两处不得分叉。
            #   ⚠️ P4 之后 ego 编码器是 LayerNorm ⇒ **BN calibration 对本类不适用也不需要**。
            "joint_adaptive": bool(args.joint_adaptive),
            "learn_topology": (not args.no_learn_topology) if args.joint_adaptive else None,
            "n_mp_layers": args.n_mp_layers if args.joint_adaptive else None,
            "joint_ego_only": bool(args.joint_ego_only),
            "joint_wta": ("per_entity" if args.joint_per_entity else "scene_level"),
        },
    }
    with open(out_dir / "models" / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)
    logger.info("train_meta.json 已写出（route_mode=%s, perturbation=%s）",
                args.route_mode, record["perturbation"]["enabled"])
    logger.info("训练完成：best_val=%.4f，best ADE=%.3fm FDE=%.3fm",
                best_val, record["best_val_ade"] or -1, record["best_val_fde"] or -1)
    if best_epoch is not None and best_epoch <= 3 and len(history["val_ade"]) > 10:
        logger.error("🔴🔴 best checkpoint 落在 epoch %d/%d（跑了 %d 个 epoch）—— 极可能是"
                     "控制信号与指标不同向（D0#19）。**不要拿这个 ckpt 去跑闭环/审计**，"
                     "先看 loss_record 的 ckpt_metric 与 val_ade 曲线。",
                     best_epoch, len(history["val_ade"]), len(history["val_ade"]))
    logger.info("checkpoint: %s（planner bare）/ %s（富）| ckpt_metric=%s best_epoch=%s "
                "best_val_ade=%.4f（该 ckpt 实际值，非曲线最小值）",
                best_bare, best_ckpt, monitor_name, best_epoch,
                best_saved["val_ade"] if best_saved["val_ade"] is not None else float("nan"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 MAGNET imitation trainer（route B，不碰 danger 线）")
    p.add_argument("--train_root", required=True, type=str, help="log-disjoint split 的 train leaf root（绝对路径）")
    p.add_argument("--val_root", required=True, type=str, help="log-disjoint split 的 val leaf root（绝对路径）")
    p.add_argument("--output_dir", required=True, type=str, help="输出目录（绝对路径）")
    p.add_argument("--ckpt_metric", default="auto", choices=["auto", "val_loss", "val_ade"],
                   help="训练控制信号（**同时**决定 checkpoint 选择 / LR 调度 / early stopping）。\n"
                        "`auto`（默认）= 多模态用 `val_ade`、单模态用 `val_loss`。\n"
                        "🔴 单模态保持 `val_loss` 是为了让既有 checkpoint **逐位可复现**；\n"
                        "🔴 多模态**必须**用 `val_ade` —— 其 val_loss 含选模 CE 项、与 ADE 反向，\n"
                        "  08-05 实测导致 best checkpoint 落在 epoch 1、LR 被多砍 4–17 倍。")
    p.add_argument("--dataset_config", default="default", help="邻接矩阵配置（对齐 planner best_model_default.pth）")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=15, help="early stopping patience")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # --- route_mode（须与预处理产物一致；写入 train_meta.json 供 planner 校验）---
    p.add_argument("--route_mode", default="none", choices=["none", "three_channel"],
                   help="训练数据的 route 注入模式；必须与 train_root 的预处理产物一致")
    # --- perturbation 增广（默认关闭，保持旧训练命令兼容）---
    p.add_argument("--use_perturbation", action="store_true", help="启用 ego 状态扰动增广（仅 train）")
    p.add_argument("--perturb_prob", type=float, default=0.5, help="单样本施加扰动的概率")
    p.add_argument("--max_lateral", type=float, default=1.0, help="[m] 横向偏移上界")
    p.add_argument("--max_longitudinal", type=float, default=0.5, help="[m] 纵向偏移上界")
    p.add_argument("--max_yaw", type=float, default=0.1, help="[rad] 朝向偏移上界")
    p.add_argument("--perturb_seed", type=int, default=None,
                   help="扰动采样器 seed；默认取实验 --seed，保证同 seed 可复现")
    p.add_argument("--head_norm", action="store_true",
                   help="修 A（Q105）：输出头第二个 ReLU 前补 LayerNorm，消掉塌陷的共模门。"
                        "继承 MagnetImitationPooled ⇒ 隐含逐样本池化，与 fr_base 恰差一个变量。"
                        "state_dict 与基线故意不兼容（多 output_head.5.* 且末层下标 6→7），"
                        "建错类时 strict load 会响亮报错。")
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="D0#7：GNN 分支逐样本池化（默认关闭 = 旧行为，逐字节一致）")
    p.add_argument("--gnn_operator", default=None, choices=["gatv2", "transformer"],
                   help="旋钮 B（D5/D6）：GNN 分支换算子 GCNConv→GATv2Conv（主格）/"
                        "TransformerConv（第二轮对照）。走 MagnetGNNOperator 子类，"
                        "🔴 隐含 gnn_pooling_fix=True（阶段123：旋钮实验必须开 fix，"
                        "与 K0 基线恰差一个变量）")
    p.add_argument("--subgraph_agg", default="mean", choices=["mean", "max", "sum"],
                   help="旋钮 A（D-2，判据 20260810）：subgraph_pooling 沿时间轴的聚合统计量。"
                        "默认 mean = 旧行为逐位不变（复用现有 .pt 缓存）；max/sum 会另建"
                        "带后缀的 .pt 缓存（杜绝静默复用 mean 缓存）。建图侧变量，不换模型类")
    # 🆕🆕 路线 B（Q108 / Q110 / Q111）：联合多智能体多模态预测 + 自适应交互拓扑。
    p.add_argument("--joint_adaptive", action="store_true",
                   help="🔴 顶层分支不是旋钮（D-e）：`MagnetJointAdaptive` —— per-entity token + "
                        "relation-aware MP + edge scorer + joint 多模态 decoder。"
                        "整条预测路径无 global pooling（P1），ego 编码器用 LayerNorm（P4，BN calibration 不再需要）。"
                        "与全部旧建类旋钮互斥。需要 agent aux 缓存（周车 target）。")
    p.add_argument("--no_learn_topology", action="store_true",
                   help="固定/完全图对照臂：边权 w≡1。🔴 scorer 仍被实例化 ⇒ 与 learned 臂参数量逐位相同（Q110 §4-1）")
    p.add_argument("--n_mp_layers", type=int, default=2,
                   help="relation-aware message passing 层数（Q110 §3.3 定 2；改它是换变量）")
    p.add_argument("--edge_rank_loss", action="store_true",
                   help="S5：把 S3 边效用的 within-target 排序损失加进联合损失（λ 按首个 batch "
                        "量级标定，不调参）。🔴 不开它，EdgeScorer 会退化成普通 attention 门，"
                        "R4-2 要的 'learn an appropriate topology' 就答不了。")
    p.add_argument("--scorer_init", default=None,
                   help="S5：用 S4 训好的 checkpoint 初始化 `stack.scorer`（只搬这一支）。")
    p.add_argument("--joint_ego_only", action="store_true",
                   help="🔴 Q110 §6.2 的 **A 臂**：结构不动、只监督 ego（= 单 ego 输出任务）。"
                        "与 B 臂（默认）的差别是**单变量 = 任务**：模态数/参数量/数据/锚点全同。"
                        "🔴 判别读法两条已事前写死（§6.2）：A≫旧 且 B≈A ⇒ 路线 B 前提不成立；"
                        "A≈旧 且 B≫A ⇒ 前提成立。不得事后择一。")
    p.add_argument("--joint_per_entity", action="store_true",
                   help="🔴 WTA 改成 per-entity（每个实体各选各的模态）。默认 scene-level = "
                        "一个模态代表整个场景的联合假设。换成 per-entity 后正文**不得**称 joint/联合预测（Q113）")
    p.add_argument("--multimodal", action="store_true",
                   help="Q77：多模态输出头（K 条候选 + 置信度，WTA 损失），直击单模态的 mode averaging。"
                        "与**全部**其它建类旋钮互斥 —— 对照格 s4b_magnet_L1 是纯父类，"
                        "本格与它的唯一差别必须只是输出头。"
                        "⚠️ §7：多模态 + WTA 是 PlanTF/VectorNet/MTR 等的既有形式化，**零 novelty**，"
                        "论文里只能作能力对照格，不得写成本文提出的方法。"
                        "🔴 val_loss 与单模态基线**不可并列**（WTA 是 K 个取最优，天然更低）；"
                        "可并列的是 top1 口径的 val_ade/val_fde（闭环拿到的就是 top1）")
    p.add_argument("--num_modes", type=int, default=6,
                   help="多模态候选数 K，默认 6（与 PlanTF `planning_model.py:31` 同）。"
                        "⚠️ 与 ego_query_heads 同理**无法从权重形状之外的地方反推口径**，须进 provenance")
    p.add_argument("--multimodal_loss", default="wta", choices=["wta", "soft"],
                   help="wta（默认）= WTA 回归 + argmin 硬标签 CE，与 PlanTF 的 "
                        "ego_reg_loss + ego_cls_loss 逐项对应；soft = 旧路线B 的软标签变体（探索档）")
    p.add_argument("--mode_weight", type=float, default=1.0,
                   help="选模损失权重（PlanTF 的 ego_cls_loss 权重为 1.0，默认与之一致）")
    p.add_argument("--context_pooling", action="store_true",
                   help="判据0：context 分支 polyline 池化（VectorNet 式 encode-then-pool）。"
                        "全模型 FLOPs 7986->50 MFLOP（159×）。隐含开启 gnn_pooling_fix 的 follow_batch。"
                        "⚠️ 与现有 12 格 ckpt 不可比，属独立实验线")
    p.add_argument("--context_pool_mode", default="max", choices=["max", "mean"],
                   help="polyline 池化方式；VectorNet 原文用 max（默认）")
    p.add_argument("--ego_query_fusion", action="store_true",
                   help="C2 靶向修复：以 ego 表征为 query 的注意力池化，替代 context/gnn 两处 "
                        "mean pooling（transformer_predictor.py:53 与 :239）。默认关闭时与父类逐字节一致。"
                        "🔴 必须与 --dataset_config nearby_only 同用：完全图上 GCN 一层即过平滑，"
                        "GNN 分支信息在池化前已销毁，融合修复救不了它。"
                        "⚠️ §7：attention 融合非本文提出，须显式引用 VectorNet/HDGT/PlanTF")
    p.add_argument("--ego_query_heads", type=int, default=4,
                   help="ego-query 注意力头数（默认 4）")
    p.add_argument("--ego_shortcut_suppress", action="store_true",
                   help="闸④（Q50）：切断 ego 捷径 —— ego 分支不读 EGO_HISTORY 的历史轨迹，"
                        "改由最后两帧增量导出的 6 通道当前状态 + state dropout 编码。"
                        "默认关闭时与 MagnetEgoQueryFusion 逐字节一致（单测判据 A）。"
                        "⚠️ §7：本闸是 PlanTF `2309.10443` 已发表贡献的移植，**零 novelty**，"
                        "论文里只能作 baseline/对照格，不得写成本文提出的方法。"
                        "⚠️ 打开后 ego 分支不再经过 ego_history_encoder 的 BatchNorm ⇒ "
                        "D0#13 的 batch 口径断层在本变体上自动消失，BN calibration 对 ego 分支成为空操作"
                        "（副作用，不是贡献）")
    p.add_argument("--ego_state_dropout", type=float, default=0.75,
                   help="闸④ 的 state dropout 概率（PlanTF 用 0.75）；仅 train 模式生效")
    p.add_argument("--oracle_gate", default=None,
                   choices=["oracle", "shuffled", "random"],
                   help="E1：注入 S3 边效用当逐边 `w`（Q108 §7 判据3 的 headroom 上界）。"
                        "缓存须先由 `build_oracle_gate_cache.py` 生成；缺缓存**直接报错不降级**。"
                        "🔴 只对 --joint_adaptive 有效；oracle 用了 GT 锚定标签 ⇒ 是上界不是可部署方法")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker 数。0=主进程（与历史实验逐字节同路径）；"
                        "R1 建议 4 —— 实测把逐帧建图的 CPU 开销吃掉，0.648→0.585 s/batch。")
    p.add_argument("--temporal_graph", action="store_true",
                   help="R1（Q76 重构）：把 GNN 分支换成**逐帧场景图 + 跨帧时序 Transformer**。"
                        "现有 subgraph_pooling 用零参数零学习的 np.mean 沿时间轴把 19 帧压成 1 "
                        "个节点（VectorNet 同一步是 MLP+max-pool）⇒ 送进 GNN 的是「20 帧平均场景」"
                        "而非「t 时刻场景」。零缓存路径，不重建 .pt。"
                        "🔴 与 --gnn_pooling_fix / --context_pooling / --ego_query_fusion 互斥。")
    p.add_argument("--temporal_topology", default="full", choices=["full", "knn30"],
                   help="R1 的每帧拓扑：full=完全图（每帧边相同）；"
                        "knn30=30 m 距离阈值（**每帧边不同**，agent 在移动）。")
    p.add_argument("--agent_aux", action="store_true",
                   help="闸⑥（Q70）：给每个 agent 节点接直接的未来轨迹监督，绕开 "
                        "global_mean_pool 的 4.65%% 稀释。需先跑 `build_agent_aux_cache.py` "
                        "为 train/val 两个 root 各生成一份缓存。"
                        "🔴 不得与 --gnn_pooling_fix/--context_pooling/--ego_query_fusion/"
                        "--ego_shortcut_suppress 同用（会同时动两个变量）。"
                        "⚠️ aux 目标是**绝对 ego-centric 位置**，与 ego 头的增量口径不同，"
                        "论文必须写明两头口径不同")
    p.add_argument("--agent_aux_shuffle", action="store_true",
                   help="闸⑥ 的对照格：batch 内跨 agent 重排 aux 目标（保容量与梯度总量、"
                        "只破坏信号）。⚠️ 训练初期 aux 头输出≈0 ⇒ smooth-L1 退化成 mean(|tgt|)、"
                        "对置换不变 ⇒ 与「真」格两条 loss 曲线会重合，这是预期不是接线错误；"
                        "**训到最后仍重合 ⇒ aux 头没学到东西，主判据不可解读**")
    p.add_argument("--agent_aux_weight", type=float, default=1.0,
                   help="闸⑥ aux loss 权重")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
