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
from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402
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
              "traj_id_to_indices", "cluster", "valid_len", "time_step_len"]
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
              optimizer, device: torch.device, train: bool) -> Tuple[float, float, float]:
    """跑一个 epoch（train=True 反传）；返回 (avg_loss, avg_ade, avg_fde)。"""
    model.train(train)
    total_loss, n = 0.0, 0
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
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * bs
            n += bs
            ade, fde = _ade_fde(output.detach(), y)
            ade_all.append(ade.cpu())
            fde_all.append(fde.cpu())

    avg_loss = total_loss / max(n, 1)
    avg_ade = float(torch.cat(ade_all).mean()) if ade_all else 0.0
    avg_fde = float(torch.cat(fde_all).mean()) if fde_all else 0.0
    return avg_loss, avg_ade, avg_fde


def build_loaders(train_root: str, val_root: str, batch_size: int,
                  dataset_config: str, train_transform=None,
                  gnn_pooling_fix: bool = False) -> Tuple[DataLoader, DataLoader, int, int]:
    """构建 train/val 两个 GraphDataset（各自独立 root，log-disjoint）。

    :param train_transform: 仅施加于 **train** 的在线增广（如 EgoPerturbation）。
        val 恒为 `transform=None` —— 增广绝不能污染模型选择依据（early stopping / 超参）。
    """
    get_adjacency_config(dataset_config)  # 设全局，GraphDataset 据此命名 .pt 并选 edge builder
    train_ds = GraphDataset(train_root, transform=train_transform)
    val_ds = GraphDataset(val_root, transform=None)   # ★ val 绝不增广
    logger.info("GraphDataset: train=%d, val=%d (adjacency=%s, train_transform=%s)",
                len(train_ds), len(val_ds), dataset_config, train_transform)
    # D0#7：follow_batch 产出 gnn_x_batch，供 MagnetImitationPooled 别名为 gnn_batch。
    # 不开修复时保持 None，DataLoader 行为与旧版逐字节相同。
    follow = ["gnn_x"] if gnn_pooling_fix else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, follow_batch=follow)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, follow_batch=follow)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def train(args: argparse.Namespace) -> None:
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

    train_loader, val_loader, n_train, n_val = build_loaders(
        args.train_root, args.val_root, args.batch_size, args.dataset_config,
        train_transform=perturb, gnn_pooling_fix=args.gnn_pooling_fix)

    # 生效副本自证（副本/sys.path 陷阱兜底）
    import inspect
    logger.info("TransformerPredictor <- %s", inspect.getfile(TransformerPredictor))
    logger.info("GraphDataset       <- %s", inspect.getfile(GraphDataset))

    # D0#7：逐样本池化修复（默认关闭 = 旧行为，父类 transformer_predictor.py 一字未改）
    if args.gnn_pooling_fix:
        from magnet_imitation_pooled import MagnetImitationPooled
        model = MagnetImitationPooled().to(device)
    else:
        model = TransformerPredictor().to(device)
    logger.info("model <- %s (gnn_pooling_fix=%s)", type(model).__name__, args.gnn_pooling_fix)
    criterion = ImprovedTrajectoryLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3, min_lr=1e-6)
    stopper = EarlyStopping(patience=args.patience)

    history = {"train_loss": [], "val_loss": [], "val_ade": [], "val_fde": [], "lr": []}
    best_val = float("inf")
    best_bare = out_dir / "models" / "best_model_default.pth"   # 对齐 planner 加载名（bare state_dict）
    best_ckpt = out_dir / "models" / "checkpoint_best.pt"       # 富 checkpoint（可恢复）

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_ade, va_fde = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(va_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_ade"].append(va_ade)
        history["val_fde"].append(va_fde)
        history["lr"].append(lr)
        logger.info("epoch %3d/%d | train=%.4f val=%.4f ADE=%.3fm FDE=%.3fm lr=%.2e (%.1fs)",
                    epoch + 1, args.epochs, tr_loss, va_loss, va_ade, va_fde, lr, time.time() - t0)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_bare)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val,
                        "val_ade": va_ade, "val_fde": va_fde, "seed": args.seed,
                        "dataset_config": args.dataset_config, "env": env,
                        "gnn_pooling_fix": bool(args.gnn_pooling_fix)}, best_ckpt)

        if stopper(va_loss, model):
            logger.info("early stopping @ epoch %d", epoch + 1)
            break

    # 收尾产物
    torch.save(model.state_dict(), out_dir / "models" / "final_model_default.pth")
    record = {
        "n_train": n_train, "n_val": n_val, "epochs_run": len(history["train_loss"]),
        "best_val_loss": best_val, "best_val_ade": min(history["val_ade"]) if history["val_ade"] else None,
        "best_val_fde": min(history["val_fde"]) if history["val_fde"] else None,
        "train_root": args.train_root, "val_root": args.val_root,
        "dataset_config": args.dataset_config, "seed": args.seed, "lr": args.lr,
        "batch_size": args.batch_size, "env": env, "history": history,
        "route_mode": args.route_mode,
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),   # D0#7：事后区分两种 run 的唯一依据
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
        "train_root": args.train_root,
        "val_root": args.val_root,
        "best_val_loss": best_val,
        "env": env,
    }
    with open(out_dir / "models" / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, ensure_ascii=False, indent=2)
    logger.info("train_meta.json 已写出（route_mode=%s, perturbation=%s）",
                args.route_mode, record["perturbation"]["enabled"])
    logger.info("训练完成：best_val=%.4f，best ADE=%.3fm FDE=%.3fm",
                best_val, record["best_val_ade"] or -1, record["best_val_fde"] or -1)
    logger.info("checkpoint: %s（planner bare）/ %s（富）", best_bare, best_ckpt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 MAGNET imitation trainer（route B，不碰 danger 线）")
    p.add_argument("--train_root", required=True, type=str, help="log-disjoint split 的 train leaf root（绝对路径）")
    p.add_argument("--val_root", required=True, type=str, help="log-disjoint split 的 val leaf root（绝对路径）")
    p.add_argument("--output_dir", required=True, type=str, help="输出目录（绝对路径）")
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
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="D0#7：GNN 分支逐样本池化（默认关闭 = 旧行为，逐字节一致）")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
