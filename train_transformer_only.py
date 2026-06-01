import argparse
import json
import os
import sys
import time
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
REPO_ROOT = str(Path(CURRENT_DIR).parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
REPO_MODELS_DIR = os.path.join(REPO_ROOT, "models")
if REPO_MODELS_DIR not in sys.path:
    sys.path.insert(0, REPO_MODELS_DIR)

import config

from transformer_only_predictor import TransformerOnlyPredictor
from src.shared.utils.dataset import GraphDataset, get_adjacency_config


def restore_trajectory_from_increments(reference_origin: torch.Tensor, increments: torch.Tensor) -> torch.Tensor:
    batch_size = reference_origin.shape[0]
    num_timesteps = increments.shape[1]
    trajectory = torch.zeros(batch_size, num_timesteps, 2, device=reference_origin.device)
    current_pos = reference_origin
    for t in range(num_timesteps):
        current_pos = current_pos + increments[:, t, :]
        trajectory[:, t, :] = current_pos
    return trajectory


def setup_logging(log_file_path: str):
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_training_folders(dataset_size: int, batch_size: int, learning_rate: float, train_path: str) -> dict:
    scenario_type = "unknown"
    if train_path:
        path_parts = train_path.split(os.sep)
        for part in path_parts:
            if part.startswith("processed_") and part != "processed_data_by_type":
                scenario_name = part.replace("processed_", "", 1)
                scenario_type = f"type_{scenario_name}"
                break
            elif part.startswith("type_"):
                scenario_type = part
                break

    main_folder = f"results_{scenario_type}_transformer_only_baseline"
    main_path = os.path.join(config.TRANSFORMER_ONLY_BASELINES_DIR, main_folder)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_folder = f"training_{timestamp}_ds{dataset_size}_bs{batch_size}_lr{learning_rate:.0e}"
    session_path = os.path.join(main_path, session_folder)

    folders = {
        "base": session_path,
        "models": os.path.join(session_path, "models"),
        "results": os.path.join(session_path, "results"),
        "logs": os.path.join(session_path, "logs"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def save_progress_info(
    folders: dict,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    ade: float,
    fde: float,
    lr: float,
    epoch_time: float,
    estimated_remaining: float,
):
    progress_info = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "ade": ade,
        "fde": fde,
        "learning_rate": lr,
        "epoch_time": epoch_time,
        "estimated_remaining_time": estimated_remaining,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    progress_file = os.path.join(folders["base"], "training_progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress_info, f, indent=2)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
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
    def __init__(self, ade_weight: float = 1.0, fde_weight: float = 2.0, smoothness_weight: float = 0.5, direction_weight: float = 0.3):
        super().__init__()
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.smoothness_weight = smoothness_weight
        self.direction_weight = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, return_components: bool = False):
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

        displacement_errors = torch.norm(pred_traj - true_traj, dim=2)
        ade_loss = torch.mean(displacement_errors)
        fde_loss = torch.norm(pred_traj[:, -1, :] - true_traj[:, -1, :], dim=1).mean()

        pred_diffs = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]
        true_diffs = true_traj[:, 1:, :] - true_traj[:, :-1, :]
        smoothness_loss = torch.mean(torch.norm(pred_diffs - true_diffs, dim=2))

        pred_direction = pred_traj[:, -1, :] - pred_traj[:, 0, :]
        true_direction = true_traj[:, -1, :] - true_traj[:, 0, :]
        pred_direction_norm = torch.clamp(torch.norm(pred_direction, dim=1, keepdim=True), min=1e-8)
        true_direction_norm = torch.clamp(torch.norm(true_direction, dim=1, keepdim=True), min=1e-8)
        pred_direction_unit = pred_direction / pred_direction_norm
        true_direction_unit = true_direction / true_direction_norm
        cosine_sim = torch.sum(pred_direction_unit * true_direction_unit, dim=1)
        direction_loss = torch.mean(1 - cosine_sim)

        total_loss = (
            self.ade_weight * ade_loss
            + self.fde_weight * fde_loss
            + self.smoothness_weight * smoothness_loss
            + self.direction_weight * direction_loss
        )

        if return_components:
            return {
                "total_loss": total_loss,
                "ade_loss": ade_loss,
                "fde_loss": fde_loss,
                "smoothness_loss": smoothness_loss,
                "direction_loss": direction_loss,
                "weighted_ade": self.ade_weight * ade_loss,
                "weighted_fde": self.fde_weight * fde_loss,
                "weighted_smoothness": self.smoothness_weight * smoothness_loss,
                "weighted_direction": self.direction_weight * direction_loss,
            }
        return total_loss


def _move_graph_batch_to_device(data, device):
    custom_attrs = {}
    attr_names = [
        "ego_history",
        "ego_last_pos",
        "gnn_x",
        "gnn_edge_index",
        "gnn_edge_attr",
        "traj_id_to_indices",
        "cluster",
        "valid_len",
        "time_step_len",
    ]
    for attr_name in attr_names:
        if hasattr(data, attr_name):
            custom_attrs[attr_name] = getattr(data, attr_name)

    data = data.to(device)

    for attr_name, attr_value in custom_attrs.items():
        if not hasattr(data, attr_name):
            if isinstance(attr_value, torch.Tensor):
                setattr(data, attr_name, attr_value.to(device))
            else:
                setattr(data, attr_name, attr_value)
    return data


def _reference_origin_from_batch(data, batch_size: int, device: torch.device) -> torch.Tensor:
    if hasattr(data, "ego_history") and data.ego_history is not None and data.ego_history.numel() > 0:
        ego_hist = data.ego_history
        if ego_hist.dim() == 3 and ego_hist.shape[0] == batch_size:
            return ego_hist[:, -1, :2]
        if ego_hist.dim() == 2 and ego_hist.shape[0] == batch_size * 20 and ego_hist.shape[1] >= 2:
            return ego_hist.view(batch_size, 20, -1)[:, -1, :2]
        if ego_hist.dim() == 2 and ego_hist.shape[0] == 20 and ego_hist.shape[1] >= 2:
            return ego_hist[-1, :2].unsqueeze(0).repeat(batch_size, 1)
    if hasattr(data, "ego_last_pos") and data.ego_last_pos is not None and data.ego_last_pos.numel() >= 2:
        if data.ego_last_pos.dim() == 2 and data.ego_last_pos.shape[0] == batch_size:
            return data.ego_last_pos[:, :2]
        if data.ego_last_pos.dim() == 1 and data.ego_last_pos.shape[0] == batch_size * 2:
            return data.ego_last_pos.view(batch_size, 2)
        if data.ego_last_pos.dim() == 1 and data.ego_last_pos.shape[0] >= 2:
            return data.ego_last_pos[:2].unsqueeze(0).repeat(batch_size, 1)
    return torch.zeros(batch_size, 2, device=device)


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    num_samples = 0
    ade_list = []
    fde_list = []

    with torch.no_grad():
        for data in val_loader:
            data = _move_graph_batch_to_device(data, device)
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

            loss = criterion(output, y)
            val_loss += loss.item() * batch_size
            num_samples += batch_size

            pred_increments = output.view(batch_size, 60, 2)
            true_increments = data.y.view(batch_size, 60, 2)
            reference_origin = _reference_origin_from_batch(data, batch_size, device)
            pred_trajectory = restore_trajectory_from_increments(reference_origin, pred_increments)
            true_trajectory = restore_trajectory_from_increments(reference_origin, true_increments)
            position_errors = torch.norm(pred_trajectory - true_trajectory, dim=2)
            ade_list.extend(torch.mean(position_errors, dim=1).cpu().numpy())
            fde_list.extend(position_errors[:, -1].cpu().numpy())

    avg_ade = sum(ade_list) / len(ade_list) if ade_list else 0.0
    avg_fde = sum(fde_list) / len(fde_list) if fde_list else 0.0
    return val_loss / max(num_samples, 1), avg_ade, avg_fde


def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer-only baseline inside GNN_Trans.")
    parser.add_argument("--train_path", default=config.TRAIN_PATH, help="Processed dataset root used by GraphDataset")
    parser.add_argument("--dataset_config", default="default", help="Processed dataset config to load, e.g. default")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    get_adjacency_config(args.dataset_config)
    full_dataset = GraphDataset(args.train_path)

    indices = list(range(len(full_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=config.VALIDATION_SPLIT, random_state=config.SEED)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = TransformerOnlyPredictor().to(device)
    criterion = ImprovedTrajectoryLoss(ade_weight=0.5, fde_weight=3.0, smoothness_weight=0.3, direction_weight=0.5)
    initial_lr = args.learning_rate
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-3, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3, verbose=True, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    folders = create_training_folders(len(train_dataset), args.batch_size, initial_lr, args.train_path)
    logger = setup_logging(os.path.join(folders["logs"], "training_log.txt"))
    logger.info("Starting transformer-only baseline training")
    logger.info(f"Device: {device}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Dataset config source: {args.dataset_config}")
    logger.info(f"Train path: {args.train_path}")

    best_val_loss = float("inf")
    train_loss_list = []
    val_loss_list = []
    ade_list = []
    fde_list = []
    lr_list = []
    loss_components_history = []

    epoch_pbar = tqdm(range(args.epochs), desc="Transformer-Only Training")
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        num_samples = 0
        start_time = time.time()

        for data in train_loader:
            data = _move_graph_batch_to_device(data, device)
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

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

            with torch.no_grad():
                components = criterion.forward(output[:1], y[:1], return_components=True)
                loss_components_history.append({k: (v.item() if torch.is_tensor(v) else v) for k, v in components.items()})

        avg_loss = epoch_loss / max(num_samples, 1)
        train_loss_list.append(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_list.append(current_lr)

        val_loss, avg_ade, avg_fde = validate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        ade_list.append(avg_ade)
        fde_list.append(avg_fde)
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time
        epoch_pbar.set_postfix({
            "Train": f"{avg_loss:.4f}",
            "Val": f"{val_loss:.4f}",
            "ADE": f"{avg_ade:.2f}",
            "FDE": f"{avg_fde:.2f}",
            "LR": f"{current_lr:.2e}",
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(folders["models"], "best_model_transformer_only.pth"))

        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        estimated_remaining_time = (args.epochs - epoch - 1) * epoch_time
        save_progress_info(
            folders,
            epoch,
            args.epochs,
            avg_loss,
            val_loss,
            avg_ade,
            avg_fde,
            current_lr,
            epoch_time,
            estimated_remaining_time,
        )
        logger.info(
            f"Epoch {epoch + 1:3d}/{args.epochs}: Train={avg_loss:.4f}, Val={val_loss:.4f}, "
            f"ADE={avg_ade:.2f}m, FDE={avg_fde:.2f}m, Time={epoch_time:.1f}s"
        )

    epoch_pbar.close()
    torch.save(model.state_dict(), os.path.join(folders["models"], "final_model_transformer_only.pth"))

    with open(os.path.join(folders["results"], f"loss_record_transformer_only_ep{len(train_loss_list)}_bs{args.batch_size}_lr{initial_lr}.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "transformer_only",
                "dataset_config": args.dataset_config,
                "train_loss": train_loss_list,
                "val_loss": val_loss_list,
                "ade_history": ade_list,
                "fde_history": fde_list,
                "learning_rates": lr_list,
                "loss_components_history": loss_components_history,
                "best_val_loss": float(best_val_loss),
                "final_ade": ade_list[-1] if ade_list else None,
                "final_fde": fde_list[-1] if fde_list else None,
                "best_ade": min(ade_list) if ade_list else None,
                "best_fde": min(fde_list) if fde_list else None,
                "epochs_trained": len(train_loss_list),
                "early_stopped": len(train_loss_list) < args.epochs,
                "dataset_size": len(train_dataset),
                "validation_size": len(val_dataset),
                "batch_size": args.batch_size,
                "initial_learning_rate": initial_lr,
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
