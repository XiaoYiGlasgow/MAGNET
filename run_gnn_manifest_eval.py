import argparse
import glob
import json
import os
import sys
import time
import types
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
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

from transformer_predictor import TransformerPredictor
from train import restore_trajectory_from_increments
from src.shared.utils.dataset import GraphDataset, get_adjacency_config


class GPUMonitor:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.max_memory = 0
        self.avg_memory = 0
        self.samples_count = 0
        self.running = False
        self.thread = None

    def _monitor_loop(self):
        import re
        while self.running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', f'--id={self.device_id}'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    mem_used = int(result.stdout.strip())
                    self.max_memory = max(self.max_memory, mem_used)
                    self.avg_memory = (self.avg_memory * self.samples_count + mem_used) / (self.samples_count + 1)
                    self.samples_count += 1
            except Exception:
                pass
            time.sleep(0.5)

    def start(self):
        if torch.cuda.is_available():
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)


def _reference_origin_from_sample(sample, device: torch.device) -> torch.Tensor:
    if hasattr(sample, "ego_history") and sample.ego_history is not None and sample.ego_history.numel() > 0:
        ego_hist = sample.ego_history
        if ego_hist.dim() == 2 and ego_hist.size(0) >= 1 and ego_hist.size(1) >= 2:
            return ego_hist[-1, :2].to(device).unsqueeze(0)
        if ego_hist.dim() == 1 and ego_hist.numel() >= 2:
            return ego_hist[-2:].to(device).unsqueeze(0)

    if hasattr(sample, "ego_last_pos") and sample.ego_last_pos is not None and sample.ego_last_pos.numel() >= 2:
        if sample.ego_last_pos.dim() == 1:
            return sample.ego_last_pos[:2].to(device).unsqueeze(0)
        return sample.ego_last_pos[:, :2].to(device)

    return torch.zeros(1, 2, device=device)


def _load_manifest(manifest_path: str) -> list:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sample_files"]


def _get_gpu_info(device_str: str) -> dict:
    info = {"device": device_str, "available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        if "cuda" in device_str:
            try:
                gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
                info["gpu_name"] = torch.cuda.get_device_name(gpu_id)
                info["gpu_count"] = torch.cuda.device_count()
            except Exception:
                pass
    return info


def evaluate_gnn_manifest(model_path: str, data_dir: str, manifest_path: str, adjacency_config: str, device: str) -> dict:
    start_time = time.time()
    
    gpu_monitor = None
    device_id = 0
    if "cuda" in device and torch.cuda.is_available():
        try:
            device_id = int(device.split(":")[-1]) if ":" in device else 0
        except Exception:
            device_id = 0
        gpu_monitor = GPUMonitor(device_id)
        gpu_monitor.start()
    
    get_adjacency_config(adjacency_config)
    dataset = GraphDataset(data_dir)

    dataset_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if len(dataset_files) != len(dataset):
        raise RuntimeError(f"Dataset size mismatch: {len(dataset_files)} files vs {len(dataset)} dataset samples")

    file_to_index = {os.path.basename(path): idx for idx, path in enumerate(dataset_files)}
    selected_files = _load_manifest(manifest_path)
    missing_files = [name for name in selected_files if name not in file_to_index]
    if missing_files:
        raise FileNotFoundError(f"Manifest files missing from dataset: {missing_files[:5]}")

    selected_indices = [file_to_index[name] for name in selected_files]

    torch_device = torch.device(device)
    model = TransformerPredictor().to(torch_device)
    state_dict = torch.load(model_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader([dataset[idx] for idx in selected_indices], batch_size=1, shuffle=False)

    ade_list = []
    fde_list = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating GNN-Trans"):
            data = data.to(torch_device)
            output = model(data)
            pred_increments = output.view(1, 60, 2)
            true_increments = data.y.view(1, 60, 2)
            reference_origin = _reference_origin_from_sample(data, torch_device)

            pred_traj = restore_trajectory_from_increments(reference_origin, pred_increments)
            true_traj = restore_trajectory_from_increments(reference_origin, true_increments)

            distances = torch.norm(pred_traj - true_traj, dim=2)
            ade_list.append(float(torch.mean(distances).item()))
            fde_list.append(float(distances[:, -1].mean().item()))

    end_time = time.time()
    total_time = end_time - start_time
    
    if gpu_monitor:
        gpu_monitor.stop()
    
    gpu_info = _get_gpu_info(device)
    if gpu_monitor and gpu_monitor.samples_count > 0:
        gpu_info["max_memory_mb"] = gpu_monitor.max_memory
        gpu_info["avg_memory_mb"] = int(gpu_monitor.avg_memory)
        gpu_info["samples_count"] = gpu_monitor.samples_count

    return {
        "model_path": model_path,
        "data_dir": data_dir,
        "adjacency_config": adjacency_config,
        "num_samples": len(selected_indices),
        "mean_ade": float(np.mean(ade_list)),
        "std_ade": float(np.std(ade_list)),
        "mean_fde": float(np.mean(fde_list)),
        "std_fde": float(np.std(fde_list)),
        "median_ade": float(np.median(ade_list)),
        "median_fde": float(np.median(fde_list)),
        "min_ade": float(np.min(ade_list)),
        "max_ade": float(np.max(ade_list)),
        "min_fde": float(np.min(fde_list)),
        "max_fde": float(np.max(fde_list)),
        "ade_list": ade_list,
        "fde_list": fde_list,
        "sample_files": selected_files,
        "timing": {
            "total_time_seconds": round(total_time, 2),
            "time_per_sample_seconds": round(total_time / len(selected_indices), 4),
        },
        "gpu_info": gpu_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNN-Trans checkpoint on a manifest of processed samples.")
    parser.add_argument("--model_path", required=True, help="Path to GNN-Trans checkpoint")
    parser.add_argument("--data_dir", required=True, help="Directory containing processed .pkl files")
    parser.add_argument("--manifest", required=True, help="Manifest json with sample_files list")
    parser.add_argument("--adjacency_config", required=True, help="Adjacency config, e.g. nearby_only")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--output_json", default=None, help="Optional path to save json results")
    args = parser.parse_args()

    results = evaluate_gnn_manifest(
        model_path=args.model_path,
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        adjacency_config=args.adjacency_config,
        device=args.device,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
