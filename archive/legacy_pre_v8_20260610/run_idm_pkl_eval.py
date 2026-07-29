import argparse
import glob
import json
import os
import sys
import time
import types
import subprocess
import threading

import numpy as np
import pandas as pd
import torch

from idm_predictor import IDMPredictor, calculate_metrics


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


def _load_paths(data_dir: str, num_samples: int, manifest_path: str = None) -> list:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {data_dir}")

    path_map = {os.path.basename(path): path for path in paths}

    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        selected_files = manifest["sample_files"][: min(num_samples, len(manifest["sample_files"]))]
        missing_files = [name for name in selected_files if name not in path_map]
        if missing_files:
            raise FileNotFoundError(f"Manifest files missing from data dir: {missing_files[:5]}")
        return [path_map[name] for name in selected_files]

    return paths[: min(num_samples, len(paths))]


def _get_gpu_info() -> dict:
    info = {"available": torch.cuda.is_available(), "device": "cpu"}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def evaluate_from_pkl_dir(data_dir: str, num_samples: int, manifest_path: str = None) -> dict:
    start_time = time.time()
    
    gpu_monitor = None
    if torch.cuda.is_available():
        gpu_monitor = GPUMonitor(0)
        gpu_monitor.start()
    
    selected_paths = _load_paths(data_dir, num_samples, manifest_path)
    predictor = IDMPredictor()

    ade_list = []
    fde_list = []

    for path in selected_paths:
        df = pd.read_pickle(path)
        ego_history_raw = df["EGO_HISTORY"].values[0]
        true_increments = df["GROUND_TRUTH"].values[0].squeeze(0).astype(np.float32)
        sample = types.SimpleNamespace(
            ego_history=torch.tensor(ego_history_raw, dtype=torch.float32),
            y=torch.tensor(true_increments, dtype=torch.float32),
            x=torch.tensor(df["POLYLINE_FEATURES"].values[0].astype(np.float32), dtype=torch.float32),
            ego_last_pos=torch.tensor(np.asarray(df["EGO_LAST_POS"].values[0], dtype=np.float32)[:2], dtype=torch.float32),
        )

        pred_traj = predictor.predict_trajectory(sample, prediction_steps=60)
        ego_pos, _, ego_heading = predictor.extract_ego_state_from_data(sample)
        true_traj = predictor.recover_global_trajectory_from_local_increments(
            true_increments,
            ego_pos,
            ego_heading,
        )

        ade, fde = calculate_metrics(pred_traj, true_traj)
        ade_list.append(float(ade))
        fde_list.append(float(fde))

    end_time = time.time()
    total_time = end_time - start_time
    
    if gpu_monitor:
        gpu_monitor.stop()
    
    gpu_info = _get_gpu_info()
    if gpu_monitor and gpu_monitor.samples_count > 0:
        gpu_info["max_memory_mb"] = gpu_monitor.max_memory
        gpu_info["avg_memory_mb"] = int(gpu_monitor.avg_memory)
        gpu_info["samples_count"] = gpu_monitor.samples_count

    return {
        "data_dir": data_dir,
        "num_samples": len(selected_paths),
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
        "sample_files": [os.path.basename(path) for path in selected_paths],
        "timing": {
            "total_time_seconds": round(total_time, 2),
            "time_per_sample_seconds": round(total_time / len(selected_paths), 4),
        },
        "gpu_info": gpu_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IDM predictor directly from processed .pkl files.")
    parser.add_argument("--data_dir", required=True, help="Directory containing processed .pkl files")
    parser.add_argument("--num_samples", type=int, default=3000, help="Number of samples to evaluate")
    parser.add_argument("--manifest", default=None, help="Optional manifest json with sample_files list")
    parser.add_argument("--output_json", default=None, help="Optional path to save json results")
    args = parser.parse_args()

    results = evaluate_from_pkl_dir(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        manifest_path=args.manifest,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to: {args.output_json}")


if __name__ == "__main__":
    main()
