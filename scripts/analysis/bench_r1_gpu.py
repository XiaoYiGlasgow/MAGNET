"""R1 起跑前的 GPU 实测（Q76）—— CPU 冒烟的 30.64 s/batch 不能外推。

测四个组合：{full, knn30} × {num_workers 0, 4}，各跑 N_WARMUP + N_MEASURE 个 batch。
输出单 batch 耗时与外推的单 epoch 分钟数，供决定 seed 数与档位。

跑法::

    python3 scripts/analysis/bench_r1_gpu.py --device cuda:0 2>&1 | tee data/bench_r1_gpu.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_DEVKIT = _REPO.parent
for p in (str(_REPO), str(_DEVKIT), str(_DEVKIT / "models"), str(_REPO / "scripts/training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_magnet_imitation import ImprovedTrajectoryLoss, build_loaders  # noqa: E402
from temporal_predictor import TemporalPredictor  # noqa: E402
from transformer_predictor import TransformerPredictor  # noqa: E402

ROOT = _REPO / "data/processed_features_scaling_L1_logsplit"
N_WARMUP = 3
N_MEASURE = 12


def bench(device: torch.device, topology: str, workers: int, batch_size: int) -> dict:
    """跑一组配置，返回单 batch 平均耗时与外推的单 epoch 分钟数。"""
    train_loader, _, n_train, _ = build_loaders(
        str(ROOT / "train"), str(ROOT / "val"), batch_size,
        dataset_config="default", train_transform=None,
        gnn_pooling_fix=False, agent_aux=False,
        temporal_graph=(topology != "baseline"),
        temporal_topology=("full" if topology == "baseline" else topology),
    )
    if workers > 0:
        # 🔴 不能事后改 loader.num_workers —— num_workers=0 时 prefetch_factor 是 None，
        # 改成 >0 后断言 `prefetch_factor > 0` 会崩。必须重建。
        from torch_geometric.loader import DataLoader as PyGDataLoader
        train_loader = PyGDataLoader(
            train_loader.dataset, batch_size=batch_size, shuffle=True,
            follow_batch=(["tg_x"] if topology != "baseline" else None),
            num_workers=workers, persistent_workers=True,
        )

    model = (TransformerPredictor() if topology == "baseline" else TemporalPredictor()).to(device)
    crit = ImprovedTrajectoryLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    edges = 0
    it = iter(train_loader)
    for i in range(N_WARMUP + N_MEASURE):
        if i == N_WARMUP:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = batch.to(device)
        if hasattr(batch, "tg_edge_index"):
            edges = max(edges, int(batch.tg_edge_index.shape[1]))
        out = model(batch)
        loss = crit(out, batch.y.view(out.size(0), -1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / N_MEASURE
    n_batches = n_train // batch_size
    return {
        "topology": topology, "workers": workers, "s_per_batch": dt,
        "min_per_epoch": dt * n_batches / 60, "max_edges": edges,
        "gpu_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        # 先落一个张量把 CUDA context 建起来，否则 reset_peak_memory_stats 会报
        # "Invalid device argument: did you call init?"
        torch.zeros(1, device=device)

    print(f"设备 {device}｜batch_size {args.batch_size}｜"
          f"warmup {N_WARMUP} + measure {N_MEASURE}\n")
    print(f"{'配置':<22}{'s/batch':>10}{'min/epoch':>12}{'45ep(h)':>10}{'max边数':>12}{'显存GB':>9}")
    print("-" * 76)

    rows = []
    for topo in ("baseline", "full", "knn30"):
        for w in (0, 4):
            if topo == "baseline" and w == 4:
                continue  # 基线只测一档做参照
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            r = bench(device, topo, w, args.batch_size)
            rows.append(r)
            print(f"{topo + ' w=' + str(w):<22}{r['s_per_batch']:>10.3f}{r['min_per_epoch']:>12.1f}"
                  f"{r['min_per_epoch'] * 45 / 60:>10.1f}{r['max_edges']:>12d}{r['gpu_mem_gb']:>9.2f}")

    base = next((r for r in rows if r["topology"] == "baseline"), None)
    if base:
        print("\n相对基线倍率：")
        for r in rows:
            if r["topology"] != "baseline":
                print(f"  {r['topology']} w={r['workers']}: "
                      f"{r['s_per_batch'] / base['s_per_batch']:.2f}×")


if __name__ == "__main__":
    main()
