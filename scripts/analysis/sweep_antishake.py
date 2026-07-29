#!/usr/bin/env python3
"""Part 4 逐项标定：在强交互(near-miss)种子上单扫每个抗震参数，
用 jerk(锯齿，越低越好) + min_dist(ego-agent 最近距离，衡量交互是否被推没) 双指标，
标定 g_ref / stop_t / bezier / tvs / proximity_weight / svo_guidance_scale 终值。

强交互种子 = 先跑 baseline 生成，按 ego-agent min_dist 选最小的 N 个（最接近昨天震荡场景）。
逐项扫 = 固定其余为抗震默认(ANTI)，单变一项；baseline(全关) 作参照。

用法(GPU2):
  CUDA_VISIBLE_DEVICES=2 python scripts/analysis/sweep_antishake.py \
    --checkpoint experiments/danger_generation/training_runs/v8_map_speed_full/best_model.pt \
    --danger_pair_dir diffusion_danger_model/dataset/danger_pairs_speed_full \
    --processed_data_dir data/processed_features_speed_full \
    --seed_mode speed_corridor --n_seeds 6 --n_candidates 16
"""
import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

GNN = "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
for _p in [GNN, f"{GNN}/src", f"{GNN}/models", f"{GNN}/utils",
           f"{GNN}/scripts", f"{GNN}/scripts/analysis",
           "/mnt/local/Xiao_Yi/nuplan-devkit/models"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_v5_svo import (load_models, build_condition,  # noqa: E402
                         SimpleEvalDataset, collate_fn)
from eval_synergy_report import GUIDE_KW  # noqa: E402
from build_benchmark import _speed_seed_indices, _close_seed_indices  # noqa: E402
from danger_generation.core.benchmark_filter import SpeedSeedConfig  # noqa: E402
from diagnose_oscillation import ego_jerk  # noqa: E402

# 抗震默认工作点（逐项扫时固定其余为此）
ANTI = dict(guidance_g_ref=0.3, guidance_stop_t=5,
            use_bezier=True, time_varying_scale=True)


def _abs(last_xy: np.ndarray, inc: np.ndarray) -> np.ndarray:
    return last_xy[None, :] + np.cumsum(inc, axis=0)


def gen_once(dm, x0, cond, eh_t, ah_t, lane_t, guide_kw, seed):
    torch.manual_seed(seed)
    gen = dm.sdedit_with_full_guidance_phase2(
        x0, cond, ego_history=eh_t, agent_history=ah_t,
        lane_polylines=lane_t, debug=False, **guide_kw)
    return gen.detach().cpu().numpy()


def per_sample_min_dist(gen_np, eh_np, ah_np):
    """每个样本 ego_gen vs agent_gen 的最小欧氏距离 (B,)。"""
    B = gen_np.shape[0]
    ego_inc = gen_np[:, :120].reshape(B, 60, 2)
    agt_inc = gen_np[:, 120:].reshape(B, 60, 2)
    out = np.zeros(B)
    for i in range(B):
        e = _abs(eh_np[i, -1, :2], ego_inc[i])
        a = _abs(ah_np[i, -1, :2], agt_inc[i])
        out[i] = np.linalg.norm(e - a, axis=1).min()
    return out


def metrics(gen_np, eh_np, ah_np, sub):
    """子集 sub 上的 (jerk, mean min_dist)。"""
    B = gen_np.shape[0]
    ego_inc = gen_np[:, :120].reshape(B, 60, 2)
    j = ego_jerk(ego_inc[sub])
    md = per_sample_min_dist(gen_np, eh_np, ah_np)[sub].mean()
    return j, float(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--danger_pair_dir", required=True)
    ap.add_argument("--processed_data_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=6,
                    help="选 min_dist 最小的 N 个强交互种子")
    ap.add_argument("--n_candidates", type=int, default=16,
                    help="候选种子数（从中按 baseline min_dist 选强交互）")
    ap.add_argument("--max_samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_mode", choices=["speed_corridor", "close"],
                    default="speed_corridor")
    ap.add_argument("--seed_close_thr", type=float, default=8.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    gnn, ce, dm, _aph, le = load_models(args.checkpoint, device)
    ds = SimpleEvalDataset(args.processed_data_dir, args.danger_pair_dir,
                           max_samples=args.max_samples)

    if args.seed_mode == "speed_corridor":
        seed_indices, _ = _speed_seed_indices(ds, SpeedSeedConfig())
    else:
        seed_indices, _ = _close_seed_indices(ds, args.seed_close_thr)
    cand = seed_indices[:args.n_candidates]
    if len(cand) < args.n_seeds:
        print(f"[error] 种子不足: cand={len(cand)} < n_seeds={args.n_seeds}")
        return

    batch = next(iter(DataLoader(Subset(ds, cand), batch_size=len(cand),
                                 shuffle=False, collate_fn=collate_fn,
                                 num_workers=0)))
    cond = build_condition(batch, gnn, ce, device, lane_encoder=le)
    x0 = batch["joint_increments"].to(device)
    eh_t = batch["ego_history"].to(device)
    ah_t = batch["agent_history"].to(device)
    lane_t = batch["lane_polylines"].to(device)
    eh_np = batch["ego_history"].numpy()
    ah_np = batch["agent_history"].numpy()

    # 1) baseline 选强交互种子
    base_kw = dict(GUIDE_KW)
    base_kw.update(guidance_g_ref=0.0, guidance_stop_t=0,
                   use_bezier=False, time_varying_scale=False)
    gen_b = gen_once(dm, x0, cond, eh_t, ah_t, lane_t, base_kw, args.seed)
    md_all = per_sample_min_dist(gen_b, eh_np, ah_np)
    sub = np.argsort(md_all)[:args.n_seeds]   # baseline min_dist 最小的 N 个（local 索引）
    picked = [cand[i] for i in sub]
    print(f"[seed] 候选 {len(cand)} → 选 baseline min_dist 最小 {len(sub)} 个强交互种子")
    for li in sub:
        print(f"    idx={cand[li]}  baseline_min_dist={md_all[li]:.2f}m  "
              f"{ds.data_list[cand[li]]['scenario_name']}")

    # 2) 逐项扫描
    def run(label, **over):
        kw = dict(GUIDE_KW)
        kw.update(ANTI)
        kw.update(over)
        gen = gen_once(dm, x0, cond, eh_t, ah_t, lane_t, kw, args.seed)
        j, md = metrics(gen, eh_np, ah_np, sub)
        print(f"  {label:34s} jerk={j:8.3f}  min_dist={md:6.3f}")

    j0, md0 = metrics(gen_b, eh_np, ah_np, sub)
    print(f"\n{'=' * 60}\n强交互 N={len(sub)} 子集逐项扫描 (jerk↓好, min_dist 保交互)\n{'=' * 60}")
    print(f"  {'BASELINE (全关)':34s} jerk={j0:8.3f}  min_dist={md0:6.3f}")
    print(f"  {'ANTI 默认 (g0.3/s5/bez/tvs)':34s}", end="")
    gen_anti = gen_once(dm, x0, cond, eh_t, ah_t, lane_t,
                        {**GUIDE_KW, **ANTI}, args.seed)
    ja, mda = metrics(gen_anti, eh_np, ah_np, sub)
    print(f" jerk={ja:8.3f}  min_dist={mda:6.3f}")

    print("\n-- 单扫 guidance_g_ref --")
    for v in [0.0, 0.1, 0.3, 0.5, 1.0]:
        run(f"g_ref={v}", guidance_g_ref=v)
    print("\n-- 单扫 guidance_stop_t --")
    for v in [0, 3, 5, 10]:
        run(f"stop_t={v}", guidance_stop_t=v)
    print("\n-- 单扫 use_bezier --")
    for v in [False, True]:
        run(f"bezier={v}", use_bezier=v)
    print("\n-- 单扫 time_varying_scale --")
    for v in [False, True]:
        run(f"tvs={v}", time_varying_scale=v)
    print("\n-- 单扫 proximity_weight --")
    for v in [0.25, 0.5, 1.0]:
        run(f"prox_w={v}", proximity_weight=v)
    print("\n-- 单扫 svo_guidance_scale --")
    for v in [0.5, 1.0]:
        run(f"svo_scale={v}", svo_guidance_scale=v)

    print("\nSWEEP_DONE")


if __name__ == "__main__":
    main()
