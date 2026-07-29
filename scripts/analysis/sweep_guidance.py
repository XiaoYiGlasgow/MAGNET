#!/usr/bin/env python3
"""控制实验：定位 v8 生成"平淡/不够危险"的主因。在 v8 _full + 正确工作点
(lane10/kin10 + agent 等效约束 + 关抗震) 上，对强交互种子逐项单扫
k / proximity_weight / svo_guidance_scale / grad_clip_norm，度量：
  - min_dist  (危险性，↓越危险)
  - ego/agent offroad (真实性，应低)
  - jerk      (平滑性)
  - dev_GT    (gen ego 轨迹 vs GT 的 L2 偏离均值，↑=编辑得多/不平淡)
判读：若增大 k 或 guidance 能在保持真实的前提下显著降 min_dist + 升 dev_GT，
则主因是 SDEdit/guidance（不必重训）；若怎么调 min_dist 都降不下、dev_GT 都接近 0，
则模型/condition 是瓶颈（转 history 敏感度测试）。

用法(GPU2):
  CUDA_VISIBLE_DEVICES=2 python scripts/analysis/sweep_guidance.py \
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
from build_benchmark import _speed_seed_indices, _close_seed_indices, _abs_traj  # noqa: E402
from danger_generation.core.benchmark_filter import (  # noqa: E402
    SpeedSeedConfig, BenchmarkFilterConfig, compute_scenario_metrics)
from diagnose_oscillation import ego_jerk  # noqa: E402

# 基础工作点：正确强约束 + agent 等效 + 关抗震（唯一变量来自扫描项）
BASE = dict(lane_weight=10.0, lambda_kinematic=10.0,
            lane_constrain_agent=True, kin_constrain_agent=True,
            guidance_g_ref=0.0, guidance_stop_t=0,
            use_bezier=False, time_varying_scale=False,
            start_anchor_steps=5)


def gen_once(dm, x0, cond, eh_t, ah_t, lane_t, kw, seed):
    torch.manual_seed(seed)
    gen = dm.sdedit_with_full_guidance_phase2(
        x0, cond, ego_history=eh_t, agent_history=ah_t,
        lane_polylines=lane_t, debug=False, **kw)
    return gen.detach().cpu().numpy()


def per_sample(gen_np, eh_np, ah_np, lane_np, gt_inc):
    """每个样本: (min_dist, ego_off, agent_off, dev_GT)。"""
    B = gen_np.shape[0]
    ego_inc = gen_np[:, :120].reshape(B, 60, 2)
    agt_inc = gen_np[:, 120:].reshape(B, 60, 2)
    md, eo, ao, dev = [], [], [], []
    fcfg = BenchmarkFilterConfig()
    for i in range(B):
        ego_xy = _abs_traj(eh_np[i, -1, :2], ego_inc[i])
        agt_xy = _abs_traj(ah_np[i, -1, :2], agt_inc[i])
        m = compute_scenario_metrics(ego_xy, agt_xy, lane_np[i], fcfg)
        md.append(float(m["min_dist"]))
        eo.append(float(m.get("ego_offroad", np.nan)))
        ao.append(float(m.get("agent_offroad", np.nan)))
        gt_xy = _abs_traj(eh_np[i, -1, :2], gt_inc[i].reshape(60, 2))
        dev.append(float(np.linalg.norm(ego_xy - gt_xy, axis=1).mean()))
    return np.array(md), np.array(eo), np.array(ao), np.array(dev)


def summarize(gen_np, eh_np, ah_np, lane_np, gt_inc, sub):
    md, eo, ao, dev = per_sample(gen_np, eh_np, ah_np, lane_np, gt_inc)
    ego_inc = gen_np[:, :120].reshape(gen_np.shape[0], 60, 2)
    j = ego_jerk(ego_inc[sub])
    return (md[sub].mean(), eo[sub].mean(), ao[sub].mean(), dev[sub].mean(), j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--danger_pair_dir", required=True)
    ap.add_argument("--processed_data_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=6)
    ap.add_argument("--n_candidates", type=int, default=16)
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
        print(f"[error] 种子不足 {len(cand)} < {args.n_seeds}"); return

    batch = next(iter(DataLoader(Subset(ds, cand), batch_size=len(cand),
                                 shuffle=False, collate_fn=collate_fn, num_workers=0)))
    cond = build_condition(batch, gnn, ce, device, lane_encoder=le)
    x0 = batch["joint_increments"].to(device)
    eh_t = batch["ego_history"].to(device)
    ah_t = batch["agent_history"].to(device)
    lane_t = batch["lane_polylines"].to(device)
    eh_np = batch["ego_history"].numpy()
    ah_np = batch["agent_history"].numpy()
    lane_np = batch["lane_polylines"].numpy()
    gt_inc = batch["gt_increments"].numpy()

    # 选强交互种子：BASE(k=50) 跑一次，按 min_dist 最小选 N
    base_kw = dict(GUIDE_KW); base_kw.update(BASE)
    gen_b = gen_once(dm, x0, cond, eh_t, ah_t, lane_t, base_kw, args.seed)
    md_all, _, _, _ = per_sample(gen_b, eh_np, ah_np, lane_np, gt_inc)
    sub = np.argsort(md_all)[:args.n_seeds]
    print(f"[seed] 候选 {len(cand)} → 选 min_dist 最小 {len(sub)} 强交互种子: {[cand[i] for i in sub]}")

    hdr = f"  {'config':28s} {'min_dist':>8} {'ego_off':>7} {'agt_off':>7} {'jerk':>7} {'dev_GT':>7}"
    def run(label, **over):
        kw = dict(GUIDE_KW); kw.update(BASE); kw.update(over)
        gen = gen_once(dm, x0, cond, eh_t, ah_t, lane_t, kw, args.seed)
        md, eo, ao, dev, j = summarize(gen, eh_np, ah_np, lane_np, gt_inc, sub)
        print(f"  {label:28s} {md:8.2f} {eo:7.3f} {ao:7.3f} {j:7.2f} {dev:7.2f}")

    print(f"\n{'='*72}\n强交互 N={len(sub)} | min_dist↓=危险 offroad↓=真实 dev_GT↑=编辑多(不平淡)\n{'='*72}")
    print(hdr)
    run("BASE (k50/prox0.5/svo1/clip1)")
    print("\n-- 单扫 k (SDEdit 偏离 GT 程度) --")
    for v in [30, 50, 70, 85]:
        run(f"k={v}", k=v)
    print("\n-- 单扫 proximity_weight --")
    for v in [0.5, 2.0, 5.0]:
        run(f"prox_w={v}", proximity_weight=v)
    print("\n-- 单扫 svo_guidance_scale --")
    for v in [1.0, 2.0, 4.0]:
        run(f"svo_scale={v}", svo_guidance_scale=v)
    print("\n-- 单扫 grad_clip_norm --")
    for v in [1.0, 3.0, 10.0]:
        run(f"grad_clip={v}", grad_clip_norm=v)
    print("\nSWEEP_GUIDANCE_DONE")


if __name__ == "__main__":
    main()
