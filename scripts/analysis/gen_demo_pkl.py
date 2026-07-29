#!/usr/bin/env python3
"""按指定 token 定向生成 benchmark pkl（baseline + antishake 两版），供 render_paper_figure
出带地图 demo 图。与 build_benchmark 的区别：按 token 选样本（非种子批量）、跳过 danger gate
（保证目标 token 全部产出）；saved 组装格式与 build_benchmark 完全一致。nuplan env 运行。

用法（对一个 split 跑；token 不在该 danger_pair_dir 的自动跳过）：
  CUDA_VISIBLE_DEVICES=2 python scripts/analysis/gen_demo_pkl.py \
    --checkpoint experiments/danger_generation/training_runs/v8_map_intersection_full/best_model.pt \
    --danger_pair_dir diffusion_danger_model/dataset/danger_pairs_intersection_fixed \
    --out_dir docs/videos/metadrive/antishake_demo_20260624
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
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
from build_benchmark import _abs_traj, _seed_geom  # noqa: E402
from danger_generation.core.benchmark_filter import (  # noqa: E402
    BenchmarkFilterConfig, compute_scenario_metrics, truncate_at_collision)

# 两版对比 agent 约束修复（共用：CLI 正确工作点 lane/kin + 关抗震，唯一变量=agent 是否受 lane/kin 约束）
VARIANTS = {
    "before_fix": dict(lane_constrain_agent=False, kin_constrain_agent=False),
    "after_fix":  dict(lane_constrain_agent=True,  kin_constrain_agent=True),
}


def build_saved(token, kid, eh, ah, ego_inc, agent_inc, gt_inc, agent_fgt,
                lane, d_raw, fcfg, kw, seed, ckpt, variant):
    """组装 build_benchmark 同款 saved dict（绝对轨迹 + metrics + 碰撞截断）。"""
    ego_xy = _abs_traj(eh[-1, :2], ego_inc)
    agent_xy = _abs_traj(ah[-1, :2], agent_inc)
    metrics = compute_scenario_metrics(ego_xy, agent_xy, lane, fcfg)
    ego_save, agent_save, coll = truncate_at_collision(ego_xy, agent_xy)
    metrics["collision_frame"] = coll
    seed_rh, seed_lat = _seed_geom(d_raw)
    return {
        "scenario_name": token, "key_agent_id": kid,
        "ego_history": eh, "agent_history": ah,
        "ego_future_gen": ego_save, "agent_future_gen": agent_save,
        "ego_future_gt": _abs_traj(eh[-1, :2], gt_inc.reshape(60, 2)),
        "agent_future_gt": agent_fgt[:, :2],
        "lane_polylines": lane, "metrics": metrics,
        "seed_geom": {"rel_heading_deg": seed_rh, "min_lateral_m": seed_lat},
        "generation": {"operating_point": dict(kw), "seed": seed,
                       "checkpoint": ckpt, "variant": variant},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--danger_pair_dir", required=True)
    ap.add_argument("--processed_data_dir",
                    default=f"{GNN}/diffusion_danger_model/processed_data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--real_map_dir",
                    default=f"{GNN}/docs/videos/metadrive/real_maps",
                    help="从中推目标 token（map_<token>.pkl）")
    ap.add_argument("--tokens", default=None,
                    help="逗号分隔 token 覆盖；默认用 real_map_dir 全部")
    ap.add_argument("--start_anchor", type=int, default=5)
    ap.add_argument("--lane_weight", type=float, default=10.0)
    ap.add_argument("--lambda_kinematic", type=float, default=10.0)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="dataset 加载范围（None=全部；目标 token 在前 N 个时可限以加速）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if args.tokens:
        tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    else:
        tokens = sorted(p.stem.replace("map_", "")
                        for p in Path(args.real_map_dir).glob("map_*.pkl"))
    print(f"[target] {len(tokens)} 个 token（来自 {'CLI' if args.tokens else 'real_maps'}）")

    gnn, ce, dm, _aph, le = load_models(args.checkpoint, device)
    ds = SimpleEvalDataset(args.processed_data_dir, args.danger_pair_dir,
                           max_samples=args.max_samples)
    name2idx = {}
    for i, d in enumerate(ds.data_list):
        name2idx.setdefault(d["scenario_name"], i)
    targets = [(t, name2idx[t]) for t in tokens if t in name2idx]
    print(f"[match] 本 split 命中 {len(targets)} / {len(tokens)} 个 token")
    if not targets:
        print("[skip] 本 danger_pair_dir 无目标 token")
        return
    idxs = [i for _, i in targets]

    batch = next(iter(DataLoader(Subset(ds, idxs), batch_size=len(idxs),
                                 shuffle=False, collate_fn=collate_fn,
                                 num_workers=0)))
    cond = build_condition(batch, gnn, ce, device, lane_encoder=le)
    x0 = batch["joint_increments"].to(device)
    eh_t = batch["ego_history"].to(device)
    ah_t = batch["agent_history"].to(device)
    lane_t = batch["lane_polylines"].to(device)
    eh_np = batch["ego_history"].numpy()
    ah_np = batch["agent_history"].numpy()
    lane_np = batch["lane_polylines"].numpy()
    gt_inc = batch["gt_increments"].numpy()
    agent_fgt = batch["agent_future_gt"].numpy()
    kids = batch["key_agent_ids"]
    fcfg = BenchmarkFilterConfig()

    for variant, ov in VARIANTS.items():
        kw = dict(GUIDE_KW)
        kw["lane_weight"] = args.lane_weight          # 正确工作点（强约束）
        kw["lambda_kinematic"] = args.lambda_kinematic
        # 关抗震（这轮聚焦工作点 + agent 等效，不引入抗震）
        kw.update(guidance_g_ref=0.0, guidance_stop_t=0,
                  use_bezier=False, time_varying_scale=False)
        kw["start_anchor_steps"] = args.start_anchor
        kw.update(ov)   # lane_constrain_agent / kin_constrain_agent（唯一变量）
        torch.manual_seed(args.seed)
        gen = dm.sdedit_with_full_guidance_phase2(
            x0, cond, ego_history=eh_t, agent_history=ah_t,
            lane_polylines=lane_t, **kw).detach().cpu().numpy()
        outd = Path(args.out_dir) / variant
        outd.mkdir(parents=True, exist_ok=True)
        for j, (tok, _) in enumerate(targets):
            ego_inc = gen[j, :120].reshape(60, 2)
            agent_inc = gen[j, 120:].reshape(60, 2)
            saved = build_saved(tok, kids[j], eh_np[j], ah_np[j], ego_inc,
                                agent_inc, gt_inc[j], agent_fgt[j], lane_np[j],
                                ds.data_list[idxs[j]], fcfg, kw, args.seed,
                                args.checkpoint, variant)
            pd.to_pickle(saved, outd / f"benchmark_{tok}_agent{kids[j]}.pkl")
        print(f"[{variant}] saved {len(targets)} pkl → {outd}")
    print("GEN_DEMO_DONE")


if __name__ == "__main__":
    main()
