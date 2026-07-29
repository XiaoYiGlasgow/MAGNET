#!/usr/bin/env python3
"""生成轨迹震荡机制验证：对同一批种子样本，对比 baseline vs 抗震两组生成，
看 ① phase2 debug 表里 guid_norm 是否随 loss 下降而衰减（软归一化生效证据），
② ego 生成增量的 jerk（二阶差分）是否下降（锯齿减弱的定量证据）。

只验证机制，不做筛选落盘（那是 build_benchmark 的职责）。复用 eval_v5_svo /
build_benchmark / eval_synergy_report 的现成件，保证与 benchmark 同一生成路径。

用法（GPU 2）：
  CUDA_VISIBLE_DEVICES=2 python scripts/analysis/diagnose_oscillation.py \
      --checkpoint experiments/danger_generation/training_runs/v8_map_speed_full/best_model.pt \
      --danger_pair_dir diffusion_danger_model/dataset/danger_pairs_speed_full \
      --processed_data_dir data/processed_features_speed_full \
      --seed_mode speed_corridor --n_seeds 4
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
from eval_synergy_report import GUIDE_KW  # noqa: E402（统一 guidance 基础配置）
from build_benchmark import _speed_seed_indices, _close_seed_indices  # noqa: E402
from danger_generation.core.benchmark_filter import SpeedSeedConfig  # noqa: E402

DT = 0.1  # s，NuPlan 10Hz


def ego_jerk(ego_inc: np.ndarray) -> float:
    """ego 生成增量 (B,60,2) 的 jerk 代理：增量=Δ位置≈v·dt，
    二阶差分 /DT^3 ≈ jerk (m/s^3)；取所有样本、所有步的 L2 均值。
    锯齿（高频抖动）会显著抬高该值，是震荡的定量指标。
    """
    v = ego_inc / DT                       # (B,60,2) 速度
    a = np.diff(v, axis=1) / DT            # (B,59,2) 加速度
    j = np.diff(a, axis=1) / DT            # (B,58,2) jerk
    return float(np.linalg.norm(j, axis=-1).mean())


def run_group(dm, x0, cond, eh_t, ah_t, lane_t, guide_kw, tag: str, seed: int):
    """跑一组生成（debug 打 guid_norm 衰减表），返回 (gen_np, ego_jerk)。"""
    print(f"\n{'=' * 72}")
    print(f"[{tag}] g_ref={guide_kw.get('guidance_g_ref', 0)} "
          f"stop_t={guide_kw.get('guidance_stop_t', 0)} "
          f"bezier={guide_kw.get('use_bezier', False)} "
          f"tvs={guide_kw.get('time_varying_scale', False)}")
    print('=' * 72)
    torch.manual_seed(seed)   # 重置采样链，两组可比
    gen = dm.sdedit_with_full_guidance_phase2(
        x0, cond, ego_history=eh_t, agent_history=ah_t,
        lane_polylines=lane_t, debug=True, **guide_kw)
    gen = gen.detach().cpu().numpy()
    B = gen.shape[0]
    ego_inc = gen[:, :120].reshape(B, 60, 2)
    j = ego_jerk(ego_inc)
    print(f"[{tag}] ego jerk (二阶差分 L2 均值): {j:.4f} m/s^3")
    return gen, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--danger_pair_dir", required=True)
    ap.add_argument("--processed_data_dir", required=True)
    ap.add_argument("--n_seeds", type=int, default=4,
                    help="取前 N 个种子样本做机制验证（默认 4）")
    ap.add_argument("--max_samples", type=int, default=256,
                    help="dataset 加载范围（种子从中筛）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_mode", choices=["speed_corridor", "close"],
                    default="speed_corridor")
    ap.add_argument("--seed_close_thr", type=float, default=8.0,
                    help="[close 模式] GT min_dist 阈值")
    ap.add_argument("--out_npz", default=None,
                    help="可选：落盘两组生成结果供后续分析/render")
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
    if len(seed_indices) == 0:
        print("[error] 无种子样本，检查 seed_mode / 数据路径")
        return
    idx = seed_indices[:args.n_seeds]
    names = [ds.data_list[i]["scenario_name"] for i in idx]
    print(f"[seed] 取前 {len(idx)} 个 {args.seed_mode} 种子: idx={idx}")
    for i, nm in zip(idx, names):
        print(f"    idx={i}  {nm}")

    batch = next(iter(DataLoader(Subset(ds, idx), batch_size=len(idx),
                                 shuffle=False, collate_fn=collate_fn,
                                 num_workers=0)))
    cond = build_condition(batch, gnn, ce, device, lane_encoder=le)
    x0 = batch["joint_increments"].to(device)
    eh_t = batch["ego_history"].to(device)
    ah_t = batch["agent_history"].to(device)
    lane_t = batch["lane_polylines"].to(device)

    # 两组只切换抗震 4 项（其余共用 GUIDE_KW，保证唯一变量）
    base_kw = dict(GUIDE_KW)
    base_kw.update(guidance_g_ref=0.0, guidance_stop_t=0,
                   use_bezier=False, time_varying_scale=False)
    anti_kw = dict(GUIDE_KW)
    anti_kw.update(guidance_g_ref=0.3, guidance_stop_t=5,
                   use_bezier=True, time_varying_scale=True)

    gen_base, j_base = run_group(dm, x0, cond, eh_t, ah_t, lane_t,
                                 base_kw, "BASELINE", args.seed)
    gen_anti, j_anti = run_group(dm, x0, cond, eh_t, ah_t, lane_t,
                                 anti_kw, "ANTISHAKE", args.seed)

    drop = 100.0 * (j_base - j_anti) / max(j_base, 1e-9)
    print(f"\n{'=' * 72}")
    print(f"[对比] ego jerk: baseline={j_base:.4f} → antishake={j_anti:.4f}  "
          f"降幅 {drop:.1f}%")
    print(f"[判读] 期望：guid_norm 末段衰减（见上表）+ jerk 显著下降 → 抗震生效")
    print('=' * 72)

    if args.out_npz:
        np.savez(args.out_npz, gen_base=gen_base, gen_anti=gen_anti,
                 idx=np.array(idx), j_base=j_base, j_anti=j_anti)
        print(f"[saved] {args.out_npz}")
    print("DIAGNOSE_DONE")


if __name__ == "__main__":
    main()
