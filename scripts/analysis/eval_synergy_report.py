"""E1 固化评估范式：单一可复用脚本，对任意 checkpoint × split 输出
"全 split + 近距子集(GT min_dist<thr)"双报告。

统一指标口径（每条生成轨迹）：danger / offroad% / egoSVO° / minDist / collide%，
并给 GT 锚点。分块跑 full guidance 避免 OOM。

用法：
  python scripts/analysis/eval_synergy_report.py \
      --checkpoint .../training_runs/v7_map_speed/best_model.pt \
      --danger_pair_dir .../dataset/danger_pairs_speed \
      --label v7_map_speed --max_samples 256 --out /tmp/v7_map_speed.json

复用 eval_v5_svo 的 load_models / build_condition / eval_single / SimpleEvalDataset，
不重写模型加载或度量逻辑。
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

GNN = '/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans'
for _p in [GNN, f'{GNN}/src', f'{GNN}/models', f'{GNN}/utils',
           f'{GNN}/scripts', '/mnt/local/Xiao_Yi/nuplan-devkit/models']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_v5_svo import (load_models, build_condition, eval_single,  # noqa: E402
                         SimpleEvalDataset, collate_fn)
from danger_score import DangerScoreConfig  # noqa: E402

PROC_DEFAULT = f'{GNN}/diffusion_danger_model/processed_data'

# Full-guidance weights match the model defaults. Low-pass filtering remains
# enabled at the previously evaluated cutoff to suppress high-frequency noise.
GUIDE_KW = dict(
    k=50, svo_guidance_scale=1.0, svo_weight=1.0, svo_timevarying=True,
    proximity_weight=0.5, near_miss_target=3.0, lane_weight=1.0,
    lambda_kinematic=0.3, time_varying_scale=False, grad_clip_norm=1.0,
    use_lowpass=True, lowpass_cutoff_ratio=0.15,
)


DT = 0.1  # s，NuPlan 10Hz；用于 accel realism 指标


def _accel_mean(dxdy: np.ndarray) -> float:
    """单条 ego 增量序列 (60,2) 的平均加速度幅值 (m/s^2)。"""
    a = np.diff(dxdy / DT, axis=0) / DT     # (59,2)
    return float(np.linalg.norm(a, axis=-1).mean())


def _metrics_for_chunk(gen_ego_np, batch, lane_b, dcfg) -> List[Dict]:
    """对一个 chunk 的生成 ego 增量逐样本算指标。返回 per-sample dict 列表。"""
    eh, ah, af = batch['ego_history'], batch['agent_history'], batch['agent_future_gt']
    recs = []
    for i in range(gen_ego_np.shape[0]):
        m = eval_single(eh[i].numpy(), ah[i].numpy(), gen_ego_np[i], af[i].numpy(),
                        dcfg, lane_segs=lane_b[i])
        recs.append(dict(
            danger=m['outcome_danger'], offroad=m['offroad_frac'] * 100,
            egoSVO=m['ego_svo_deg'], minDist=m['min_distance'],
            collide=float(m['has_collision']) * 100,
            accel=_accel_mean(gen_ego_np[i]),
        ))
    return recs


def _aggregate(recs: List[Dict]) -> Dict:
    """对 per-sample 指标取均值。"""
    keys = ['danger', 'offroad', 'egoSVO', 'minDist', 'collide', 'accel']
    return {k: float(np.mean([r[k] for r in recs])) for k in keys}


def evaluate(checkpoint: str, danger_pair_dir: str, proc: str,
             max_samples: int, chunk: int, close_thr: float) -> Dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcfg = DangerScoreConfig()
    gnn, ce, dm, _aph, le = load_models(checkpoint, device)
    ds = SimpleEvalDataset(proc, danger_pair_dir, max_samples=max_samples)
    n = len(ds)

    # 逐样本记录：gt_md(用于近距掩码) + 三个 mode 的指标
    per_sample: List[Dict] = [dict() for _ in range(n)]

    for start in range(0, n, chunk):
        idx = list(range(start, min(start + chunk, n)))
        batch = next(iter(DataLoader(Subset(ds, idx), batch_size=len(idx),
                                     shuffle=False, collate_fn=collate_fn, num_workers=0)))
        B = batch['ego_history'].shape[0]
        lane_b = batch['lane_polylines'].numpy()
        cond = build_condition(batch, gnn, ce, device, lane_encoder=le)
        x0 = batch['joint_increments'].to(device)
        eh_t = batch['ego_history'].to(device)
        ah_t = batch['agent_history'].to(device)
        lane_t = batch['lane_polylines'].to(device)

        # GT（ego GT 增量）
        gt_ego = batch['gt_increments'].numpy().reshape(B, 60, 2)
        rec_gt = _metrics_for_chunk(gt_ego, batch, lane_b, dcfg)
        # baseline（纯 SDEdit）
        with torch.no_grad():
            gen_b = dm.sdedit_generate_dx_dy(x0, cond, k=GUIDE_KW['k'])
        rec_b = _metrics_for_chunk(gen_b[:, :120].reshape(B, 60, 2).cpu().numpy(),
                                   batch, lane_b, dcfg)
        # full guidance
        gen_f = dm.sdedit_with_full_guidance_phase2(
            x0, cond, ego_history=eh_t, agent_history=ah_t, lane_polylines=lane_t,
            **GUIDE_KW)
        rec_f = _metrics_for_chunk(gen_f[:, :120].reshape(B, 60, 2).detach().cpu().numpy(),
                                   batch, lane_b, dcfg)

        for j, s in enumerate(idx):
            per_sample[s] = dict(
                gt_md=rec_gt[j]['minDist'],   # GT ego-agent 最近距离 → 近距掩码
                GT=rec_gt[j], baseline=rec_b[j], full=rec_f[j])
        print(f'  chunk {start}-{start + len(idx)} done')

    # 全 split + 近距子集聚合
    close_mask = [bool(ps['gt_md'] < close_thr) for ps in per_sample]
    n_close = int(sum(close_mask))
    out = {'n_total': int(n), 'n_close': n_close, 'close_thr': float(close_thr), 'subsets': {}}
    for sub_name, mask in [('full_split', [True] * n), ('close_subset', close_mask)]:
        sel = [ps for ps, m in zip(per_sample, mask) if m]
        out['subsets'][sub_name] = {
            'n': len(sel),
            'GT': _aggregate([ps['GT'] for ps in sel]) if sel else None,
            'baseline': _aggregate([ps['baseline'] for ps in sel]) if sel else None,
            'full': _aggregate([ps['full'] for ps in sel]) if sel else None,
        }
    return out


def _print_table(label: str, out: Dict) -> None:
    print(f'\n===== {label} | N={out["n_total"]} | '
          f'近距子集(GT md<{out["close_thr"]}m)={out["n_close"]} =====')
    for sub_name in ['full_split', 'close_subset']:
        sub = out['subsets'][sub_name]
        print(f'\n[{sub_name}] n={sub["n"]}')
        print(f"  {'config':>10} {'danger':>7} {'offroad':>8} {'egoSVO':>7} "
              f"{'minDist':>8} {'collide':>8} {'accel':>7}")
        for cfg in ['GT', 'baseline', 'full']:
            m = sub[cfg]
            if m is None:
                continue
            print(f"  {cfg:>10} {m['danger']:>7.3f} {m['offroad']:>7.1f}% "
                  f"{m['egoSVO']:>6.1f}° {m['minDist']:>7.2f}m {m['collide']:>7.1f}% "
                  f"{m['accel']:>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--danger_pair_dir', required=True)
    ap.add_argument('--processed_data_dir', default=PROC_DEFAULT)
    ap.add_argument('--label', default='eval')
    ap.add_argument('--max_samples', type=int, default=256)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--close_thr', type=float, default=8.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = evaluate(args.checkpoint, args.danger_pair_dir, args.processed_data_dir,
                   args.max_samples, args.chunk, args.close_thr)
    _print_table(args.label, out)
    if args.out:
        out['label'] = args.label
        out['checkpoint'] = args.checkpoint
        Path(args.out).write_text(json.dumps(
            out, indent=2,
            default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
        print(f'\n[saved] {args.out}')
    print('SYNERGY_REPORT_DONE')


if __name__ == '__main__':
    main()
