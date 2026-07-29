"""
定性可视化脚本 - 危险场景生成

阶段1 scan  (--mode scan):
    遍历全部 81 个 pkl（N=8 推理），输出 scan_results.json（按 Δdanger 降序）。
    用于人工挑选 2-3 个代表性场景。

阶段2 render (--mode render --scenarios <file1> ...):
    对选定 pkl 重跑 N=8 推理，渲染 2×2 论文图（含道路 polyline 灰线背景）。
    输出到 danger_gen_ablation/results/qualitative/。

用法：
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    conda activate nuplan

    python danger_gen_ablation/visualize_qualitative.py --mode scan
    python danger_gen_ablation/visualize_qualitative.py --mode render \\
        --scenarios danger_pair_features_xxxx_agent1.pkl
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
import pandas as pd

# ---------- path setup (consistent with train_ablation.py) ----------
_ablation_dir = os.path.dirname(os.path.abspath(__file__))
gnn_trans_root = os.path.dirname(_ablation_dir)

sys.path = [p for p in sys.path if os.path.basename(os.path.normpath(p)) != 'nuplan-devkit']
sys.path.insert(0, gnn_trans_root)
sys.path.insert(0, os.path.join(gnn_trans_root, 'models'))
sys.path.insert(0, os.path.join(gnn_trans_root, 'utils'))

_models_path = '/mnt/local/Xiao_Yi/nuplan-devkit/models'
if _models_path not in sys.path:
    sys.path.insert(0, _models_path)

_viz_dir = os.path.join(gnn_trans_root, 'visualization')
if _viz_dir not in sys.path:
    sys.path.insert(0, _viz_dir)

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import config
from matplotlib_visualizer import MatplotlibVisualizer, EvalSample
from visualization.trajectory_converter import (
    kinematic_decode_np,
    kinematic_cumulative_decode_np,
    xy_displacement_decode_np,
    bicycle_model_decode_np,
    recover_trajectory_from_increment,
    check_kinematic_validity,
    future_gt_to_kinematic_np,
    agent_future_to_kinematic_np,
)

# ---------- constants ----------
_V310_MODEL = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_10', 'best_model.pt',
)
_V313_MODEL = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_13', 'best_model.pt',
)
_V314_MODEL = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_14', 'best_model.pt',
)
_V315_MODEL = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_15', 'best_model.pt',
)
_V316_MODEL = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_16', 'best_model.pt',
)
# Active model: v3.16 reverts to (Δx,Δy) + direction_reversal + jerk constraints
_ACTIVE_MODEL = _V316_MODEL
_ACTIVE_BICYCLE = False    # v3.15: bicycle model (a, δ) space
_ACTIVE_XY = True          # v3.14/v3.16: (Δx, Δy) displacement space
_ACTIVE_CUMULATIVE = False # v3.13: (v_norm, θ_abs) — kept for fallback comparison
_OUTPUT_DIR = os.path.join(_ablation_dir, 'results', 'qualitative')

MIN_GT_DANGER = 3.0
N_INFERENCE = 8
MARGIN_THRESH = 0.5


# ============================================================
# Helpers
# ============================================================

def _extract_lane_polylines(
    polyline_features: np.ndarray,
    traj_id_to_indices: dict,
    proximity_thresh: float = 15.0,
) -> List[np.ndarray]:
    """Return non-agent rows from polyline_features as connected (N,2) lane segments."""
    agent_rows: set = set()
    for start, end in traj_id_to_indices.values():
        agent_rows.update(range(int(start), int(end) + 1))

    lane_idxs = [i for i in range(len(polyline_features)) if i not in agent_rows]
    if not lane_idxs:
        return []

    segments: List[List[int]] = []
    current: List[int] = [lane_idxs[0]]
    for k in range(1, len(lane_idxs)):
        prev, curr = lane_idxs[k - 1], lane_idxs[k]
        if curr != prev + 1:
            segments.append(current)
            current = [curr]
        else:
            dist = float(np.linalg.norm(polyline_features[curr, :2] - polyline_features[prev, :2]))
            if dist > proximity_thresh:
                segments.append(current)
                current = [curr]
            else:
                current.append(curr)
    segments.append(current)
    return [polyline_features[seg, :2] for seg in segments if len(seg) >= 2]


def _inference_best_of_n(
    viz: MatplotlibVisualizer,
    sample_data: dict,
    device: str,
    n_samples: int = N_INFERENCE,
    use_sdedit: bool = False,
    sdedit_k: int = 40,
    cumulative_space: bool = False,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """Best-of-N diffusion inference (baseline or SDEdit).

    Args:
        use_sdedit: if True, use SDEdit inference (GT-anchored); else pure noise baseline.
        sdedit_k: SDEdit noise level (1..99).  Lower → closer to GT.
        cumulative_space: if True, model output is (v_norm, θ_abs) per frame (v3.13+);
            if False, output is (Δv, Δθ) increments (v3.10).

    Returns:
        (gen_traj (60,5), gen_danger, gt_danger, inherent_danger, best_increments)
    """
    agent_future = sample_data['key_agent_future']
    ego_h = torch.tensor(sample_data['ego_history'], dtype=torch.float32, device=device)
    agent_h = torch.tensor(sample_data['agent_history'], dtype=torch.float32, device=device)

    gt_result = viz.compute_danger(
        sample_data['ego_history'], sample_data['ego_future_gt'],
        sample_data['agent_history'], agent_future, viz.danger_config,
    )
    gt_danger = float(gt_result['outcome_danger']['total_score'])
    inherent_danger = float(gt_result['inherent_danger']['total_score'])

    danger_target = gt_danger + 2.0
    ego_v0 = float(np.sqrt(sample_data['start_vx'] ** 2 + sample_data['start_vy'] ** 2))
    _graph = sample_data['graph_data'].to(device) if sample_data['graph_data'] is not None else None

    best_inc: Optional[np.ndarray] = None
    best_score = -float('inf')
    for _ in range(n_samples):
        if use_sdedit:
            cand = viz.generate_trajectory_sdedit(
                ego_h, agent_h, sample_data['svo_angle'],
                inherent_danger, danger_target,
                sample_data=sample_data,
                k=sdedit_k,
                graph_data=_graph,
            )
        else:
            cand = viz.generate_trajectory(
                ego_h, agent_h, sample_data['svo_angle'],
                inherent_danger, danger_target, graph_data=_graph,
            )

        joint = len(cand) == 240
        ego_inc = cand[:120] if joint else cand
        if _ACTIVE_BICYCLE:
            et = bicycle_model_decode_np(ego_inc, sample_data['ego_init_state'])
        elif _ACTIVE_XY:
            et = xy_displacement_decode_np(ego_inc, sample_data['start_pos'])
        elif cumulative_space:
            et = kinematic_cumulative_decode_np(ego_inc, sample_data['start_pos'])
        else:
            et = kinematic_decode_np(ego_inc, ego_v0, sample_data['start_heading'], sample_data['start_pos'])

        if joint:
            if _ACTIVE_BICYCLE:
                ag = bicycle_model_decode_np(cand[120:], sample_data['agent_init_state']).full_trajectory
            elif _ACTIVE_XY:
                al = sample_data['agent_history'][-1]
                ag = xy_displacement_decode_np(cand[120:], al[:2]).full_trajectory
            elif cumulative_space:
                al = sample_data['agent_history'][-1]
                ag = kinematic_cumulative_decode_np(cand[120:], al[:2]).full_trajectory
            else:
                al = sample_data['agent_history'][-1]
                av0 = float(np.sqrt(al[3] ** 2 + al[4] ** 2))
                ag = kinematic_decode_np(cand[120:], av0, al[2], al[:2]).full_trajectory
        else:
            ag = agent_future

        score = float(viz.compute_danger(
            sample_data['ego_history'], et.full_trajectory,
            sample_data['agent_history'], ag, viz.danger_config,
        )['outcome_danger']['total_score'])

        if score > best_score:
            best_score, best_inc = score, cand

    joint = len(best_inc) == 240
    ego_inc = best_inc[:120] if joint else best_inc
    if _ACTIVE_BICYCLE:
        gen_traj = bicycle_model_decode_np(ego_inc, sample_data['ego_init_state'])
    elif _ACTIVE_XY:
        gen_traj = xy_displacement_decode_np(ego_inc, sample_data['start_pos'])
    elif cumulative_space:
        gen_traj = kinematic_cumulative_decode_np(ego_inc, sample_data['start_pos'])
    else:
        gen_traj = kinematic_decode_np(ego_inc, ego_v0, sample_data['start_heading'], sample_data['start_pos'])
    return gen_traj.full_trajectory, float(best_score), gt_danger, inherent_danger, best_inc


def _load_polylines(pkl_file: Path, processed_dir: Path) -> List[np.ndarray]:
    """Load road lane polylines from the matching processed_data file."""
    scenario_id = pkl_file.stem.replace('danger_pair_features_', '').split('_agent')[0]
    processed_file = processed_dir / f'features_{scenario_id}.pkl'
    if not processed_file.exists():
        print(f'  [Warn] processed file not found: {processed_file.name}')
        return []
    try:
        proc = pd.read_pickle(processed_file)
        pf = np.asarray(proc.iloc[0]['POLYLINE_FEATURES'])
        tid = proc.iloc[0]['TRAJ_ID_TO_INDICES']
        segs = _extract_lane_polylines(pf, tid)
        return segs
    except Exception as e:
        print(f'  [Warn] polyline load failed: {e}')
        return []


# ============================================================
# Scan mode
# ============================================================

def cmd_scan(args) -> None:
    """Scan all pkls with N=8, save scan_results.json sorted by danger improvement."""
    output_dir = Path(_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[Scan] Loading model: {_ACTIVE_MODEL}')
    viz = MatplotlibVisualizer(
        model_path=_ACTIVE_MODEL,
        data_dir=config.DIFFUSION_DANGER_PAIR_DIR,
        processed_data_dir=config.DIFFUSION_PROCESSED_DATA_DIR,
        device=args.device,
        min_gt_danger=0.0,
    )

    pkl_files = sorted(Path(config.DIFFUSION_DANGER_PAIR_DIR).glob('danger_pair_features_*.pkl'))
    print(f'[Scan] {len(pkl_files)} pkl files  |  N_INFERENCE={N_INFERENCE}  |  MIN_GT_DANGER={MIN_GT_DANGER}')

    records: List[dict] = []
    for idx, pkl_file in enumerate(pkl_files):
        print(f'\n[{idx + 1}/{len(pkl_files)}] {pkl_file.name}')
        rec: dict = {'pkl_file': pkl_file.name}

        try:
            sample_data = viz.load_sample(pkl_file)
        except Exception as e:
            print(f'  [Skip] load error: {e}')
            rec.update({'error': str(e), 'margin_met': False})
            records.append(rec)
            continue

        # GT danger (no inference needed)
        agent_future = sample_data['key_agent_future']
        gt_result = viz.compute_danger(
            sample_data['ego_history'], sample_data['ego_future_gt'],
            sample_data['agent_history'], agent_future, viz.danger_config,
        )
        gt_danger = float(gt_result['outcome_danger']['total_score'])
        rec['gt_danger'] = gt_danger
        rec['svo_angle'] = float(sample_data['svo_angle'])
        rec['scenario_name'] = str(sample_data.get('scenario_name', ''))

        if gt_danger < MIN_GT_DANGER:
            print(f'  [Skip] gt_danger={gt_danger:.2f} < {MIN_GT_DANGER}')
            rec.update({'gen_danger': None, 'danger_improvement': None,
                        'margin_met': False, 'skip_reason': 'low_gt_danger'})
            records.append(rec)
            continue

        try:
            _, gen_danger, _, _, _ = _inference_best_of_n(
                viz, sample_data, args.device,
                use_sdedit=args.sdedit, sdedit_k=args.sdedit_k,
                cumulative_space=_ACTIVE_CUMULATIVE,
            )
            delta = gen_danger - gt_danger
            margin_met = delta >= MARGIN_THRESH
            rec.update({'gen_danger': gen_danger, 'danger_improvement': delta, 'margin_met': margin_met})
            print(f'  gt={gt_danger:.2f}  gen={gen_danger:.2f}  Δ={delta:+.2f}  '
                  f'margin={"✓" if margin_met else "✗"}')
        except Exception as e:
            print(f'  [Inference error] {e}')
            rec.update({'gen_danger': None, 'danger_improvement': None,
                        'margin_met': False, 'error': str(e)})

        records.append(rec)

    # Sort: margin_met first, then by improvement descending; failures last
    records.sort(key=lambda r: r.get('danger_improvement') or -9999, reverse=True)

    out_path = output_dir / 'scan_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    ok = [r for r in records if r.get('margin_met')]
    print(f'\n[Scan Done] {len(ok)}/{len(records)} scenes with margin_met=True')
    print('Top candidates (by Δdanger):')
    for r in ok[:15]:
        print(f"  {r['pkl_file']:62s}  Δ={r['danger_improvement']:+7.2f}  "
              f"gt={r['gt_danger']:.2f}  svo={r.get('svo_angle', 0):5.1f}°")
    print(f'\nResults saved: {out_path}')


# ============================================================
# Render mode
# ============================================================

def _plot_comparison(
    sample: EvalSample,
    polyline_segments: List[np.ndarray],
    save_path: Path,
) -> None:
    """2×2 publication figure with polyline background on the trajectory subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    delta = sample.gen_danger - sample.gt_danger
    margin_str = 'Met ✓' if delta >= MARGIN_THRESH else 'Not Met ✗'
    fig.suptitle(
        f'{sample.scenario_name}  |  SVO={sample.svo_angle:.1f}°  |  '
        f'ΔDanger={delta:+.2f}  (Margin {margin_str})',
        fontsize=11, fontweight='bold',
    )

    # --- [0,0] Trajectory + polyline background ---
    ax = axes[0, 0]
    for seg in polyline_segments:
        ax.plot(seg[:, 0], seg[:, 1], color='#cccccc', linewidth=0.6, zorder=1)
    ax.plot(sample.ego_history[:, 0], sample.ego_history[:, 1],
            'b-', lw=1.5, alpha=0.6, label='Ego History', zorder=2)
    ax.plot(sample.gt_trajectory[:, 0], sample.gt_trajectory[:, 1],
            'g-', lw=2.5, label=f'GT Future  (D={sample.gt_danger:.1f})', zorder=3)
    ax.plot(sample.gen_trajectory[:, 0], sample.gen_trajectory[:, 1],
            'r--', lw=2.5, label=f'Generated  (D={sample.gen_danger:.1f})', zorder=4)
    if sample.agent_history is not None:
        ax.plot(sample.agent_history[:, 0], sample.agent_history[:, 1],
                color='orange', lw=1.5, alpha=0.6, label='Agent History', zorder=2)
    if sample.agent_future is not None:
        ax.plot(sample.agent_future[:, 0], sample.agent_future[:, 1],
                color='darkorange', lw=1.5, ls='-.', alpha=0.7, label='Agent GT Future', zorder=2)
    if sample.gen_agent_future is not None:
        ax.plot(sample.gen_agent_future[:, 0], sample.gen_agent_future[:, 1],
                color='darkorange', lw=2.0, ls='-', alpha=0.9, label='Agent Generated', zorder=3)
    ax.scatter(sample.ego_history[-1, 0], sample.ego_history[-1, 1], c='blue', s=80, zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # --- [0,1] Speed profile ---
    ax = axes[0, 1]
    gt_spd = np.sqrt(sample.gt_trajectory[:, 3] ** 2 + sample.gt_trajectory[:, 4] ** 2)
    gen_spd = np.sqrt(sample.gen_trajectory[:, 3] ** 2 + sample.gen_trajectory[:, 4] ** 2)
    t = np.arange(60) * 0.1
    ax.plot(t, gt_spd, 'g-', lw=2, label='GT Speed')
    ax.plot(t, gen_spd, 'r--', lw=2, label='Generated Speed')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [1,0] Danger bar chart with improvement annotation ---
    ax = axes[1, 0]
    ax.bar(['GT', 'Generated'], [sample.gt_danger, sample.gen_danger],
           color=['#4CAF50', '#F44336'], alpha=0.85, edgecolor='black', lw=0.8)
    ax.axhline(y=sample.gt_danger + MARGIN_THRESH, color='navy', ls='--', lw=1.5,
               label=f'Target (GT+{MARGIN_THRESH})')
    top = max(sample.gt_danger, sample.gen_danger)
    ax.annotate(
        f'Δ={delta:+.2f}',
        xy=(1, sample.gen_danger), xytext=(0.5, top * 1.08),
        fontsize=11, fontweight='bold', color='darkred',
        arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
    )
    ax.set_ylabel('Danger Score')
    ax.set_title('Danger Scores')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- [1,1] Info text ---
    ax = axes[1, 1]
    m = sample.validity.get('metrics', {})
    lines = [
        f'Scenario: {sample.scenario_name}',
        f'SVO Angle: {sample.svo_angle:.1f}°',
        '',
        'Generated Metrics:',
        f'  Max Speed: {m.get("max_speed", 0):.2f} m/s',
        f'  Max Accel: {m.get("max_accel", 0):.2f} m/s²',
        f'  Max Jerk:  {m.get("max_jerk", 0):.2f} m/s³',
        '',
        'Validity:',
        f'  Speed:   {chr(10003) if sample.validity.get("speed_valid") else chr(10007)}',
        f'  Accel:   {chr(10003) if sample.validity.get("accel_valid") else chr(10007)}',
        f'  Jerk:    {chr(10003) if sample.validity.get("jerk_valid") else chr(10007)}',
        f'  Overall: {chr(10003) if sample.validity.get("overall_valid") else chr(10007)}',
        '',
        f'GT Danger:   {sample.gt_danger:.3f}',
        f'Gen Danger:  {sample.gen_danger:.3f}',
        f'Improvement: {delta:+.3f}',
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontsize=9, va='top', family='monospace')
    ax.axis('off')
    ax.set_title('Sample Info')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [Saved] {save_path}')


def _build_gt_ego_traj(ego_history: np.ndarray, ego_future_gt_inc: np.ndarray) -> np.ndarray:
    """Convert (60,2) Δx/Δy increments → (60,5) trajectory [x,y,heading,vx,vy].

    ego_future_gt_inc[0] is always (0,0) by preprocess design (ego at origin at t=0).
    At t=0 we therefore inherit history velocity to avoid a false speed=0 artifact.
    """
    last = ego_history[-1]  # [x, y, heading, vx, vy]
    pos = last[:2].copy()
    traj = np.zeros((len(ego_future_gt_inc), 5))
    for t, inc in enumerate(ego_future_gt_inc):
        pos = pos + inc
        traj[t, :2] = pos
        if t == 0:
            # inc[0]=(0,0) by design; use history vx/vy to avoid false speed spike
            traj[t, 2] = last[2]
            traj[t, 3] = last[3]
            traj[t, 4] = last[4]
        else:
            traj[t, 2] = float(np.arctan2(inc[1], inc[0]))
            traj[t, 3] = inc[0] / 0.1
            traj[t, 4] = inc[1] / 0.1
    return traj


def _simple_validity(traj: np.ndarray) -> dict:
    """Minimal validity dict from a (60,5) trajectory."""
    speed = np.sqrt(traj[:, 3]**2 + traj[:, 4]**2)
    accel = np.abs(np.diff(speed) / 0.1)
    jerk  = np.abs(np.diff(accel) / 0.1)
    max_accel = float(accel.max()) if len(accel) else 0.0
    return {
        'speed_valid': bool(speed.max() < 50),
        'accel_valid': bool(max_accel < 20),
        'jerk_valid':  bool(jerk.max() < 200 if len(jerk) else True),
        'overall_valid': bool(max_accel < 20),
        'metrics': {
            'max_speed': float(speed.max()),
            'max_accel': max_accel,
            'max_jerk':  float(jerk.max()) if len(jerk) else 0.0,
        },
    }


def cmd_render(args) -> None:
    """Render 2×2 publication figures for the specified scenario pkl files.

    If --model is supplied, uses DangerSceneGenerator (v3.19+) with k=args.sdedit_k
    and smooth_sigma=args.smooth_sigma.  Otherwise falls back to the legacy
    MatplotlibVisualizer pipeline.
    """
    output_dir = Path(args.output_dir) if args.output_dir else Path(_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model or _ACTIVE_MODEL

    # ---- NEW path: DangerSceneGenerator ----
    if args.model:
        from generate_danger_scenes import DangerSceneGenerator
        print(f'[Render] Using DangerSceneGenerator: {model_path}')
        generator = DangerSceneGenerator(model_path, device=args.device)
        for pkl_name in args.scenarios:
            p = Path(pkl_name)
            if p.is_absolute():
                pkl_path = p
            elif p.exists():
                pkl_path = p.resolve()
            else:
                pkl_path = Path(gnn_trans_root) / pkl_name
            if not pkl_path.exists():
                print(f'[Skip] Not found: {pkl_path}')
                continue
            print(f'\n[Render] {pkl_path.name}')

            raw = pd.read_pickle(pkl_path)
            ego_history    = np.array(raw['ego_history'])
            agent_history  = np.array(raw['key_agent_history'])
            agent_gt_future = np.array(raw['key_agent_future'])
            ego_future_inc  = np.array(raw['ego_future_gt'])

            result = generator.generate_from_pkl(
                str(pkl_path),
                num_samples=args.num_samples,
                use_sdedit=True,
                sdedit_k=args.sdedit_k,
                sdedit_guidance_scale=0.1,
                smooth_sigma=args.smooth_sigma,
            )
            gt_danger   = result['condition_info']['gt_danger']
            svo_angle   = float(result['condition_info'].get('svo_angle', 0.0))
            best_sample = max(result['samples'], key=lambda s: s['generated_danger'])
            gen_danger  = best_sample['generated_danger']
            gen_ego     = best_sample['generated_trajectory']   # (60,5)
            gen_agent   = best_sample['agent_trajectory']       # (60,5)

            gt_ego_traj = _build_gt_ego_traj(ego_history, ego_future_inc)
            validity    = _simple_validity(gen_ego)

            eval_sample = EvalSample(
                sample_idx=0,
                scenario_name=str(raw.get('scenario_name', pkl_path.stem)),
                ego_history=ego_history,
                agent_history=agent_history,
                gt_trajectory=gt_ego_traj,
                gen_trajectory=gen_ego,
                gt_danger=gt_danger,
                gen_danger=gen_danger,
                svo_angle=svo_angle,
                validity=validity,
                agent_future=agent_gt_future,
                gen_agent_future=gen_agent,
            )
            save_path = output_dir / f'{pkl_path.stem}_comparison.png'
            _plot_comparison(eval_sample, [], save_path)

        print(f'\n[Render Done] Output directory: {output_dir}')
        return

    # ---- LEGACY path: MatplotlibVisualizer ----
    print(f'[Render] Loading model (legacy): {model_path}')
    viz = MatplotlibVisualizer(
        model_path=model_path,
        data_dir=config.DIFFUSION_DANGER_PAIR_DIR,
        processed_data_dir=config.DIFFUSION_PROCESSED_DATA_DIR,
        device=args.device,
        min_gt_danger=0.0,
    )

    pkl_dir = Path(config.DIFFUSION_DANGER_PAIR_DIR)
    processed_dir = Path(config.DIFFUSION_PROCESSED_DATA_DIR)

    for pkl_name in args.scenarios:
        pkl_file = pkl_dir / pkl_name
        if not pkl_file.exists():
            print(f'[Skip] Not found: {pkl_file}')
            continue

        print(f'\n[Render] {pkl_name}')
        sample_data = viz.load_sample(pkl_file)

        gen_traj, gen_danger, gt_danger, _, best_inc = _inference_best_of_n(
            viz, sample_data, args.device,
            use_sdedit=args.sdedit, sdedit_k=args.sdedit_k,
            cumulative_space=_ACTIVE_CUMULATIVE,
        )

        gt_traj_data = recover_trajectory_from_increment(
            sample_data['ego_future_gt'][:, :2].flatten(),
            sample_data['start_pos'],
            sample_data['start_heading'],
            sample_data['start_vx'],
            sample_data['start_vy'],
        )

        joint = len(best_inc) == 240
        ego_inc = best_inc[:120] if joint else best_inc
        if _ACTIVE_BICYCLE:
            gen_traj_obj = bicycle_model_decode_np(ego_inc, sample_data['ego_init_state'])
        elif _ACTIVE_XY:
            gen_traj_obj = xy_displacement_decode_np(ego_inc, sample_data['start_pos'])
        elif _ACTIVE_CUMULATIVE:
            gen_traj_obj = kinematic_cumulative_decode_np(ego_inc, sample_data['start_pos'])
        else:
            ego_v0 = float(np.sqrt(sample_data['start_vx'] ** 2 + sample_data['start_vy'] ** 2))
            gen_traj_obj = kinematic_decode_np(
                ego_inc, ego_v0, sample_data['start_heading'], sample_data['start_pos'],
            )
        validity = check_kinematic_validity(gen_traj_obj)

        eval_sample = EvalSample(
            sample_idx=0,
            scenario_name=str(sample_data.get('scenario_name', pkl_file.stem)),
            ego_history=sample_data['ego_history'],
            agent_history=sample_data['agent_history'],
            gt_trajectory=gt_traj_data.full_trajectory,
            gen_trajectory=gen_traj,
            gt_danger=gt_danger,
            gen_danger=gen_danger,
            svo_angle=float(sample_data['svo_angle']),
            validity=validity,
            agent_future=sample_data.get('key_agent_future'),
        )

        polyline_segs = _load_polylines(pkl_file, processed_dir)
        print(f'  Lane segments: {len(polyline_segs)}')

        save_path = output_dir / f'{pkl_file.stem}_comparison.png'
        _plot_comparison(eval_sample, polyline_segs, save_path)

    print(f'\n[Render Done] Output directory: {output_dir}')


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Qualitative visualization for danger scene generation'
    )
    parser.add_argument('--mode', choices=['scan', 'render'], required=True,
                        help='scan: produce scan_results.json; render: draw paper figures')
    parser.add_argument('--scenarios', nargs='+', default=[],
                        help='(render only) pkl filenames to render')
    parser.add_argument('--device', default='cuda', help='torch device (default: cuda)')
    parser.add_argument('--sdedit', action='store_true',
                        help='use SDEdit inference (GT-anchored) instead of pure-noise baseline')
    parser.add_argument('--sdedit_k', type=int, default=20,
                        help='SDEdit noise level 1..99 (default 20; lower = closer to GT)')
    parser.add_argument('--model', default=None,
                        help='model checkpoint path; triggers DangerSceneGenerator render path')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='Gaussian sigma for vx/vy smoothing (0 = off; default 1.0)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='number of samples per scene (DangerSceneGenerator path; default 3)')
    parser.add_argument('--output_dir', default=None,
                        help='output directory (default: danger_gen_ablation/results/qualitative/)')
    args = parser.parse_args()

    if args.mode == 'render' and not args.scenarios:
        parser.error('--mode render requires --scenarios <file1> ...')

    if args.mode == 'scan':
        cmd_scan(args)
    else:
        cmd_render(args)


if __name__ == '__main__':
    main()
