"""
Inspect generated trajectories from OK samples in detail.
Prints positions, speeds, inter-vehicle distance to judge physical plausibility.
"""
import os, sys
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from generate_danger_scenes import DangerSceneGenerator

MODEL_PATH  = "experiments/danger_generation/training_runs/v3_17_danger_norm/best_model.pt"
DT          = 0.1   # 10 Hz
PRINT_STEP  = 5     # print every N frames

OK_PKLS = [
    "diffusion_danger_model/dataset/danger_pairs/danger_pair_0017c8d4340958ae_agent19.pkl",
    "diffusion_danger_model/dataset/danger_pairs/danger_pair_048151552d335ff0_agent9.pkl",
    "diffusion_danger_model/dataset/danger_pairs/danger_pair_21a734875ff2546e_agent1.pkl",
]


def traj_stats(traj_np: np.ndarray, label: str):
    """traj_np: (60, 5) [x, y, heading, vx, vy]"""
    x, y = traj_np[:, 0], traj_np[:, 1]
    speed = np.sqrt(traj_np[:, 3]**2 + traj_np[:, 4]**2)
    dx = np.diff(x); dy = np.diff(y)
    step_dist = np.sqrt(dx**2 + dy**2)
    accel = np.diff(speed) / DT

    print(f"  [{label}]")
    print(f"    位置范围: x=[{x.min():.2f}, {x.max():.2f}]  y=[{y.min():.2f}, {y.max():.2f}]")
    print(f"    速度(m/s): mean={speed.mean():.2f}  max={speed.max():.2f}  min={speed.min():.2f}")
    print(f"    加速度(m/s²): max={np.abs(accel).max():.2f}  mean={np.abs(accel).mean():.2f}")
    print(f"    每步位移(m): mean={step_dist.mean():.3f}  max={step_dist.max():.3f}")

    # print sampled positions
    print(f"    轨迹采样 (每{PRINT_STEP}帧):")
    for i in range(0, 60, PRINT_STEP):
        print(f"      t={i*DT:.1f}s  x={x[i]:.2f}  y={y[i]:.2f}  v={speed[i]:.2f}m/s")


def inter_vehicle_distance(ego_np, agent_np):
    """Compute frame-by-frame distance between ego and agent."""
    dx = ego_np[:, 0] - agent_np[:, 0]
    dy = ego_np[:, 1] - agent_np[:, 1]
    return np.sqrt(dx**2 + dy**2)


def main():
    model_path = os.path.join(project_dir, MODEL_PATH)
    generator  = DangerSceneGenerator(model_path, device="cuda")

    for pkl_rel in OK_PKLS:
        pkl_path = os.path.join(project_dir, pkl_rel)
        name     = os.path.basename(pkl_path)
        print(f"\n{'='*60}")
        print(f"场景: {name}")
        print(f"{'='*60}")

        result = generator.generate_from_pkl(
            pkl_path,
            use_sdedit=True,
            sdedit_k=30,
            sdedit_guidance_scale=0.1,
            num_samples=3,
        )

        gt_danger = result['condition_info']['gt_danger']
        samples   = result['samples']

        # pick best sample
        best_idx  = max(range(len(samples)), key=lambda i: samples[i]['generated_danger'])
        best      = samples[best_idx]
        danger    = best['generated_danger']

        print(f"gt_danger={gt_danger:.3f}  generated_danger={danger:.3f}  (样本{best_idx})")

        ego_traj   = best['generated_trajectory']  # (60, 5)
        agent_traj = best['agent_trajectory']      # (60, 5)

        traj_stats(ego_traj,   "ego(生成)")
        traj_stats(agent_traj, "agent(生成)")

        dist = inter_vehicle_distance(ego_traj, agent_traj)
        print(f"  [车间距离]")
        print(f"    min={dist.min():.2f}m  mean={dist.mean():.2f}m  初始={dist[0]:.2f}m")
        close_frames = np.where(dist < 5.0)[0]
        if len(close_frames):
            print(f"    < 5m 的帧: {len(close_frames)} 帧, 最近={dist.min():.2f}m @ t={close_frames[0]*DT:.1f}s")
        else:
            print(f"    始终 >= 5m (无近距离交互)")

        # GT comparison
        print(f"\n  [对比GT]")
        raw = __import__('pandas').read_pickle(pkl_path)
        ego_gt_inc = raw['ego_future_gt']   # (60, 2) increments
        ego_lp = raw['ego_history'][-1, :2]
        ego_gt_pos = np.cumsum(ego_gt_inc, axis=0) + ego_lp
        agent_gt = raw['key_agent_future'][:, :2]  # (60, 2) absolute pos
        dist_gt = np.sqrt((ego_gt_pos[:,0]-agent_gt[:,0])**2 + (ego_gt_pos[:,1]-agent_gt[:,1])**2)
        print(f"    GT 车间距: min={dist_gt.min():.2f}m  mean={dist_gt.mean():.2f}m")
        ego_gt_speed = np.sqrt(ego_gt_inc[:,0]**2 + ego_gt_inc[:,1]**2) / DT
        print(f"    GT ego速度: mean={ego_gt_speed.mean():.2f}  max={ego_gt_speed.max():.2f} m/s")


if __name__ == "__main__":
    main()
