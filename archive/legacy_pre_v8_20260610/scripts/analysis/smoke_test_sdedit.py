"""
SDEdit smoke test: compare generated_danger vs gt_danger on a few pkl files.
Usage:
  python scripts/analysis/smoke_test_sdedit.py
  python scripts/analysis/smoke_test_sdedit.py --model danger_gen_ablation/results/no_danger_loss/best_model.pt
"""
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from generate_danger_scenes import DangerSceneGenerator

DEFAULT_MODEL = "experiments/danger_generation/training_runs/v3_17_danger_norm/best_model.pt"
PKL_DIR    = "diffusion_danger_model/dataset/danger_pairs"
N_FILES    = 5
NUM_SAMPLES = 3
SDEDIT_K   = 20
GUIDANCE   = 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=DEFAULT_MODEL, help='relative path to model checkpoint')
    args = parser.parse_args()

    model_path = os.path.join(project_dir, args.model)
    pkl_dir    = os.path.join(project_dir, PKL_DIR)
    print(f"Model: {args.model}\n")

    import glob
    pkls = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))[:N_FILES]
    print(f"Found {len(pkls)} pkl files, testing first {N_FILES}\n")

    generator = DangerSceneGenerator(model_path, device="cuda")

    for pkl_path in pkls:
        name = os.path.basename(pkl_path)
        try:
            result = generator.generate_from_pkl(
                pkl_path,
                use_sdedit=True,
                sdedit_k=SDEDIT_K,
                sdedit_guidance_scale=GUIDANCE,
                num_samples=NUM_SAMPLES,
            )
            gt_danger  = result['condition_info']['gt_danger']
            best_danger = max(s['generated_danger'] for s in result['samples'])
            dangers     = [round(s['generated_danger'], 3) for s in result['samples']]
            improved    = best_danger > gt_danger

            print(f"[{'OK' if improved else 'FAIL'}] {name}")
            print(f"      gt_danger={gt_danger:.3f}  generated={dangers}  best={best_danger:.3f}  delta={best_danger-gt_danger:+.3f}")
        except Exception as e:
            import traceback
            print(f"[ERR] {name}: {e}")
            traceback.print_exc()
        print()

if __name__ == "__main__":
    main()
