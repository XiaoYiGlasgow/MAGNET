"""
Classifier Guidance baseline evaluation entry point.

Usage:
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

    python baselines/classifier_guidance/run_cg_eval.py \\
        --model_path experiments/danger_generation/training_runs/full_training/best_model.pt \\
        --data_dir   data/danger_pair_scenarios \\
        --num_samples 40 \\
        --output_dir  experiments/danger_generation/baselines/classifier_guidance/ \\
        --guidance_scale 50.0 \\
        --guide_every    10 \\
        --device cpu

For GPU:
    --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_GNN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

_REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "config.py").exists()),
    _GNN_ROOT,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(description="Classifier Guidance baseline evaluation")
    p.add_argument("--model_path",    required=True,  help="Path to .pt checkpoint")
    p.add_argument("--data_dir",      default=config.DANGER_PAIR_SCENARIOS_DIR)
    p.add_argument("--num_samples",   type=int,   default=40)
    p.add_argument("--output_dir",    default=str(Path(config.DANGER_GENERATION_BASELINES_DIR) / "classifier_guidance"))
    p.add_argument("--guidance_scale",type=float, default=50.0, help="Gradient scale at each guided step")
    p.add_argument("--guide_every",   type=int,   default=10,   help="Apply guidance every N denoising steps")
    p.add_argument("--device",        default="cpu")
    return p.parse_args()


def print_summary(summary: dict, output_dir: str) -> None:
    n = summary["n_evaluated"]
    print("\n" + "=" * 58)
    print(f"  Baseline : {summary['baseline']}")
    print(f"  Samples  : {n}  (skipped: {summary['n_skipped']})")
    print("=" * 58)
    print(f"  Margin Met Rate         : {summary['margin_met_rate']*100:.1f}%")
    print(f"  Mean GT Danger          : {summary['mean_gt_danger']:.2f}")
    print(f"  Mean Gen Danger         : {summary['mean_gen_danger']:.2f}")
    print(f"  Mean Danger Improvement : {summary['mean_danger_improvement']:+.2f}")
    print("-" * 58)
    print(f"  Collision Rate          : {summary['collision_rate']*100:.1f}%")
    print(f"  Mean Min Distance       : {summary['mean_min_distance']:.2f} m")
    ttc = summary["mean_ttc"]
    ttc_str = f"{ttc:.2f} s" if ttc != float("inf") else "inf"
    print(f"  Mean TTC (near-miss)    : {ttc_str}")
    print(f"  Near Miss Rate          : {summary['near_miss_rate']*100:.1f}%")
    print("=" * 58)
    print(f"  Results saved to        : {Path(output_dir) / 'results.json'}")
    print()


if __name__ == "__main__":
    args = parse_args()

    from baselines.classifier_guidance.classifier_guidance import ClassifierGuidanceBaseline

    baseline = ClassifierGuidanceBaseline(
        model_path     = args.model_path,
        data_dir       = args.data_dir,
        output_dir     = args.output_dir,
        guidance_scale = args.guidance_scale,
        guide_every    = args.guide_every,
        device         = args.device,
    )

    summary = baseline.evaluate(num_samples=args.num_samples)
    print_summary(summary, args.output_dir)
