"""
Unified evaluation entry point for adversarial baselines.

Usage:
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans

    # Random Perturbation
    python baselines/run_baseline_eval.py \\
        --baseline random \\
        --data_dir data/danger_pair_scenarios \\
        --num_samples 40 \\
        --output_dir experiments/danger_generation/baselines/random/

    # Gradient Ascent (CPU)
    python baselines/run_baseline_eval.py \\
        --baseline gradient \\
        --data_dir data/danger_pair_scenarios \\
        --num_samples 40 \\
        --output_dir experiments/danger_generation/baselines/gradient/ \\
        --device cpu

    # Gradient Ascent (GPU)
    python baselines/run_baseline_eval.py \\
        --baseline gradient \\
        --data_dir data/danger_pair_scenarios \\
        --num_samples 40 \\
        --output_dir experiments/danger_generation/baselines/gradient/ \\
        --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_GNN_ROOT = Path(__file__).resolve().parent.parent
if str(_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GNN_ROOT))

_REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "config.py").exists()),
    _GNN_ROOT,
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config


def build_baseline(args):
    if args.baseline == "random":
        from baselines.random_perturbation import RandomPerturbation
        return RandomPerturbation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            pos_sigma=args.pos_sigma,
            speed_sigma=args.speed_sigma,
            seed=args.seed,
        )
    elif args.baseline == "gradient":
        from baselines.gradient_ascent import GradientAscent
        return GradientAscent(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            lr=args.lr,
            n_steps=args.n_steps,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")


def print_summary(summary: dict, output_dir: str) -> None:
    n = summary["n_evaluated"]
    print("\n" + "=" * 55)
    print(f"  Baseline: {summary['baseline']}")
    print(f"  Samples evaluated: {n}  (skipped: {summary['n_skipped']})")
    print("=" * 55)
    print(f"  Margin Met Rate         : {summary['margin_met_rate']*100:.1f}%")
    print(f"  Mean GT Danger          : {summary['mean_gt_danger']:.2f}")
    print(f"  Mean Gen Danger         : {summary['mean_gen_danger']:.2f}")
    print(f"  Mean Danger Improvement : {summary['mean_danger_improvement']:+.2f}")
    print("-" * 55)
    print(f"  Collision Rate          : {summary['collision_rate']*100:.1f}%")
    print(f"  Mean Min Distance       : {summary['mean_min_distance']:.2f} m")
    ttc = summary["mean_ttc"]
    ttc_str = f"{ttc:.2f} s" if ttc != float("inf") else "inf"
    print(f"  Mean TTC (near-miss)    : {ttc_str}")
    print(f"  Near Miss Rate          : {summary['near_miss_rate']*100:.1f}%")
    print("=" * 55)
    print(f"  Results saved to: {Path(output_dir) / 'results.json'}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial trajectory baseline evaluation")

    parser.add_argument("--baseline", choices=["random", "gradient"], required=True)
    parser.add_argument(
        "--data_dir",
        default=config.DANGER_PAIR_SCENARIOS_DIR,
        help="Directory containing danger_pair_features_*.pkl files",
    )
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--output_dir", default=None, help="Output directory (default: experiments/danger_generation/baselines/<name>/)")

    # Random Perturbation args
    parser.add_argument("--pos_sigma", type=float, default=0.5, help="Position noise std (m)")
    parser.add_argument("--speed_sigma", type=float, default=1.0, help="Speed noise std (m/s)")
    parser.add_argument("--seed", type=int, default=42)

    # Gradient Ascent args
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate")
    parser.add_argument("--n_steps", type=int, default=200, help="Gradient ascent steps")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(config.DANGER_GENERATION_BASELINES_DIR) / args.baseline)

    baseline = build_baseline(args)
    summary = baseline.evaluate(num_samples=args.num_samples)
    print_summary(summary, args.output_dir)
