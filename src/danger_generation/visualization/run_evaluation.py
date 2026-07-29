"""
Diffusion Trajectory Evaluation Runner

Unified entry point for trajectory visualization and evaluation.

Usage:
    # Quick matplotlib validation
    python run_evaluation.py --mode quick --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 10

    # Bokeh/HTML visualization
    python run_evaluation.py --mode bokeh --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 5

    # Full evaluation (both modes)
    python run_evaluation.py --mode full --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 10

    # Generate and analyze statistics
    python run_evaluation.py --mode stats --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 100
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)
_root_dir = os.path.dirname(os.path.dirname(_parent_dir))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import config

from visualization.matplotlib_visualizer import MatplotlibVisualizer
from visualization.nuboard_visualizer import NuBoardVisualizer


def run_quick_validation(args):
    """Run matplotlib quick validation."""
    print("=" * 60)
    print("Diffusion Trajectory - Quick Validation")
    print("=" * 60)

    viz = MatplotlibVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        device=args.device,
        min_gt_danger=args.min_gt_danger,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"quick_validation_{timestamp}"

    results = viz.evaluate_samples(
        num_samples=args.num_samples,
        output_dir=str(output_dir),
        show_plots=args.show,
        num_inference_samples=args.num_inference_samples,
        use_sdedit=args.sdedit,
        sdedit_k=args.sdedit_k,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Samples evaluated: {results['total_samples']}")
    print(f"Margin met rate:   {results['margin_met_rate']:.1%}  ← 主指标")
    print(f"Mean GT danger:    {results['mean_gt_danger']:.2f}")
    print(f"--- Joint Risk: danger(ego_gen, agent_gen) ---")
    print(f"Mean Gen danger:   {results['mean_gen_danger']:.2f}")
    print(f"Danger improvement: {results['danger_improvement']:+.2f}")
    print(f"--- Ego-to-GT-Agent Risk: danger(ego_gen, agent_GT) ---")
    print(f"Mean Gen danger (vs GT agent): {results.get('mean_gen_danger_vs_gt_agent', float('nan')):.2f}")
    print(f"Danger improvement (vs GT agent): {results.get('danger_improvement_vs_gt_agent', float('nan')):+.2f}")
    print(f"---")
    print(f"Valid rate (≤80m/s, no NaN): {results['valid_rate']:.1%}")
    print(f"NaN rate:  {results.get('nan_rate', float('nan')):.1%}")
    print(f"Over 40m/s rate: {results.get('over40_rate', float('nan')):.1%}")
    print(f"Over 80m/s rate: {results.get('over80_rate', float('nan')):.1%}")
    print(f"\nResults saved to: {output_dir}")

    return results


def run_bokeh_visualization(args):
    """Run Bokeh/HTML visualization."""
    print("=" * 60)
    print("Diffusion Trajectory - Bokeh Visualization")
    print("=" * 60)

    viz = NuBoardVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"bokeh_visualization_{timestamp}"

    scenarios = viz.prepare_nuboard_data(
        num_samples=args.num_samples,
        output_dir=str(output_dir)
    )

    html_file = viz.create_bokeh_visualization(scenarios, str(output_dir))

    print("\n" + "=" * 60)
    print("Visualization Summary")
    print("=" * 60)
    print(f"Scenarios prepared: {len(scenarios)}")
    print(f"Margin met: {sum(1 for s in scenarios if s.gen_danger > s.gt_danger + 0.5)}")
    print(f"HTML file: {html_file}")
    print(f"\nOpen in browser: file://{html_file}")

    return scenarios


def run_real_nuboard(args):
    """Export diffusion outputs into a real NuBoard bundle and launch NuBoard."""
    print("=" * 60)
    print("Diffusion Trajectory - Real NuBoard Export")
    print("=" * 60)

    viz = NuBoardVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        nuplan_root=args.nuplan_root,
        device=args.device
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"nuboard_visualization_{timestamp}"

    scenarios = viz.prepare_nuboard_data(
        num_samples=args.num_samples,
        output_dir=str(output_dir)
    )
    bundle = viz.export_real_nuboard_bundle(
        scenarios=scenarios,
        output_dir=str(output_dir)
    )

    print("\n" + "=" * 60)
    print("NuBoard Bundle Summary")
    print("=" * 60)
    print(f"Scenarios prepared: {len(scenarios)}")
    print(f"NuBoard file: {bundle.nuboard_file}")
    print(f"Simulation dir: {bundle.simulations_dir}")
    print(f"Launching: http://localhost:{args.port}")

    viz.launch_real_nuboard(str(bundle.nuboard_file), port=args.port)

    return bundle


def run_full_evaluation(args):
    """Run both quick validation and Bokeh visualization."""
    print("=" * 60)
    print("Diffusion Trajectory - Full Evaluation")
    print("=" * 60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = Path(args.output_dir) / f"full_evaluation_{timestamp}"

    # Step 1: Quick validation with matplotlib
    print("\n[Step 1] Quick Validation...")
    quick_output = base_output_dir / "quick_validation"

    viz_quick = MatplotlibVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        device=args.device,
        min_gt_danger=args.min_gt_danger,
    )

    quick_results = viz_quick.evaluate_samples(
        num_samples=args.num_samples,
        output_dir=str(quick_output),
        show_plots=False,
        num_inference_samples=args.num_inference_samples,
        use_sdedit=args.sdedit,
        sdedit_k=args.sdedit_k,
    )

    # Step 2: Bokeh visualization for selected samples
    print("\n[Step 2] Bokeh Visualization...")
    bokeh_output = base_output_dir / "bokeh_visualization"

    viz_bokeh = NuBoardVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device
    )

    # Use fewer samples for Bokeh (more detailed)
    bokeh_samples = min(5, args.num_samples)
    scenarios = viz_bokeh.prepare_nuboard_data(
        num_samples=bokeh_samples,
        output_dir=str(bokeh_output)
    )

    html_file = viz_bokeh.create_bokeh_visualization(scenarios, str(bokeh_output))

    # Create combined report
    combined_report = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'quick_validation': {
            'num_samples': args.num_samples,
            'margin_met_rate': quick_results['margin_met_rate'],
            'valid_rate': quick_results['valid_rate'],
            'mean_gt_danger': quick_results['mean_gt_danger'],
            'mean_gen_danger': quick_results['mean_gen_danger'],
            'danger_improvement': quick_results['danger_improvement'],
            'output_dir': str(quick_output)
        },
        'bokeh_visualization': {
            'num_samples': bokeh_samples,
            'html_file': str(html_file),
            'output_dir': str(bokeh_output)
        }
    }

    report_file = base_output_dir / "combined_report.json"
    with open(report_file, 'w') as f:
        json.dump(combined_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Full Evaluation Summary")
    print("=" * 60)
    print(f"\nQuick Validation ({args.num_samples} samples):")
    print(f"  Margin met rate: {quick_results['margin_met_rate']:.1%}")
    print(f"  Valid rate: {quick_results['valid_rate']:.1%}")
    print(f"  Danger improvement: {quick_results['danger_improvement']:+.2f}")

    print(f"\nBokeh Visualization ({bokeh_samples} samples):")
    print(f"  HTML file: {html_file}")

    print(f"\nCombined report: {report_file}")
    print(f"Full output directory: {base_output_dir}")

    return combined_report


def run_statistics_analysis(args):
    """Run large-scale statistics analysis."""
    print("=" * 60)
    print("Diffusion Trajectory - Statistics Analysis")
    print("=" * 60)

    viz = MatplotlibVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        processed_data_dir=args.processed_data_dir,
        device=args.device,
        min_gt_danger=args.min_gt_danger,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"stats_analysis_{timestamp}"

    # Run evaluation with more samples
    results = viz.evaluate_samples(
        num_samples=args.num_samples,
        output_dir=str(output_dir),
        show_plots=False,
        num_inference_samples=args.num_inference_samples,
        use_sdedit=args.sdedit,
        sdedit_k=args.sdedit_k,
    )

    # Additional analysis
    import numpy as np

    gt_dangers = results['gt_dangers']
    gen_dangers = results['gen_dangers']
    improvements = [g - t for g, t in zip(gen_dangers, gt_dangers)]

    min_distances = results.get('min_distances', [])
    stats = {
        'num_samples':          results['total_samples'],
        'skipped':              results['skipped_count'],
        'margin_met_rate':      results['margin_met_rate'],
        'margin_met_raw_rate':  results.get('margin_met_raw_rate', float('nan')),
        'collision_rate':       results.get('collision_rate', float('nan')),
        'mean_min_distance':    float(np.mean(min_distances)) if min_distances else float('nan'),
        'valid_rate':           results['valid_rate'],
        'nan_rate':             results.get('nan_rate', float('nan')),
        'over40_rate':          results.get('over40_rate', float('nan')),
        'over80_rate':          results.get('over80_rate', float('nan')),
        'gt_danger': {
            'mean':   float(np.mean(gt_dangers)),
            'std':    float(np.std(gt_dangers)),
            'min':    float(np.min(gt_dangers)),
            'max':    float(np.max(gt_dangers)),
            'median': float(np.median(gt_dangers)),
        },
        'gen_danger': {
            'mean':   float(np.mean(gen_dangers)),
            'std':    float(np.std(gen_dangers)),
            'min':    float(np.min(gen_dangers)),
            'max':    float(np.max(gen_dangers)),
            'median': float(np.median(gen_dangers)),
        },
        'improvement': {
            'mean':          float(np.mean(improvements)),
            'std':           float(np.std(improvements)),
            'positive_rate': sum(1 for i in improvements if i > 0) / len(improvements),
        },
        'metrics_gt':  results['metrics_gt'],
        'metrics_gen': results['metrics_gen'],
    }

    # Save detailed stats
    stats_file = output_dir / "detailed_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Single unified summary print
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"  Evaluated: {stats['num_samples']}  Skipped: {stats['skipped']}")
    print(f"  Mean GT Danger:  {stats['gt_danger']['mean']:.2f} ± {stats['gt_danger']['std']:.2f}")
    print(f"  Mean Gen Danger: {stats['gen_danger']['mean']:.2f} ± {stats['gen_danger']['std']:.2f}  "
          f"(median {stats['gen_danger']['median']:.2f})")
    print()
    print(f"  Margin Met:        {stats['margin_met_rate']:.1%}  ← 主指标（排除碰撞）")
    print(f"  Margin Met (raw):  {stats['margin_met_raw_rate']:.1%}  ← 含碰撞，仅供参考")
    print(f"  Collision Rate:    {stats['collision_rate']:.1%}  ← 越低越好")
    print(f"  Mean Min Dist:     {stats['mean_min_distance']:.2f}m")
    print()
    print(f"  Valid (≤80m/s, no NaN): {stats['valid_rate']:.1%}")
    print(f"  Positive Improvement:   {stats['improvement']['positive_rate']:.1%}")
    print(f"\nDetailed stats: {stats_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Diffusion Trajectory Evaluation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (10 samples)
  python run_evaluation.py --mode quick --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 10

  # Bokeh visualization (5 samples)
  python run_evaluation.py --mode bokeh --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 5

  # Full evaluation (both modes)
  python run_evaluation.py --mode full --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 10

  # Statistics analysis (100 samples)
  python run_evaluation.py --mode stats --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 100

  # Export real NuBoard bundle and launch NuBoard
  python run_evaluation.py --mode nuboard --model_path experiments/danger_generation/training_runs/full_training/best_model.pt --num_samples 5 --port 5006
        """
    )

    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'bokeh', 'full', 'stats', 'nuboard'],
                        help='Evaluation mode')

    parser.add_argument('--model_path', type=str,
                        default=os.path.join(config.DIFFUSION_TRAINING_EXPERIMENTS_DIR, 'full_training', 'best_model.pt'),
                        help='Path to trained model checkpoint')

    parser.add_argument('--data_dir', type=str,
                        default=config.DANGER_PAIR_SCENARIOS_DIR,
                        help='Directory containing danger_pair pkl files')

    parser.add_argument('--processed_data_dir', type=str,
                        default=config.PROCESSED_DATA_DIR,
                        help='Directory containing processed graph files for gnn_trans checkpoints')

    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to evaluate')

    parser.add_argument('--output_dir', type=str, default=config.DIFFUSION_VISUALIZATION_RESULTS_DIR,
                        help='Base output directory')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    parser.add_argument('--nuplan_root', type=str,
                        default='/mnt/local/Xiao_Yi/nuplan-devkit',
                        help='Path to the local nuPlan devkit root')

    parser.add_argument('--port', type=int, default=5006,
                        help='Port used when launching real NuBoard')

    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (quick mode only)')

    parser.add_argument('--min_gt_danger', type=float, default=0.5,
                        help='Skip samples with GT outcome danger below this threshold. '
                             'Under the tent danger score (v4.0+), typical values are 0-8; '
                             '0.5 filters near-zero trivial samples. Set to 0.0 to disable filtering.')

    parser.add_argument('--num_inference_samples', type=int, default=8,
                        help='Best-of-N inference: generate this many diffusion candidates '
                             'per scene and keep the most dangerous one (default: 8).')

    parser.add_argument('--sdedit', action='store_true',
                        help='Use SDEdit inference (start from GT+noise) instead of pure noise. '
                             'Improves kinematic plausibility and reduces collision rate.')

    parser.add_argument('--sdedit_k', type=int, default=50,
                        help='SDEdit noise level k in [1, 99]. Lower → stays closer to GT; '
                             'higher → more freedom to deviate (default: 50).')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            print("[Warning] CUDA not available, using CPU")
            args.device = 'cpu'

    # Run appropriate mode
    if args.mode == 'quick':
        results = run_quick_validation(args)
    elif args.mode == 'bokeh':
        results = run_bokeh_visualization(args)
    elif args.mode == 'full':
        results = run_full_evaluation(args)
    elif args.mode == 'stats':
        results = run_statistics_analysis(args)
    elif args.mode == 'nuboard':
        results = run_real_nuboard(args)

    print("\n[Evaluation Complete]")

    return results


if __name__ == "__main__":
    main()
