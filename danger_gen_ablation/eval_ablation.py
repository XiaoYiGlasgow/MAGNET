"""
消融评估脚本 - 危险场景生成

评估4组消融实验，输出对比表和 ablation_summary.json。

A1 no_danger_loss : danger_gen_ablation/results/no_danger_loss/best_model.pt
A2 no_svo         : danger_gen_ablation/results/no_svo/best_model.pt
A3 no_gnn         : danger_gen_ablation/results/no_gnn/best_model.pt
A4 N=1 inference  : experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt
                    （N=1 采样，无需重训）
Full (baseline)   : experiments/danger_generation/training_runs/full_training_v3_10/best_model.pt
                    （N=8 采样，参考值）

用法：
    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    conda activate nuplan
    python danger_gen_ablation/eval_ablation.py
    python danger_gen_ablation/eval_ablation.py --num_samples 40 --device cuda
    python danger_gen_ablation/eval_ablation.py --groups A1 A3 A4   # 只跑部分组
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

_ablation_dir = os.path.dirname(os.path.abspath(__file__))
gnn_trans_root = os.path.dirname(_ablation_dir)

sys.path.insert(0, gnn_trans_root)
sys.path.insert(0, os.path.join(gnn_trans_root, 'models'))
sys.path.insert(0, os.path.join(gnn_trans_root, 'utils'))

_models_path = '/mnt/local/Xiao_Yi/nuplan-devkit/models'
if _models_path not in sys.path:
    sys.path.insert(0, _models_path)

# MatplotlibVisualizer 在 visualization/ symlink 下
_viz_dir = os.path.join(gnn_trans_root, 'visualization')
if _viz_dir not in sys.path:
    sys.path.insert(0, _viz_dir)

import config
from matplotlib_visualizer import MatplotlibVisualizer


# ============================================================
# 消融组定义
# ============================================================

_v310_model = os.path.join(
    gnn_trans_root,
    'experiments', 'danger_generation', 'training_runs',
    'full_training_v3_10', 'best_model.pt'
)

ABLATION_GROUPS = {
    'Full': {
        'model_path': _v310_model,
        'num_inference_samples': 8,
        'description': 'Full model (v3.10, N=8) — baseline reference',
    },
    'A1': {
        'model_path': os.path.join(_ablation_dir, 'results', 'no_danger_loss', 'best_model.pt'),
        'num_inference_samples': 8,
        'description': 'A1 w/o DangerLoss (danger_weight=0)',
    },
    'A2': {
        'model_path': os.path.join(_ablation_dir, 'results', 'no_svo', 'best_model.pt'),
        'num_inference_samples': 8,
        'description': 'A2 w/o SVO (SVO input zeroed)',
    },
    'A3': {
        'model_path': os.path.join(_ablation_dir, 'results', 'no_gnn', 'best_model.pt'),
        'num_inference_samples': 8,
        'description': 'A3 w/o GNN (GNN branch zeroed)',
    },
    'A4': {
        'model_path': _v310_model,
        'num_inference_samples': 1,
        'description': 'A4 N=1 inference (v3.10 model, Best-of-1)',
    },
}

DATA_DIR      = config.DIFFUSION_DANGER_PAIR_DIR
PROCESSED_DIR = config.DIFFUSION_PROCESSED_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation evaluation for danger scene generation')
    parser.add_argument('--num_samples', type=int, default=40,
                        help='Number of scenes to evaluate per group (default 40)')
    parser.add_argument('--min_gt_danger', type=float, default=2.0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--groups', nargs='+', default=None,
                        choices=list(ABLATION_GROUPS.keys()),
                        help='Ablation groups to evaluate. Default: all groups.')
    return parser.parse_args()


def eval_group(group_name: str, group_cfg: dict, args) -> dict:
    """评估单个消融组，返回结果字典。"""
    model_path = group_cfg['model_path']
    n_infer    = group_cfg['num_inference_samples']

    if not os.path.exists(model_path):
        print(f'\n[{group_name}] SKIP — model not found: {model_path}')
        return {
            'group': group_name,
            'description': group_cfg['description'],
            'status': 'skipped',
            'model_path': model_path,
        }

    print(f'\n{"=" * 60}')
    print(f'[{group_name}] {group_cfg["description"]}')
    print(f'  model : {model_path}')
    print(f'  N     : {n_infer}')
    print('=' * 60)

    viz = MatplotlibVisualizer(
        model_path=model_path,
        data_dir=DATA_DIR,
        processed_data_dir=PROCESSED_DIR,
        device=args.device,
        min_gt_danger=args.min_gt_danger,
    )

    # 临时输出目录（不保存图片，只取统计值）
    tmp_output = os.path.join(
        _ablation_dir, 'results', group_name.lower(),
        f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )

    results = viz.evaluate_samples(
        num_samples=args.num_samples,
        output_dir=tmp_output,
        show_plots=False,
        num_inference_samples=n_infer,
    )

    row = {
        'group':              group_name,
        'description':        group_cfg['description'],
        'status':             'ok',
        'model_path':         model_path,
        'num_inference':      n_infer,
        'total_samples':      results.get('total_samples', 0),
        'margin_met_rate':    results.get('margin_met_rate', 0.0),
        'valid_rate':         results.get('valid_rate', 0.0),
        'danger_improvement': results.get('danger_improvement', 0.0),
        'mean_gt_danger':     results.get('mean_gt_danger', 0.0),
        'mean_gen_danger':    results.get('mean_gen_danger', 0.0),
        'collision_rate':     results.get('collision_rate', 0.0),
        'mean_min_dist':      results.get('mean_min_dist', 0.0),
    }
    return row


def print_table(rows: list):
    """打印对比结果表格。"""
    print('\n' + '=' * 90)
    print(f'{"Group":<10} {"Margin Met":>12} {"Valid Rate":>12} {"Danger Impr.":>14} '
          f'{"Collision":>10} {"N":>4}')
    print('-' * 90)
    for r in rows:
        if r.get('status') == 'skipped':
            print(f'{r["group"]:<10}  {"[SKIPPED]"}')
            continue
        print(f'{r["group"]:<10} '
              f'{r["margin_met_rate"]:>11.1%} '
              f'{r["valid_rate"]:>11.1%} '
              f'{r["danger_improvement"]:>+13.2f} '
              f'{r["collision_rate"]:>9.1%} '
              f'{r["num_inference"]:>4}')
    print('=' * 90)


def main():
    args = parse_args()

    groups_to_run = args.groups if args.groups else list(ABLATION_GROUPS.keys())
    print(f'Evaluating groups: {groups_to_run}')
    print(f'Samples per group: {args.num_samples}')
    print(f'Device: {args.device}')

    all_results = []
    for name in groups_to_run:
        cfg = ABLATION_GROUPS[name]
        row = eval_group(name, cfg, args)
        all_results.append(row)

    print_table(all_results)

    # 保存汇总 JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_samples': args.num_samples,
        'min_gt_danger': args.min_gt_danger,
        'results': all_results,
    }
    summary_path = Path(_ablation_dir) / 'results' / 'ablation_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSummary saved: {summary_path}')


if __name__ == '__main__':
    main()
