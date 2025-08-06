#!/usr/bin/env python3

"""
GNN_Trans Model Evaluation Script
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

import config
from models.transformer_predictor import TransformerPredictor
from utils.dataset import GraphDataset

def restore_trajectory_from_increments(origin, increments):
    """
    将增量轨迹还原为绝对轨迹

    Args:
        origin: 起始绝对位置 [batch_size, 2]
        increments: 增量序列 [batch_size, time_steps, 2]

    Returns:
        absolute_trajectory: 绝对轨迹 [batch_size, time_steps, 2]
    """
    batch_size, time_steps, _ = increments.shape
    absolute_trajectory = torch.zeros_like(increments)

    # 第一个点：起始位置 + 第一个增量
    absolute_trajectory[:, 0, :] = origin + increments[:, 0, :]

    # 后续点：累加增量
    for i in range(1, time_steps):
        absolute_trajectory[:, i, :] = absolute_trajectory[:, i-1, :] + increments[:, i, :]

    return absolute_trajectory

def evaluate_model():
    """Evaluate model performance"""
    print("Starting GNN_Trans model evaluation...")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create evaluation results folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = Path(config.WEIGHT_PATH) / f"evaluation_results_{timestamp}"
    results_folder.mkdir(exist_ok=True)
    print(f"Results will be saved to: {results_folder}")

    # Load test data
    test_dataset = GraphDataset(config.TEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Load model
    model = TransformerPredictor().to(device)

    # Find latest model weights
    weight_path = Path(config.WEIGHT_PATH)
    weight_files = list(weight_path.glob("*.pth"))

    if not weight_files:
        print("No trained model weights found")
        return

    latest_weight = max(weight_files, key=os.path.getctime)
    print(f"Loading model weights: {latest_weight}")

    model.load_state_dict(torch.load(latest_weight, map_location=device))
    model.eval()

    # Evaluation metrics
    total_loss = 0.0
    ade_list = []  # Average Displacement Error
    fde_list = []  # Final Displacement Error
    mse_list = []  # Per-sample MSE
    num_samples = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)

            # Get ground truth trajectory
            batch_size = data.y.shape[0]
            y_true = data.y.view(batch_size, -1).to(device)

            # MSE loss (still calculated on increments)
            mse_loss = torch.nn.functional.mse_loss(output, y_true, reduction='none')
            batch_mse = torch.mean(mse_loss.view(batch_size, -1), dim=1)
            mse_list.extend(batch_mse.cpu().numpy())

            total_loss += torch.sum(batch_mse).item()

            # ADE/FDE calculation: restore to absolute trajectories first
            pred_increments = output.view(batch_size, -1, 2)  # [batch, time_steps, 2]
            true_increments = y_true.view(batch_size, -1, 2)

            # Assume starting position is origin (0, 0) - because data is relative increments
            origin = torch.zeros(batch_size, 2, device=device)

            # Restore to absolute trajectories
            pred_absolute = restore_trajectory_from_increments(origin, pred_increments)
            true_absolute = restore_trajectory_from_increments(origin, true_increments)

            # Calculate ADE and FDE for absolute trajectories
            displacement_errors = torch.norm(pred_absolute - true_absolute, dim=2)  # [batch, time_steps]
            ade = torch.mean(displacement_errors, dim=1)  # [batch]
            fde = displacement_errors[:, -1]  # [batch]

            ade_list.extend(ade.cpu().numpy())
            fde_list.extend(fde.cpu().numpy())

            num_samples += batch_size

            if i % 50 == 0:
                print(f"Evaluated: {i * 16}/{len(test_dataset)} samples")

    # Calculate final metrics
    avg_mse = total_loss / num_samples
    avg_ade = np.mean(ade_list)
    avg_fde = np.mean(fde_list)
    std_ade = np.std(ade_list)
    std_fde = np.std(fde_list)

    print(f"\nEvaluation Results:")
    print(f"   MSE Loss: {avg_mse:.6f}")
    print(f"   ADE: {avg_ade:.4f} ± {std_ade:.4f} meters")
    print(f"   FDE: {avg_fde:.4f} ± {std_fde:.4f} meters")

    # Save detailed evaluation results
    results = {
        'timestamp': timestamp,
        'model_file': str(latest_weight.name),
        'test_dataset_size': int(num_samples),
        'metrics': {
            'mse_loss': {
                'mean': float(avg_mse),
                'std': float(np.std(mse_list))
            },
            'ade': {
                'mean': float(avg_ade),
                'std': float(std_ade),
                'min': float(np.min(ade_list)),
                'max': float(np.max(ade_list)),
                'median': float(np.median(ade_list)),
                'percentile_90': float(np.percentile(ade_list, 90)),
                'percentile_95': float(np.percentile(ade_list, 95))
            },
            'fde': {
                'mean': float(avg_fde),
                'std': float(std_fde),
                'min': float(np.min(fde_list)),
                'max': float(np.max(fde_list)),
                'median': float(np.median(fde_list)),
                'percentile_90': float(np.percentile(fde_list, 90)),
                'percentile_95': float(np.percentile(fde_list, 95))
            }
        },
        'raw_data': {
            'ade_values': [float(x) for x in ade_list],
            'fde_values': [float(x) for x in fde_list],
            'mse_values': [float(x) for x in mse_list]
        }
    }

    # Save main results
    result_file = results_folder / 'evaluation_summary.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to: {result_file}")

    # Save raw data separately
    raw_data_file = results_folder / 'raw_error_data.npz'
    np.savez(raw_data_file,
             ade=np.array(ade_list),
             fde=np.array(fde_list),
             mse=np.array(mse_list))
    print(f"Raw data saved to: {raw_data_file}")

    # Plot error distribution
    plot_error_distribution(ade_list, fde_list, mse_list, results_folder)

    # Generate evaluation report
    generate_evaluation_report(results, results_folder)

    return results

def plot_error_distribution(ade_list, fde_list, mse_list, results_folder):
    """Plot comprehensive error distribution"""

    # Main error distribution plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # ADE distribution
    ax1.hist(ade_list, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('ADE (meters)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Average Displacement Error Distribution')
    ax1.axvline(np.mean(ade_list), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(ade_list):.3f}m')
    ax1.axvline(np.median(ade_list), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(ade_list):.3f}m')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FDE distribution
    ax2.hist(fde_list, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('FDE (meters)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Final Displacement Error Distribution')
    ax2.axvline(np.mean(fde_list), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(fde_list):.3f}m')
    ax2.axvline(np.median(fde_list), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(fde_list):.3f}m')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # MSE distribution
    ax3.hist(mse_list, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('MSE Loss')
    ax3.set_ylabel('Frequency')
    ax3.set_title('MSE Loss Distribution')
    ax3.axvline(np.mean(mse_list), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(mse_list):.6f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Box plot comparison
    ax4.boxplot([ade_list, fde_list], labels=['ADE', 'FDE'])
    ax4.set_ylabel('Error (meters)')
    ax4.set_title('Error Distribution Comparison')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = results_folder / 'error_distribution_comprehensive.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Error distribution plot saved to: {plot_path}")
    plt.close()

    # Create cumulative distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sorted_ade = np.sort(ade_list)
    sorted_fde = np.sort(fde_list)
    p_ade = np.arange(1, len(sorted_ade) + 1) / len(sorted_ade)
    p_fde = np.arange(1, len(sorted_fde) + 1) / len(sorted_fde)

    ax.plot(sorted_ade, p_ade, label='ADE', linewidth=2)
    ax.plot(sorted_fde, p_fde, label='FDE', linewidth=2)
    ax.set_xlabel('Error (meters)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Prediction Errors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cdf_path = results_folder / 'error_cdf.png'
    plt.savefig(cdf_path, dpi=300, bbox_inches='tight')
    print(f"CDF plot saved to: {cdf_path}")
    plt.close()

def generate_evaluation_report(results, results_folder):
    """Generate a text report of evaluation results"""

    report_content = f"""
# GNN_Trans Model Evaluation Report

## Evaluation Details
- **Timestamp**: {results['timestamp']}
- **Model File**: {results['model_file']}
- **Test Dataset Size**: {results['test_dataset_size']} samples

## Performance Metrics

### Average Displacement Error (ADE)
- **Mean**: {results['metrics']['ade']['mean']:.4f} ± {results['metrics']['ade']['std']:.4f} meters
- **Median**: {results['metrics']['ade']['median']:.4f} meters
- **Min/Max**: {results['metrics']['ade']['min']:.4f} / {results['metrics']['ade']['max']:.4f} meters
- **90th Percentile**: {results['metrics']['ade']['percentile_90']:.4f} meters
- **95th Percentile**: {results['metrics']['ade']['percentile_95']:.4f} meters

### Final Displacement Error (FDE)
- **Mean**: {results['metrics']['fde']['mean']:.4f} ± {results['metrics']['fde']['std']:.4f} meters
- **Median**: {results['metrics']['fde']['median']:.4f} meters
- **Min/Max**: {results['metrics']['fde']['min']:.4f} / {results['metrics']['fde']['max']:.4f} meters
- **90th Percentile**: {results['metrics']['fde']['percentile_90']:.4f} meters
- **95th Percentile**: {results['metrics']['fde']['percentile_95']:.4f} meters

### MSE Loss
- **Mean**: {results['metrics']['mse_loss']['mean']:.6f} ± {results['metrics']['mse_loss']['std']:.6f}

## Performance Analysis

### Quality Assessment
"""

    # Add quality assessment
    ade_mean = results['metrics']['ade']['mean']
    fde_mean = results['metrics']['fde']['mean']

    if ade_mean < 1.0:
        report_content += "- **ADE**: Excellent performance (< 1.0m)\n"
    elif ade_mean < 2.0:
        report_content += "- **ADE**: Good performance (1.0-2.0m)\n"
    elif ade_mean < 5.0:
        report_content += "- **ADE**: Acceptable performance (2.0-5.0m)\n"
    else:
        report_content += "- **ADE**: Poor performance (> 5.0m)\n"

    if fde_mean < 2.0:
        report_content += "- **FDE**: Excellent performance (< 2.0m)\n"
    elif fde_mean < 4.0:
        report_content += "- **FDE**: Good performance (2.0-4.0m)\n"
    elif fde_mean < 8.0:
        report_content += "- **FDE**: Acceptable performance (4.0-8.0m)\n"
    else:
        report_content += "- **FDE**: Poor performance (> 8.0m)\n"

    report_content += f"""
### Consistency Analysis
- **ADE Standard Deviation**: {results['metrics']['ade']['std']:.4f}m
- **FDE Standard Deviation**: {results['metrics']['fde']['std']:.4f}m
- **Prediction Consistency**: {"High" if results['metrics']['ade']['std'] < 1.0 else "Medium" if results['metrics']['ade']['std'] < 2.0 else "Low"}

## Files Generated
- `evaluation_summary.json`: Complete evaluation results
- `raw_error_data.npz`: Raw error data for further analysis
- `error_distribution_comprehensive.png`: Error distribution plots
- `error_cdf.png`: Cumulative distribution functions
- `evaluation_report.md`: This report
"""

    # Save report
    report_path = results_folder / 'evaluation_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Evaluation report saved to: {report_path}")

if __name__ == "__main__":
    evaluate_model()
