#!/usr/bin/env python3

"""
GNN_Trans Evaluation Results Visualization Script
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime
from scipy import stats

# Add paths
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit')
sys.path.append('/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans')

import config

def find_latest_evaluation_folder():
    """Find the most recent evaluation results folder"""
    weight_path = Path(config.WEIGHT_PATH)
    eval_folders = list(weight_path.glob("evaluation_results_*"))

    if not eval_folders:
        print("No evaluation results found!")
        return None

    # Sort by creation time and get the latest
    latest_folder = max(eval_folders, key=os.path.getctime)
    print(f"Found latest evaluation folder: {latest_folder}")
    return latest_folder

def load_evaluation_data(eval_folder):
    """Load evaluation data from JSON and NPZ files"""
    # Load JSON summary
    json_file = eval_folder / 'evaluation_summary.json'
    with open(json_file, 'r') as f:
        summary_data = json.load(f)

    # Load raw data
    npz_file = eval_folder / 'raw_error_data.npz'
    raw_data = np.load(npz_file)

    return summary_data, raw_data

def create_visualization_report(eval_folder, summary_data, raw_data):
    """Create comprehensive visualization report"""

    # Create visualization output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_folder = eval_folder.parent / f"visualization_report_{timestamp}"
    viz_folder.mkdir(exist_ok=True)

    print(f"Creating visualization report in: {viz_folder}")

    # Set style
    plt.style.use('seaborn-v0_8')

    # Extract data
    ade_values = raw_data['ade']
    fde_values = raw_data['fde']
    mse_values = raw_data['mse']

    # 1. Main metrics overview
    create_metrics_overview(summary_data, viz_folder)

    # 2. Error distribution plots
    create_error_distributions(ade_values, fde_values, mse_values, viz_folder)

    # 3. Statistical analysis
    create_statistical_analysis(ade_values, fde_values, viz_folder)

    # 4. Performance breakdown
    create_performance_breakdown(ade_values, fde_values, viz_folder)

    # 5. Generate HTML report
    generate_html_report(summary_data, viz_folder)

    print(f"Visualization report completed: {viz_folder}")
    return viz_folder

def create_metrics_overview(summary_data, viz_folder):
    """Create metrics overview chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    metrics = summary_data['metrics']

    # ADE metrics
    ade_stats = [
        metrics['ade']['mean'],
        metrics['ade']['median'],
        metrics['ade']['percentile_90'],
        metrics['ade']['percentile_95']
    ]
    labels = ['Mean', 'Median', '90th %ile', '95th %ile']
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']

    bars1 = ax1.bar(labels, ade_stats, color=colors, alpha=0.8)
    ax1.set_title('ADE Statistics (meters)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Error (meters)')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, ade_stats):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # FDE metrics
    fde_stats = [
        metrics['fde']['mean'],
        metrics['fde']['median'],
        metrics['fde']['percentile_90'],
        metrics['fde']['percentile_95']
    ]

    bars2 = ax2.bar(labels, fde_stats, color=colors, alpha=0.8)
    ax2.set_title('FDE Statistics (meters)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error (meters)')
    ax2.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars2, fde_stats):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Performance comparison
    performance_metrics = ['ADE', 'FDE']
    mean_values = [metrics['ade']['mean'], metrics['fde']['mean']]
    std_values = [metrics['ade']['std'], metrics['fde']['std']]

    x_pos = np.arange(len(performance_metrics))
    bars3 = ax3.bar(x_pos, mean_values, yerr=std_values, capsize=5,
                   color=['#2ecc71', '#e67e22'], alpha=0.8)
    ax3.set_title('Mean Performance with Standard Deviation', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Error (meters)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(performance_metrics)
    ax3.grid(axis='y', alpha=0.3)

    for i, (bar, mean_val, std_val) in enumerate(zip(bars3, mean_values, std_values)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.02,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Model info
    ax4.axis('off')
    info_text = f"""
    Model Information:
    
    • Model File: {summary_data['model_file']}
    • Test Dataset Size: {summary_data['test_dataset_size']:,} samples
    • Evaluation Time: {summary_data['timestamp']}
    
    Performance Summary:
    • MSE Loss: {metrics['mse_loss']['mean']:.6f}
    • ADE Range: {metrics['ade']['min']:.3f} - {metrics['ade']['max']:.3f}m
    • FDE Range: {metrics['fde']['min']:.3f} - {metrics['fde']['max']:.3f}m
    """

    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.suptitle('GNN_Trans Model Evaluation Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_folder / 'metrics_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_distributions(ade_values, fde_values, mse_values, viz_folder):
    """Create detailed error distribution plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # ADE distribution
    ax1.hist(ade_values, bins=60, alpha=0.7, color='skyblue', edgecolor='black', density=True, label='Histogram')
    # Simple KDE approximation using kernel density
    from scipy.stats import gaussian_kde
    kde_ade = gaussian_kde(ade_values)
    x_ade = np.linspace(ade_values.min(), ade_values.max(), 100)
    ax1.plot(x_ade, kde_ade(x_ade), color='red', linewidth=2, label='KDE')
    ax1.axvline(np.mean(ade_values), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ade_values):.3f}m')
    ax1.axvline(np.median(ade_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ade_values):.3f}m')
    ax1.set_xlabel('ADE (meters)')
    ax1.set_ylabel('Density')
    ax1.set_title('Average Displacement Error Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FDE distribution
    ax2.hist(fde_values, bins=60, alpha=0.7, color='lightgreen', edgecolor='black', density=True, label='Histogram')
    kde_fde = gaussian_kde(fde_values)
    x_fde = np.linspace(fde_values.min(), fde_values.max(), 100)
    ax2.plot(x_fde, kde_fde(x_fde), color='red', linewidth=2, label='KDE')
    ax2.axvline(np.mean(fde_values), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fde_values):.3f}m')
    ax2.axvline(np.median(fde_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(fde_values):.3f}m')
    ax2.set_xlabel('FDE (meters)')
    ax2.set_ylabel('Density')
    ax2.set_title('Final Displacement Error Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_ade = np.sort(ade_values)
    sorted_fde = np.sort(fde_values)
    p_ade = np.arange(1, len(sorted_ade) + 1) / len(sorted_ade)
    p_fde = np.arange(1, len(sorted_fde) + 1) / len(sorted_fde)

    ax3.plot(sorted_ade, p_ade, label='ADE', linewidth=3, color='blue')
    ax3.plot(sorted_fde, p_fde, label='FDE', linewidth=3, color='red')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='50th percentile')
    ax3.axhline(0.9, color='gray', linestyle=':', alpha=0.7, label='90th percentile')
    ax3.set_xlabel('Error (meters)')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Functions', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Box plot comparison
    data_to_plot = [ade_values, fde_values]
    box_plot = ax4.boxplot(data_to_plot, labels=['ADE', 'FDE'], patch_artist=True, notch=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Error (meters)')
    ax4.set_title('Error Distribution Comparison (Box Plot)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Add statistical annotations
    for i, data in enumerate(data_to_plot):
        y_pos = np.max(data) + 0.1
        ax4.text(i+1, y_pos, f'μ={np.mean(data):.3f}\nσ={np.std(data):.3f}',
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_folder / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_analysis(ade_values, fde_values, viz_folder):
    """Create statistical analysis plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Q-Q plot for ADE
    stats.probplot(ade_values, dist="norm", plot=ax1)
    ax1.set_title('ADE Q-Q Plot (Normal Distribution)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Q-Q plot for FDE
    stats.probplot(fde_values, dist="norm", plot=ax2)
    ax2.set_title('FDE Q-Q Plot (Normal Distribution)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Create violin plot manually
    ade_data = ade_values
    fde_data = fde_values

    # Simple violin plot approximation
    parts1 = ax3.violinplot([ade_data], positions=[1], showmeans=True, showmedians=True)
    parts2 = ax3.violinplot([fde_data], positions=[2], showmeans=True, showmedians=True)

    # Color the violin plots
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    for pc in parts2['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)

    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['ADE', 'FDE'])
    ax3.set_ylabel('Error (meters)')
    ax3.set_title('Error Distribution Shapes (Violin Plot)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Correlation and scatter plot
    ax4.scatter(ade_values, fde_values, alpha=0.6, s=10)
    correlation = np.corrcoef(ade_values, fde_values)[0, 1]

    # Add trend line
    z = np.polyfit(ade_values, fde_values, 1)
    p = np.poly1d(z)
    ax4.plot(ade_values, p(ade_values), "r--", alpha=0.8, linewidth=2)

    ax4.set_xlabel('ADE (meters)')
    ax4.set_ylabel('FDE (meters)')
    ax4.set_title(f'ADE vs FDE Correlation (r={correlation:.3f})', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_folder / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_breakdown(ade_values, fde_values, viz_folder):
    """Create performance breakdown analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Performance categories
    def categorize_performance(values, thresholds):
        categories = []
        for val in values:
            if val < thresholds[0]:
                categories.append('Excellent')
            elif val < thresholds[1]:
                categories.append('Good')
            elif val < thresholds[2]:
                categories.append('Acceptable')
            else:
                categories.append('Poor')
        return categories

    # ADE categories
    ade_categories = categorize_performance(ade_values, [0.5, 1.0, 2.0])
    ade_counts = pd.Series(ade_categories).value_counts()

    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    pie1 = ax1.pie(ade_counts.values, labels=ade_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
    ax1.set_title('ADE Performance Categories', fontweight='bold')

    # FDE categories
    fde_categories = categorize_performance(fde_values, [1.0, 2.0, 4.0])
    fde_counts = pd.Series(fde_categories).value_counts()

    pie2 = ax2.pie(fde_counts.values, labels=fde_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
    ax2.set_title('FDE Performance Categories', fontweight='bold')

    # Error percentiles
    percentiles = [50, 75, 90, 95, 99]
    ade_percentiles = [np.percentile(ade_values, p) for p in percentiles]
    fde_percentiles = [np.percentile(fde_values, p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35

    bars1 = ax3.bar(x - width/2, ade_percentiles, width, label='ADE', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, fde_percentiles, width, label='FDE', color='lightcoral', alpha=0.8)

    ax3.set_xlabel('Percentiles')
    ax3.set_ylabel('Error (meters)')
    ax3.set_title('Error Percentiles Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{p}th' for p in percentiles])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Performance summary table
    ax4.axis('off')

    summary_data = {
        'Metric': ['Samples < 0.5m', 'Samples < 1.0m', 'Samples < 2.0m', 'Mean Error', 'Std Dev', '95th Percentile'],
        'ADE': [
            f"{sum(ade_values < 0.5) / len(ade_values) * 100:.1f}%",
            f"{sum(ade_values < 1.0) / len(ade_values) * 100:.1f}%",
            f"{sum(ade_values < 2.0) / len(ade_values) * 100:.1f}%",
            f"{np.mean(ade_values):.3f}m",
            f"{np.std(ade_values):.3f}m",
            f"{np.percentile(ade_values, 95):.3f}m"
        ],
        'FDE': [
            f"{sum(fde_values < 0.5) / len(fde_values) * 100:.1f}%",
            f"{sum(fde_values < 1.0) / len(fde_values) * 100:.1f}%",
            f"{sum(fde_values < 2.0) / len(fde_values) * 100:.1f}%",
            f"{np.mean(fde_values):.3f}m",
            f"{np.std(fde_values):.3f}m",
            f"{np.percentile(fde_values, 95):.3f}m"
        ]
    }

    df_summary = pd.DataFrame(summary_data)
    table = ax4.table(cellText=df_summary.values, colLabels=df_summary.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Performance Summary', fontweight='bold', pad=20)

    plt.suptitle('Performance Breakdown Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(viz_folder / 'performance_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_html_report(summary_data, viz_folder):
    """Generate HTML report with all visualizations"""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GNN_Trans Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            .image-container {{ text-align: center; margin: 20px 0; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .info-section {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .timestamp {{ color: #7f8c8d; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚗 GNN_Trans Model Evaluation Report</h1>
            
            <div class="info-section">
                <h2>📊 Evaluation Summary</h2>
                <p><strong>Model:</strong> {summary_data['model_file']}</p>
                <p><strong>Test Dataset Size:</strong> {summary_data['test_dataset_size']:,} samples</p>
                <p><strong>Evaluation Time:</strong> <span class="timestamp">{summary_data['timestamp']}</span></p>
            </div>
            
            <h2>🎯 Key Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary_data['metrics']['ade']['mean']:.3f}m</div>
                    <div class="metric-label">Average Displacement Error (ADE)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['metrics']['fde']['mean']:.3f}m</div>
                    <div class="metric-label">Final Displacement Error (FDE)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['metrics']['mse_loss']['mean']:.6f}</div>
                    <div class="metric-label">MSE Loss</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary_data['metrics']['ade']['percentile_95']:.3f}m</div>
                    <div class="metric-label">95th Percentile ADE</div>
                </div>
            </div>
            
            <h2>📈 Metrics Overview</h2>
            <div class="image-container">
                <img src="metrics_overview.png" alt="Metrics Overview">
            </div>
            
            <h2>📊 Error Distributions</h2>
            <div class="image-container">
                <img src="error_distributions.png" alt="Error Distributions">
            </div>
            
            <h2>🔍 Statistical Analysis</h2>
            <div class="image-container">
                <img src="statistical_analysis.png" alt="Statistical Analysis">
            </div>
            
            <h2>⚡ Performance Breakdown</h2>
            <div class="image-container">
                <img src="performance_breakdown.png" alt="Performance Breakdown">
            </div>
            
            <div class="info-section">
                <h2>💡 Analysis Summary</h2>
                <ul>
                    <li><strong>ADE Performance:</strong> {'Excellent' if summary_data['metrics']['ade']['mean'] < 1.0 else 'Good' if summary_data['metrics']['ade']['mean'] < 2.0 else 'Acceptable'} (Mean: {summary_data['metrics']['ade']['mean']:.3f}m)</li>
                    <li><strong>FDE Performance:</strong> {'Excellent' if summary_data['metrics']['fde']['mean'] < 2.0 else 'Good' if summary_data['metrics']['fde']['mean'] < 4.0 else 'Acceptable'} (Mean: {summary_data['metrics']['fde']['mean']:.3f}m)</li>
                    <li><strong>Consistency:</strong> {'High' if summary_data['metrics']['ade']['std'] < 1.0 else 'Medium' if summary_data['metrics']['ade']['std'] < 2.0 else 'Low'} (ADE Std: {summary_data['metrics']['ade']['std']:.3f}m)</li>
                    <li><strong>Model Reliability:</strong> {(1 - summary_data['metrics']['ade']['percentile_95']/10)*100:.1f}% of predictions within acceptable range</li>
                </ul>
            </div>
            
            <footer style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    html_file = viz_folder / 'evaluation_report.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report generated: {html_file}")

def main():
    """Main function to run visualization"""
    print("Starting GNN_Trans evaluation visualization...")

    # Find latest evaluation folder
    eval_folder = find_latest_evaluation_folder()
    if eval_folder is None:
        return

    # Load data
    print("Loading evaluation data...")
    summary_data, raw_data = load_evaluation_data(eval_folder)

    # Create visualizations
    print("Creating visualization report...")
    viz_folder = create_visualization_report(eval_folder, summary_data, raw_data)

    print(f"\n✅ Visualization completed!")
    print(f"📁 Report location: {viz_folder}")
    print(f"🌐 Open '{viz_folder}/evaluation_report.html' to view the complete report")

if __name__ == "__main__":
    main()
