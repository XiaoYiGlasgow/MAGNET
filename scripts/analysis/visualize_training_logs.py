#!/usr/bin/env python3
"""
Visualize training parameters from 20 training_log.txt files
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
RESULTS_BASE_DIR = config.ALL_RESULTS_DIR

def create_output_directory() -> str:
    """Create unified output directory"""
    output_dir = config.ANALYSIS_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_training_data(log_file_path: str) -> Dict[str, List[float]]:
    """Extract training data from training_log.txt file"""
    if not os.path.exists(log_file_path):
        return {}

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Match format: Epoch XX/YY: Train=X.XXXX, Val=X.XXXX, ADE=X.XXm, FDE=X.XXm
        pattern = r'Epoch\s+(\d+)/\d+:\s+Train=([\d.]+),\s+Val=([\d.]+),\s+ADE=([\d.]+)m,\s+FDE=([\d.]+)m'
        matches = re.findall(pattern, content)

        if not matches:
            return {}

        epochs = []
        train_losses = []
        val_losses = []
        ades = []
        fdes = []

        for match in matches:
            epoch_num, train_loss, val_loss, ade, fde = match
            epochs.append(int(epoch_num))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            ades.append(float(ade))
            fdes.append(float(fde))

        return {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'ades': ades,
            'fdes': fdes
        }
    except Exception as e:
        print(f"Parse failed: {log_file_path}, error: {e}")
        return {}

def collect_all_training_data() -> Dict[str, Dict[str, List[float]]]:
    """Collect training data from all 20 results folders"""
    base_dir = RESULTS_BASE_DIR

    scenario_types = [
        "high_lateral_acceleration",
        "high_magnitude_speed",
        "medium_magnitude_speed",
        "near_multiple_vehicles",
        "traversing_intersection"
    ]

    adjacency_types = [
        "default_adjacency",
        "ego_only_adjacency",
        "nearby_only_adjacency",
        "no_ego_road_adjacency"
    ]

    all_data = {}

    for scenario in scenario_types:
        for adjacency in adjacency_types:
            folder_name = f"results_type_{scenario}_{adjacency}"
            folder_path = os.path.join(base_dir, folder_name)

            if os.path.exists(folder_path):
                # Find training subfolder
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isdir(item_path) and item.startswith('training_'):
                        log_file = os.path.join(item_path, 'logs', 'training_log.txt')
                        data = extract_training_data(log_file)

                        if data:
                            # Create simplified label
                            scenario_short = scenario.replace('_', ' ').title()
                            adj_short = adjacency.replace('_adjacency', '').replace('_', ' ').title()
                            label = f"{scenario_short}_{adj_short}"
                            all_data[label] = data
                            print(f"Successfully extracted data: {label}")
                            break

    return all_data

def create_visualization(all_data: Dict[str, Dict[str, List[float]]], output_dir: str) -> None:
    """Create visualization of training parameter changes"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('GNN_Trans Training Parameters Summary (20 Experiments) - Full 100 Epochs',
                 fontsize=16, fontweight='bold')

    # Set color mapping
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_data)))

    # 1. Training loss changes
    ax1 = axes[0, 0]
    for i, (label, data) in enumerate(all_data.items()):
        if data.get('train_losses'):
            epochs = data['epochs']
            train_losses = data['train_losses']
            ax1.plot(epochs, train_losses,
                    color=colors[i], alpha=0.7, linewidth=1.5, label=label)

    ax1.set_title('Training Loss Changes (Full Training Process)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_xlim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # 2. Validation loss changes
    ax2 = axes[0, 1]
    for i, (label, data) in enumerate(all_data.items()):
        if data.get('val_losses'):
            epochs = data['epochs']
            val_losses = data['val_losses']
            ax2.plot(epochs, val_losses,
                    color=colors[i], alpha=0.7, linewidth=1.5, label=label)

    ax2.set_title('Validation Loss Changes (Full Training Process)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlim(0, 100)
    ax2.grid(True, alpha=0.3)

    # 3. ADE changes
    ax3 = axes[1, 0]
    for i, (label, data) in enumerate(all_data.items()):
        if data.get('ades'):
            epochs = data['epochs']
            ades = data['ades']
            ax3.plot(epochs, ades,
                    color=colors[i], alpha=0.7, linewidth=1.5, label=label)

    ax3.set_title('ADE (Average Displacement Error) Changes (Full Training Process)',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ADE (meters)')
    ax3.set_xlim(0, 100)
    ax3.grid(True, alpha=0.3)

    # 4. FDE changes
    ax4 = axes[1, 1]
    for i, (label, data) in enumerate(all_data.items()):
        if data.get('fdes'):
            epochs = data['epochs']
            fdes = data['fdes']
            ax4.plot(epochs, fdes,
                    color=colors[i], alpha=0.7, linewidth=1.5, label=label)

    ax4.set_title('FDE (Final Displacement Error) Changes (Full Training Process)',
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('FDE (meters)')
    ax4.set_xlim(0, 100)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'training_parameters_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")

    plt.close()

def create_summary_heatmap(all_data: Dict[str, Dict[str, List[float]]], output_dir: str) -> None:
    """Create heatmap of final performance metrics"""
    # Extract final performance metrics
    summary_data = []
    labels = []

    for label, data in all_data.items():
        if data.get('ades') and data.get('fdes') and data.get('train_losses') and data.get('val_losses'):
            summary_data.append([
                data['train_losses'][-1],  # Final training loss
                data['val_losses'][-1],    # Final validation loss
                min(data['ades']),         # Best ADE
                min(data['fdes'])          # Best FDE
            ])
            labels.append(label)

    if summary_data:
        df = pd.DataFrame(summary_data,
                         index=labels,
                         columns=['Final Train Loss', 'Final Val Loss', 'Best ADE', 'Best FDE'])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize data for color mapping
        data_normalized = df.values
        im = ax.imshow(data_normalized, cmap='RdYlBu_r', aspect='auto')

        # Set labels
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.index)

        # Add value annotations
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                text = ax.text(j, i, f'{data_normalized[i, j]:.4f}',
                             ha="center", va="center", color="black")

        ax.set_title('MAGNET Training Results Final Performance Metrics Heatmap',
                     fontsize=14, fontweight='bold')

        # Add colorbar
        plt.colorbar(im)
        plt.tight_layout()

        # Save heatmap
        heatmap_path = os.path.join(output_dir, 'performance_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to: {heatmap_path}")
        plt.close()

def create_summary_table(all_data: Dict[str, Dict[str, List[float]]], output_dir: str) -> None:
    """Create summary table"""
    summary_rows = []

    for label, data in all_data.items():
        if data.get('ades') and data.get('fdes') and data.get('train_losses') and data.get('val_losses'):
            final_epoch = max(data['epochs']) if data['epochs'] else 0
            best_ade = min(data['ades'])
            best_fde = min(data['fdes'])
            best_ade_epoch = data['epochs'][data['ades'].index(best_ade)]
            best_fde_epoch = data['epochs'][data['fdes'].index(best_fde)]

            summary_rows.append({
                'Experiment Name': label,
                'Final Epoch': final_epoch,
                'Best ADE': f"{best_ade:.4f}",
                'Best ADE Epoch': best_ade_epoch,
                'Best FDE': f"{best_fde:.4f}",
                'Best FDE Epoch': best_fde_epoch,
                'Final Train Loss': f"{data['train_losses'][-1]:.6f}",
                'Final Val Loss': f"{data['val_losses'][-1]:.6f}"
            })

    if summary_rows:
        df = pd.DataFrame(summary_rows)

        # Save CSV
        csv_path = os.path.join(output_dir, 'training_summary_table.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Summary table saved to: {csv_path}")

        # Print table
        print("\n" + "="*120)
        print("Training Results Summary Table")
        print("="*120)
        print(df.to_string(index=False))

def main():
    """Main function"""
    print("Starting to collect training data...")
    all_data = collect_all_training_data()

    if not all_data:
        print("No valid training data found")
        return

    print(f"Successfully collected data from {len(all_data)} experiments")

    # Create unified output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")

    # Create main parameter change visualization
    print("Creating training parameter visualization...")
    create_visualization(all_data, output_dir)

    # Create performance heatmap
    print("Creating performance heatmap...")
    create_summary_heatmap(all_data, output_dir)

    # Create summary table
    print("Creating summary table...")
    create_summary_table(all_data, output_dir)

    print("All visualizations and tables completed!")

if __name__ == "__main__":
    main()
