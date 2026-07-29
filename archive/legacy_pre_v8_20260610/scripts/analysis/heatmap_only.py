#!/usr/bin/env python3
"""
Generate a performance heatmap from training_log.txt files.
- Collect final train/val loss and best ADE/FDE per experiment.
- Save heatmap to experiments/analysis/magnet_performance_heatmap.png.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config

# Global font: Times New Roman, larger size for IEEE readability
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

BASE_DIR = "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"
RESULTS_BASE_DIR = config.ALL_RESULTS_DIR
OUTPUT_DIR = config.ANALYSIS_RESULTS_DIR

SCENARIO_TYPES: List[str] = [
    "high_lateral_acceleration",
    "high_magnitude_speed",
    "medium_magnitude_speed",
    "near_multiple_vehicles",
    "traversing_intersection",
]

ADJACENCY_TYPES: List[str] = [
    "default_adjacency",
    "ego_only_adjacency",
    "nearby_only_adjacency",
    "no_ego_road_adjacency",
]


def ensure_output_dir(path: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def extract_training_data(log_file_path: str) -> Dict[str, List[float]]:
    """Extract epoch-wise metrics from a training_log.txt.

    Expected line format:
    Epoch XX/YY: Train=A.BBBB, Val=C.DDDD, ADE=E.EEm, FDE=F.FFm
    """
    if not os.path.exists(log_file_path):
        return {}

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return {}

    pattern = (
        r"Epoch\s+(\d+)/\d+:\s+Train=([\d.]+),\s+Val=([\d.]+),\s+ADE=([\d.]+)m,\s+FDE=([\d.]+)m"
    )
    matches = re.findall(pattern, content)
    if not matches:
        return {}

    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    ades: List[float] = []
    fdes: List[float] = []

    for m in matches:
        epoch_num, train_loss, val_loss, ade, fde = m
        epochs.append(int(epoch_num))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        ades.append(float(ade))
        fdes.append(float(fde))

    return {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "ades": ades,
        "fdes": fdes,
    }


def collect_all_training_data(base_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """Collect data from results_type_* folders.

    Label format: "<Scenario>_<Adjacency>" with title-cased words.
    """
    all_data: Dict[str, Dict[str, List[float]]] = {}

    for scenario in SCENARIO_TYPES:
        for adjacency in ADJACENCY_TYPES:
            folder_name = f"results_type_{scenario}_{adjacency}"
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # find a training_* subfolder with logs/training_log.txt
            training_subfolders = [
                os.path.join(folder_path, p) for p in os.listdir(folder_path)
                if p.startswith("training_") and os.path.isdir(os.path.join(folder_path, p))
            ]
            # Prefer latest by name for determinism, but any valid one works
            for sub in sorted(training_subfolders):
                log_file = os.path.join(sub, "logs", "training_log.txt")
                data = extract_training_data(log_file)
                if data:
                    scenario_label = scenario.replace("_", " ").title()
                    adjacency_label = adjacency.replace("_adjacency", "").replace("_", " ").title()
                    label = f"{scenario_label} & {adjacency_label}"
                    # Sanitize label: replace any hyphen-like characters with space and strip trailing hyphens/spaces
                    label = re.sub(r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]", " ", label)
                    label = label.rstrip(" -\u2013\u2014\u2010\u2011\u2212")
                    all_data[label] = data
                    break

    return all_data


def build_summary_dataframe(all_data: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """Build a summary DataFrame containing only Best ADE and Best FDE for each experiment."""
    rows: List[Dict[str, float]] = []
    index_labels: List[str] = []

    for label, data in all_data.items():
        if not (data.get("ades") and data.get("fdes")):
            continue
        best_ade = min(data["ades"]) if data["ades"] else np.nan
        best_fde = min(data["fdes"]) if data["fdes"] else np.nan

        rows.append({
            "Best ADE/m": best_ade,
            "Best FDE/m": best_fde,
        })
        index_labels.append(label)

    df = pd.DataFrame(rows, index=index_labels)
    return df


def plot_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Plot and save the performance heatmap. Returns saved path."""
    if df.empty:
        raise ValueError("No data available to plot heatmap.")

    fig, ax = plt.subplots(figsize=(12, 16))

    # Normalize values directly for visualization (no scaling beyond actual values)
    data = df.values
    im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto")

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=18, fontweight="bold")
    ax.set_yticklabels(df.index, fontsize=18)
    # Prevent wrapping/clipping artifacts on y tick labels and give more left margin
    for tick in ax.get_yticklabels():
        try:
            tick.set_wrap(False)
            tick.set_clip_on(False)
        except Exception:
            pass
    ax.tick_params(axis="y", which="major", pad=14)
    plt.subplots_adjust(left=0.32)

    # Annotate cells with values, larger font for readability
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(
                j,
                i,
                f"{data[i, j]:.4f}",
                ha="center",
                va="center",
                color="black",
                fontsize=16,
                fontweight="bold",
            )

    ax.set_title(
        "MAGNET Training Results Heatmap",
        fontsize=22,
        fontweight="bold",
        loc="center",
    )

    # Colorbar with larger tick labels
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    ensure_output_dir(output_dir)
    save_path = os.path.join(output_dir, "magnet_performance_heatmap.png")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    all_data = collect_all_training_data(RESULTS_BASE_DIR)
    if not all_data:
        print("No valid training data found.")
        return

    df = build_summary_dataframe(all_data)
    try:
        save_path = plot_heatmap(df, OUTPUT_DIR)
        print(f"Heatmap saved to: {save_path}")
    except Exception as exc:
        print(f"Failed to create heatmap: {exc}")


if __name__ == "__main__":
    main()
