#!/usr/bin/env python3
"""
Launch nuboard to visualize GNN_Trans results
"""

import os
import subprocess
import sys
from pathlib import Path

def launch_nuboard_for_gnn_trans():
    """
    Launch nuboard to visualize GNN_Trans simulation results.
    """

    # Set paths
    nuplan_root = "/mnt/local/Xiao_Yi/nuplan-devkit"
    results_dir = "/mnt/local/Xiao_Yi/gnn_trans_simulation_results"

    # Check if results exist
    if not Path(results_dir).exists():
        print(f"Results directory not found: {results_dir}")
        print("Please run the simulation first with: python run_gnn_trans_simulation.py")
        return

    # Change to nuplan directory
    os.chdir(nuplan_root)

    # Launch nuboard command
    cmd = [
        "python", "run_nuboard.py",
        f"simulation_path={results_dir}",
        "port=5006"
    ]

    print("Launching nuboard for GNN_Trans visualization...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Results path: {results_dir}")
    print("\nOnce started, open http://localhost:5006 in your browser")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching nuboard: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down nuboard...")
        return 0

if __name__ == "__main__":
    exit_code = launch_nuboard_for_gnn_trans()
    sys.exit(exit_code or 0)
