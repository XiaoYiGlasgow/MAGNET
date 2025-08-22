#!/usr/bin/env python3
"""
Run GNN_Trans simulation and launch nuBoard for visualization
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run simulation and launch nuboard"""

    # Set up paths
    nuplan_root = "/mnt/local/Xiao_Yi/nuplan-devkit"
    gnn_trans_root = "/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans"

    # Change to nuplan directory
    os.chdir(nuplan_root)

    # Add paths to Python path
    sys.path.insert(0, nuplan_root)
    sys.path.insert(0, gnn_trans_root)

    print("=" * 60)
    print("Running GNN_Trans Simulation for nuBoard Visualization")
    print("=" * 60)

    # Step 1: Run simulation with GNN_Trans planner
    print("\nStep 1: Running simulation...")

    simulation_cmd = [
        "python", "nuplan/planning/script/run_simulation.py",
        "+simulation=closed_loop_reactive_agents",
        "planner=ml_planner",
        "planner.ml_planner.model_config=transformer_model",
        "scenario_builder=nuplan_mini",
        "scenario_filter=one_of_each_scenario_type",
        "scenario_filter.limit_total_scenarios=5",
        f"hydra.searchpath=[file://{gnn_trans_root}/config]",
        "experiment_name=gnn_trans_simulation",
        "output_dir=/tmp/nuplan_simulation"
    ]

    try:
        result = subprocess.run(simulation_cmd, check=True, capture_output=True, text=True)
        print("Simulation completed successfully!")
        print("Simulation output:", result.stdout[-500:])  # Show last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        print("Error output:", e.stderr)
        return False

    # Step 2: Launch nuBoard
    print("\nStep 2: Launching nuBoard...")

    nuboard_cmd = [
        "python", "nuplan/planning/script/run_nuboard.py",
        "simulation_path=/tmp/nuplan_simulation"
    ]

    try:
        print("Starting nuBoard server...")
        print("You can access nuBoard at: http://localhost:5006")
        print("Press Ctrl+C to stop the server")
        subprocess.run(nuboard_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"nuBoard launch failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\nnuBoard server stopped by user")
        return True

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("GNN_Trans nuBoard visualization completed!")
    else:
        print("Failed to run GNN_Trans nuBoard visualization")
        sys.exit(1)
