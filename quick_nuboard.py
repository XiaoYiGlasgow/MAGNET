#!/usr/bin/env python3
"""
Quick launch script for GNN_Trans nuBoard visualization
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Quick launch nuboard with existing results"""

    # Check for existing nuboard results
    possible_results = [
        "/mnt/local/Xiao_Yi/nuplan_results",
        "/mnt/local/Xiao_Yi/nuplan-devkit/outputs",
        "/tmp/nuplan_simulation"
    ]

    results_path = None
    for path in possible_results:
        if Path(path).exists():
            # Check if it contains nuboard files
            nuboard_files = list(Path(path).glob("**/*.nuboard"))
            if nuboard_files:
                results_path = path
                print(f"Found nuboard results at: {results_path}")
                break

    if not results_path:
        print("No existing simulation results found.")
        print("Available paths checked:")
        for path in possible_results:
            print(f"  - {path}: {'exists' if Path(path).exists() else 'not found'}")

        # Try to use the nuplan_results directory
        results_path = "/mnt/local/Xiao_Yi/nuplan_results"
        if Path(results_path).exists():
            print(f"Using: {results_path}")
        else:
            print("Please run a simulation first or check your results directory.")
            return False

    # Change to nuplan directory
    nuplan_root = "/mnt/local/Xiao_Yi/nuplan-devkit"
    os.chdir(nuplan_root)

    print("=" * 60)
    print("Launching nuBoard for GNN_Trans Visualization")
    print("=" * 60)
    print(f"Results path: {results_path}")
    print("Server will start at: http://localhost:5006")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    # Launch nuBoard
    nuboard_cmd = [
        "python", "nuplan/planning/script/run_nuboard.py",
        f"simulation_path={results_path}"
    ]

    try:
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
    if not success:
        sys.exit(1)
