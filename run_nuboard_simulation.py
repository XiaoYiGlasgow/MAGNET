#!/usr/bin/env python3
"""Run GNN_Trans simulation and optionally launch NuBoard.

This is the maintained planning entrypoint. It standardizes on the local
`simple_simulation.yaml` + `gnn_trans_planner` chain instead of the older
mixed entrypoints.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


NUPLAN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit")
GNN_TRANS_ROOT = NUPLAN_ROOT / "GNN_Trans"
DEFAULT_OUTPUT_DIR = Path("/mnt/local/Xiao_Yi/gnn_trans_simulation_results")


def build_simulation_command(output_dir: Path, config_name: str, planner_name: str) -> list[str]:
    """Build the nuPlan simulation command using the local GNN_Trans config tree."""
    return [
        sys.executable,
        "nuplan/planning/script/run_simulation.py",
        "--config-path",
        str(GNN_TRANS_ROOT / "configs"),
        "--config-name",
        config_name,
        f"planner={planner_name}",
        f"output_dir={output_dir}",
        "verbose=true",
    ]


def build_nuboard_command(output_dir: Path, port: int) -> list[str]:
    """Build the NuBoard launch command."""
    return [
        sys.executable,
        "nuplan/planning/script/run_nuboard.py",
        f"simulation_path={output_dir}",
        f"port={port}",
    ]


def run_simulation(output_dir: Path, config_name: str, planner_name: str) -> bool:
    """Run the maintained GNN_Trans simulation chain."""
    os.chdir(NUPLAN_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_simulation_command(output_dir=output_dir, config_name=config_name, planner_name=planner_name)

    print("=" * 60)
    print("Running GNN_Trans planning simulation")
    print("=" * 60)
    print("Command:")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"Simulation completed. Results saved to: {output_dir}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Simulation failed: {exc}")
        return False


def launch_nuboard(output_dir: Path, port: int) -> bool:
    """Launch NuBoard for a completed simulation output directory."""
    os.chdir(NUPLAN_ROOT)
    if not output_dir.exists():
        print(f"Simulation results not found: {output_dir}")
        return False

    cmd = build_nuboard_command(output_dir=output_dir, port=port)
    print("=" * 60)
    print("Launching NuBoard")
    print("=" * 60)
    print("Command:")
    print(" ".join(cmd))
    print(f"Open http://localhost:{port} after the server starts")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"NuBoard launch failed: {exc}")
        return False
    except KeyboardInterrupt:
        print("\nNuBoard stopped by user")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GNN_Trans planning simulation and launch NuBoard.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for simulation outputs and NuBoard bundle inputs.",
    )
    parser.add_argument(
        "--config-name",
        default="simple_simulation",
        help="Config file name under GNN_Trans/configs to use for simulation.",
    )
    parser.add_argument(
        "--planner-name",
        default="gnn_trans_local_planner",
        help="Planner config group name to use.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5006,
        help="NuBoard server port.",
    )
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Launch NuBoard only, assuming simulation results already exist.",
    )
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="Alias of --skip-simulation for compatibility with older scripts.",
    )
    parser.add_argument(
        "--no-nuboard",
        action="store_true",
        help="Run simulation only without launching NuBoard.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    should_skip_sim = args.skip_simulation or args.launch_only
    if not should_skip_sim:
        success = run_simulation(output_dir=output_dir, config_name=args.config_name, planner_name=args.planner_name)
        if not success:
            return 1

    if args.no_nuboard:
        return 0

    return 0 if launch_nuboard(output_dir=output_dir, port=args.port) else 1


if __name__ == "__main__":
    raise SystemExit(main())
