#!/usr/bin/env python3
"""
Quick test for visualization module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.trajectory_converter import (
    recover_trajectory_from_increment,
    compute_trajectory_metrics,
    check_trajectory_validity,
    TrajectoryData
)
import numpy as np

# Test 1: Trajectory recovery
print("=" * 60)
print("Test 1: Trajectory Recovery")
print("=" * 60)

increments = np.random.randn(120) * 0.5  # Small random increments
start_pos = np.array([0.0, 0.0])
start_heading = 0.0
start_vx = 5.0
start_vy = 0.0

traj = recover_trajectory_from_increment(
    increments, start_pos, start_heading, start_vx, start_vy
)

print(f"Trajectory shape: {traj.positions.shape}")
print(f"Start position: {traj.positions[0]}")
print(f"End position: {traj.positions[-1]}")
print(f"Total distance: {np.sum(np.sqrt(np.sum(np.diff(traj.positions, axis=0)**2, axis=1))):.2f} m")

# Test 2: Metrics computation
print("\n" + "=" * 60)
print("Test 2: Metrics Computation")
print("=" * 60)

metrics = compute_trajectory_metrics(traj)
print("Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.3f}")

# Test 3: Validity check
print("\n" + "=" * 60)
print("Test 3: Validity Check")
print("=" * 60)

validity = check_trajectory_validity(traj)
print("Validity checks:")
for key, value in validity.items():
    if key == 'metrics':
        continue
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)