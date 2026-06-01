import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd


def _compute_features(path: str) -> dict:
    df = pd.read_pickle(path)
    increments = df["GROUND_TRUTH"].values[0].squeeze(0).astype(np.float32)
    step_norms = np.linalg.norm(increments, axis=1)
    total_distance = float(np.sum(step_norms))

    nonzero = step_norms > 1e-6
    if np.sum(nonzero) >= 2:
        directions = np.arctan2(increments[nonzero, 1], increments[nonzero, 0])
        angle_diffs = np.diff(directions)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        direction_change = float(np.sum(np.abs(angle_diffs)))
    else:
        direction_change = 0.0

    return {
        "file": os.path.basename(path),
        "total_distance": total_distance,
        "direction_change": direction_change,
    }


def _assign_bin(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    bins = np.zeros(len(values), dtype=int)
    bins[values > q1] = 1
    bins[values > q2] = 2
    return bins


def generate_manifest(data_dir: str, sample_count: int, seed: int) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    if len(paths) < sample_count:
        raise ValueError(f"Requested {sample_count} samples but only found {len(paths)} files")

    records = [_compute_features(path) for path in paths]
    distances = np.array([record["total_distance"] for record in records], dtype=np.float32)
    direction_changes = np.array([record["direction_change"] for record in records], dtype=np.float32)
    distance_bins = _assign_bin(distances)
    direction_bins = _assign_bin(direction_changes)

    buckets = {}
    for idx, record in enumerate(records):
        key = (int(distance_bins[idx]), int(direction_bins[idx]))
        buckets.setdefault(key, []).append(record["file"])

    rng = random.Random(seed)
    total_files = len(records)
    sorted_keys = sorted(buckets.keys())
    target_by_bucket = {}
    fractional_parts = []

    allocated = 0
    for key in sorted_keys:
        bucket_size = len(buckets[key])
        raw_target = sample_count * bucket_size / total_files
        base_target = min(bucket_size, int(np.floor(raw_target)))
        target_by_bucket[key] = base_target
        allocated += base_target
        fractional_parts.append((raw_target - base_target, key))

    remaining = sample_count - allocated
    for _, key in sorted(fractional_parts, reverse=True):
        if remaining <= 0:
            break
        if target_by_bucket[key] < len(buckets[key]):
            target_by_bucket[key] += 1
            remaining -= 1

    selected_files = []
    for key in sorted_keys:
        files = list(buckets[key])
        rng.shuffle(files)
        selected_files.extend(files[: target_by_bucket[key]])

    if len(selected_files) > sample_count:
        rng.shuffle(selected_files)
        selected_files = selected_files[:sample_count]
    elif len(selected_files) < sample_count:
        chosen = set(selected_files)
        leftovers = [record["file"] for record in records if record["file"] not in chosen]
        rng.shuffle(leftovers)
        selected_files.extend(leftovers[: sample_count - len(selected_files)])

    selected_files = sorted(selected_files)

    selected_set = set(selected_files)
    bucket_summary = {}
    for key in sorted_keys:
        bucket_name = f"distance_{key[0]}_turn_{key[1]}"
        bucket_summary[bucket_name] = {
            "available": len(buckets[key]),
            "selected": sum(1 for file_name in buckets[key] if file_name in selected_set),
        }

    return {
        "data_dir": data_dir,
        "sample_count": sample_count,
        "seed": seed,
        "stratification": ["total_distance", "direction_change"],
        "sample_files": selected_files,
        "bucket_summary": bucket_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a stratified manifest for medium_magnitude_speed samples.")
    parser.add_argument("--data_dir", required=True, help="Directory containing processed .pkl files")
    parser.add_argument("--sample_count", type=int, default=3000, help="Number of samples to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_json", required=True, help="Path to save manifest json")
    args = parser.parse_args()

    manifest = generate_manifest(args.data_dir, args.sample_count, args.seed)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to: {args.output_json}")
    print(json.dumps({k: manifest[k] for k in ['sample_count', 'seed', 'bucket_summary']}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
