"""E1 · log-disjoint train/val split（file 级 partition，防同一 nuPlan log 跨 split 泄漏）。

背景：`features_<token>.pkl` 文件名只含 scenario token（=initial_lidar_token），不含 log 身份。
预处理旁路 `log_manifest.json`（token→log_name，见 process_nuplan_to_12d.py 的 E1 改动）提供分组依据。
本脚本按 `log_name` 分组，整 log 分到 train 或 val，用 symlink 在独立 out_dir 下建 train/ val/ 两个
GraphDataset root（`GraphDataset.process()` 用 `os.walk` 递归找 pkl，symlink 子目录可直接吃）。

设计取舍：走 file 级两 root，而非在 InMemoryDataset 上按 index 切——index↔文件顺序脆弱，且避免改动
E0 冻结的 dataset.py。分组逻辑参考 train_diffusion_v9._split_train_val 的模式（仅借鉴，不 import danger 代码）。

运行（conda nuplan env）：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/split_log_disjoint.py \
      --data_dir /abs/path/processed_features_magnet_e1_full [--val_ratio 0.1 --seed 42 --force]
产出：<out_dir>/train、<out_dir>/val（symlink 到真 pkl）、<out_dir>/split_manifest.json。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("split_log_disjoint")

_PREFIX = "features_"
_SUFFIX = ".pkl"


def load_token_to_log(manifest_path: Path) -> Dict[str, str]:
    """读取预处理旁路 manifest，返回 token→log_name 映射。"""
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)
    mapping = {r["token"]: r["log_name"] for r in rows}
    logger.info("manifest 载入 %d 条 token→log_name（唯一 log 数=%d）",
                len(mapping), len(set(mapping.values())))
    return mapping


def list_feature_pkls(data_dir: Path) -> List[Path]:
    """只列 data_dir 顶层 features_*.pkl（不递归，避免把已存在的 split 子目录符号链接算进来）。"""
    return sorted(p for p in data_dir.glob(f"{_PREFIX}*{_SUFFIX}") if p.is_file())


def _token_of(pkl: Path) -> str:
    return pkl.name[len(_PREFIX):-len(_SUFFIX)]


def group_by_log(pkls: List[Path], token_to_log: Dict[str, str]) -> Tuple[Dict[str, List[Path]], int]:
    """按 log_name 把 pkl 分组；manifest 缺失的 token 各成单例组（绝不跨 split 泄漏）。"""
    groups: Dict[str, List[Path]] = defaultdict(list)
    orphans = 0
    for p in pkls:
        tok = _token_of(p)
        log = token_to_log.get(tok)
        if log is None:
            orphans += 1
            log = f"__orphan__{tok}"  # 单例组=只含自己，不与他人共 split
        groups[log].append(p)
    if orphans:
        logger.warning("有 %d 个 pkl 的 token 不在 manifest 中，已各自作为单例 log 组处理", orphans)
    return groups, orphans


def assign_logs(logs: List[str], val_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    """整 log 分配到 train/val；固定 seed，保证可复现。"""
    if len(logs) < 2:
        raise ValueError(f"log 数={len(logs)} < 2，无法做 log-disjoint split（数据太少）")
    ordered = sorted(logs)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = max(1, round(len(ordered) * val_ratio))
    n_val = min(n_val, len(ordered) - 1)  # 至少留 1 个 log 给 train
    val_logs = set(ordered[:n_val])
    train_logs = set(ordered[n_val:])
    return train_logs, val_logs


def _symlink_group(groups: Dict[str, List[Path]], logs: Set[str], dst_root: Path) -> int:
    dst_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for log in logs:
        for pkl in groups[log]:
            link = dst_root / pkl.name
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(pkl.resolve(), link)
            count += 1
    return count


def materialize_split(
    groups: Dict[str, List[Path]],
    train_logs: Set[str],
    val_logs: Set[str],
    out_dir: Path,
    force: bool,
) -> Tuple[Path, Path, int, int]:
    """在 out_dir 下建 train/ val/ symlink 子目录。"""
    train_root, val_root = out_dir / "train", out_dir / "val"
    for root in (train_root, val_root):
        if root.exists():
            if not force:
                raise FileExistsError(f"{root} 已存在；用 --force 覆盖重建")
            shutil.rmtree(root)
    n_train = _symlink_group(groups, train_logs, train_root)
    n_val = _symlink_group(groups, val_logs, val_root)
    return train_root, val_root, n_train, n_val


def run_split(data_dir: Path, manifest_path: Path, out_dir: Path,
              val_ratio: float, seed: int, force: bool) -> dict:
    """完整流程：分组→分配→物化→断言无交集→写 split_manifest.json。"""
    token_to_log = load_token_to_log(manifest_path)
    pkls = list_feature_pkls(data_dir)
    logger.info("data_dir 顶层 features_*.pkl 数=%d", len(pkls))
    if not pkls:
        raise FileNotFoundError(f"{data_dir} 下没有 features_*.pkl")

    groups, orphans = group_by_log(pkls, token_to_log)
    train_logs, val_logs = assign_logs(list(groups.keys()), val_ratio, seed)

    # log-disjoint 硬断言
    overlap = train_logs & val_logs
    assert not overlap, f"log-disjoint 违背：{len(overlap)} 个 log 同时在 train/val：{list(overlap)[:3]}"

    train_root, val_root, n_train, n_val = materialize_split(
        groups, train_logs, val_logs, out_dir, force)

    summary = {
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "n_logs_total": len(groups),
        "n_logs_train": len(train_logs),
        "n_logs_val": len(val_logs),
        "n_samples_train": n_train,
        "n_samples_val": n_val,
        "n_orphan_tokens": orphans,
        "train_root": str(train_root),
        "val_root": str(val_root),
        "val_logs": sorted(val_logs),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("log-disjoint split 完成：logs train/val=%d/%d，samples train/val=%d/%d",
                len(train_logs), len(val_logs), n_train, n_val)
    logger.info("train_root=%s", train_root)
    logger.info("val_root  =%s", val_root)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 log-disjoint train/val split（file 级 symlink partition）")
    p.add_argument("--data_dir", required=True, type=str,
                   help="含 features_*.pkl 与 log_manifest.json 的预处理输出目录（绝对路径）")
    p.add_argument("--manifest", type=str, default=None,
                   help="token→log_name manifest 路径（默认 <data_dir>/log_manifest.json）")
    p.add_argument("--out_dir", type=str, default=None,
                   help="split 输出目录（默认 <data_dir>_logsplit）")
    p.add_argument("--val_ratio", type=float, default=0.1, help="按 log 数计的 val 比例")
    p.add_argument("--seed", type=int, default=42, help="log 洗牌随机种子（可复现）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的 train/ val/ 子目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else data_dir / "log_manifest.json"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir.parent / f"{data_dir.name}_logsplit"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}（预处理是否跑过 E1 版？）")
    run_split(data_dir, manifest_path, out_dir, args.val_ratio, args.seed, args.force)


if __name__ == "__main__":
    main()
