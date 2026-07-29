"""E2 · log-disjoint **三分** split（train/val/test），回应 R4-7（阶段11 §4.3 P0）。

R4-7 原文批评：loss 权重按 val 选、early stopping 按 val ADE、拓扑按 val ADE 选，最终 Table II
又在**同一个 val** 上报结果 —— 调参 / 选拓扑 / 选 checkpoint / 报结果四件事共用一份数据。
这是阶段11 §4.3 中「唯一一条即使其他全改好也能单独毙稿的效度问题」。

本脚本把 E1 的两分（`split_log_disjoint.py`，train/val）升级为三分：

    train  用于训练
    val    用于 early stopping / 超参 / 拓扑选择（沿用旧用途，允许被"看"）
    test   **只在最终报 headline 时用一次**，不参与任何选择

沿用 E1 的核心不变量：
  - **整 log 分配**（file 级 partition），防同一 nuPlan log 跨 split 泄漏；
  - manifest 缺失的 token 各成单例组，绝不与他人共 split；
  - absolute symlink 物化到独立 root，`GraphDataset.process()` 的 `os.walk` 可直接吃；
  - 固定 seed 可复现 + 三向硬断言。

**新旧可比性**：分配只依赖 `sorted(log_names)` + seed，故对 E1（无 route）与 E2（route 三通道）
用同一 seed 跑本脚本，会得到**完全相同的 log 划分** —— D8 的闭环对比因此是
「同数据、同 split、仅 route 差异」的干净对照（阶段⑫ §3）。

运行（conda nuplan env）：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/split_log_disjoint3.py \
      --data_dir /abs/.../processed_features_magnet_e2_route \
      [--val_ratio 0.1 --test_ratio 0.1 --seed 42 --force]
产出：<data_dir>_split3/{train,val,test} + split3_manifest.json
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
logger = logging.getLogger("split_log_disjoint3")

_PREFIX = "features_"
_SUFFIX = ".pkl"
_SPLITS = ("train", "val", "test")


def load_token_to_log(manifest_path: Path) -> Dict[str, str]:
    with open(manifest_path, encoding="utf-8") as f:
        rows = json.load(f)
    mapping = {r["token"]: r["log_name"] for r in rows}
    logger.info("manifest 载入 %d 条 token→log_name（唯一 log 数=%d）",
                len(mapping), len(set(mapping.values())))
    return mapping


def list_feature_pkls(data_dir: Path) -> List[Path]:
    """只列顶层 features_*.pkl，不递归（避免把已存在的 split 子目录算进来）。"""
    return sorted(p for p in data_dir.glob(f"{_PREFIX}*{_SUFFIX}") if p.is_file())


def _token_of(pkl: Path) -> str:
    return pkl.name[len(_PREFIX):-len(_SUFFIX)]


def group_by_log(pkls: List[Path], token_to_log: Dict[str, str]) -> Tuple[Dict[str, List[Path]], int]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    orphans = 0
    for p in pkls:
        tok = _token_of(p)
        log = token_to_log.get(tok)
        if log is None:
            orphans += 1
            log = f"__orphan__{tok}"
        groups[log].append(p)
    if orphans:
        logger.warning("有 %d 个 pkl 的 token 不在 manifest 中，已各自作为单例 log 组处理", orphans)
    return groups, orphans


def assign_logs(logs: List[str], val_ratio: float, test_ratio: float,
                seed: int) -> Dict[str, Set[str]]:
    """整 log 三分。固定 seed；与 E1 两分脚本同样先 `sorted` 再 shuffle，保证可复现。"""
    if len(logs) < 3:
        raise ValueError(f"log 数={len(logs)} < 3，无法做三分 log-disjoint split")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio+test_ratio={val_ratio + test_ratio} 必须 < 1.0")

    ordered = sorted(logs)
    random.Random(seed).shuffle(ordered)

    n = len(ordered)
    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    if n_val + n_test > n - 1:  # 至少留 1 个 log 给 train
        raise ValueError(f"n_val({n_val})+n_test({n_test}) 超出可用 log 数 {n}")

    return {
        "val": set(ordered[:n_val]),
        "test": set(ordered[n_val:n_val + n_test]),
        "train": set(ordered[n_val + n_test:]),
    }


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


def materialize(groups: Dict[str, List[Path]], assign: Dict[str, Set[str]],
                out_dir: Path, force: bool) -> Dict[str, int]:
    roots = {s: out_dir / s for s in _SPLITS}
    for s, root in roots.items():
        if root.exists():
            if not force:
                raise FileExistsError(f"{root} 已存在；用 --force 覆盖重建")
            shutil.rmtree(root)
    return {s: _symlink_group(groups, assign[s], roots[s]) for s in _SPLITS}


def run(data_dir: Path, manifest_path: Path, out_dir: Path,
        val_ratio: float, test_ratio: float, seed: int, force: bool) -> dict:
    token_to_log = load_token_to_log(manifest_path)
    pkls = list_feature_pkls(data_dir)
    logger.info("data_dir 顶层 features_*.pkl 数=%d", len(pkls))
    if not pkls:
        raise FileNotFoundError(f"{data_dir} 下没有 features_*.pkl")

    groups, orphans = group_by_log(pkls, token_to_log)
    assign = assign_logs(list(groups.keys()), val_ratio, test_ratio, seed)

    # 三向 log-disjoint 硬断言
    for a in _SPLITS:
        for b in _SPLITS:
            if a < b:
                ov = assign[a] & assign[b]
                assert not ov, f"log-disjoint 违背：{a}∩{b} 有 {len(ov)} 个 log，如 {list(ov)[:3]}"
    assert sum(len(assign[s]) for s in _SPLITS) == len(groups), "log 分配未覆盖全部组"

    counts = materialize(groups, assign, out_dir, force)

    summary = {
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_logs_total": len(groups),
        "n_orphan_tokens": orphans,
        **{f"n_logs_{s}": len(assign[s]) for s in _SPLITS},
        **{f"n_samples_{s}": counts[s] for s in _SPLITS},
        **{f"{s}_root": str(out_dir / s) for s in _SPLITS},
        "logs": {s: sorted(assign[s]) for s in _SPLITS},
        "note": "test 仅用于最终 headline 报数，不参与任何超参/拓扑/checkpoint 选择（R4-7）",
    }
    with open(out_dir / "split3_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("三分完成：logs %d/%d/%d，samples %d/%d/%d（train/val/test）",
                *[len(assign[s]) for s in _SPLITS], *[counts[s] for s in _SPLITS])
    logger.info("split3_manifest.json 已写出：%s", out_dir / "split3_manifest.json")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="log-disjoint 三分 split（train/val/test），回应 R4-7")
    p.add_argument("--data_dir", type=str, required=True, help="含 features_*.pkl 的目录（绝对路径）")
    p.add_argument("--manifest", type=str, default=None, help="log_manifest.json，默认 data_dir 下同名文件")
    p.add_argument("--out_dir", type=str, default=None, help="默认 <data_dir>_split3")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="覆盖重建已存在的 split 子目录")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    data_dir = Path(a.data_dir).resolve()
    manifest = Path(a.manifest).resolve() if a.manifest else data_dir / "log_manifest.json"
    out_dir = Path(a.out_dir).resolve() if a.out_dir else data_dir.parent / f"{data_dir.name}_split3"
    out_dir.mkdir(parents=True, exist_ok=True)
    run(data_dir, manifest, out_dir, a.val_ratio, a.test_ratio, a.seed, a.force)


if __name__ == "__main__":
    main()
