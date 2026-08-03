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


def assign_logs_from_splitter(logs: List[str], splitter_yaml: Path) -> Tuple[Set[str], Set[str]]:
    """🔴 Q64 口径：直接读 PlanTF splitter yaml 的 `log_splits`，**不自己抽样**。

    第二轮 scaling ladder 的核心前提是「同 log 池、同 token、同 benchmark」。若 MAGNET 侧
    再用 `rng.shuffle` 自划一次 val，同一级上两个模型看到的训练数据就不同，配对比较不成立。
    ⇒ 两条管线必须消费同一份划分（与 S1 共享清单同源：谁都不许自己再抽一次）。

    yaml 由 GNN_Trans 自动生成，规则 = log 名排序后下标 %10==0 进 val（确定性、无种子）。

    Args:
        logs: 本地实际存在 pkl 的 log 名（预处理产物决定）。
        splitter_yaml: `planTF/config/splitter/gnn_scaling_L{n}.yaml` 的绝对路径。

    Returns:
        (train_logs, val_logs)，均已与本地 logs 取交集。

    Raises:
        ValueError: yaml 结构不对、train/val 有交集，或本地 log 不是 yaml 全集的子集。
    """
    import yaml  # 局部 import：只有走本分支才需要 PyYAML

    with open(splitter_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        splits = cfg["log_splits"]
        y_train, y_val = set(splits["train"]), set(splits["val"])
    except (KeyError, TypeError) as e:
        raise ValueError(f"{splitter_yaml} 缺 log_splits.train / log_splits.val") from e

    # 硬断言 1：yaml 自身 train/val 不得有交集
    inter = y_train & y_val
    if inter:
        raise ValueError(f"{splitter_yaml} 的 train/val 有 {len(inter)} 个交集：{sorted(inter)[:3]}")

    y_all = y_train | y_val
    local = set(logs)

    # 本地**多出**的 log 是**预期内的**：S4 预处理一次产出 L3 全量（2 292 log），
    # 同一个 data_dir 同时服务 L1/L2/L3 三级，低级别的 splitter 自然只覆盖其中一部分。
    # ⇒ 这里只报数，不失败；真正该防的是下面的 missing。
    extra = local - y_all
    if extra:
        logger.info("本地有 %d 个 log 不在本级 splitter 里（%d 级只取 %d 个）—— "
                    "这是嵌套阶梯的预期行为，已按 splitter 取子集",
                    len(extra), len(y_all), len(y_all))

    # 🔴 硬断言 2：**splitter 要的每一个 log 本地都必须有 pkl**。
    #    缺 log 意味着这一级实际训练的数据比设计的少 ⇒ 两条管线不再是同一批 log
    #    ⇒ 配对比较不成立（Q64 要防的正是这件事）。断点续跑请等预处理跑完再 split。
    missing = y_all - local
    if missing:
        raise ValueError(
            f"🔴 splitter 要的 {len(y_all)} 个 log 里有 {len(missing)} 个本地没有 pkl"
            f"（例：{sorted(missing)[:3]}）⇒ 该级数据不完整，两条管线的 log 池会不一致。"
            f" 请确认预处理已跑完（features_*.pkl 数达标）后再 split。"
        )

    train_logs, val_logs = y_train & local, y_val & local
    if not val_logs:
        raise ValueError("按 splitter 划分后 val 为空 —— 本地 pkl 覆盖不到任何 val log")

    logger.info("Q64 口径：读 %s ⇒ train/val = %d/%d log（yaml 全集 %d，本地实有 %d）",
                splitter_yaml.name, len(train_logs), len(val_logs), len(y_all), len(local))
    return train_logs, val_logs


def assign_logs(logs: List[str], val_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    """整 log 分配到 train/val；固定 seed，保证可复现。

    ⚠️ 第二轮 scaling ladder **不得用本函数** —— 它自己抽一次样，会与 PlanTF 侧的
    splitter 划分不同。第二轮请走 `--splitter_yaml`（见 `assign_logs_from_splitter`，Q64）。
    """
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
              val_ratio: float, seed: int, force: bool,
              splitter_yaml: Path | None = None) -> dict:
    """完整流程：分组→分配→物化→断言无交集→写 split_manifest.json。

    `splitter_yaml` 非 None 时走 Q64 口径（读 PlanTF 同一份划分），此时 `val_ratio` / `seed`
    对划分**不起作用**，仅原样记进 summary 以便区分产物来源。
    """
    token_to_log = load_token_to_log(manifest_path)
    pkls = list_feature_pkls(data_dir)
    logger.info("data_dir 顶层 features_*.pkl 数=%d", len(pkls))
    if not pkls:
        raise FileNotFoundError(f"{data_dir} 下没有 features_*.pkl")

    groups, orphans = group_by_log(pkls, token_to_log)
    if splitter_yaml is not None:
        train_logs, val_logs = assign_logs_from_splitter(list(groups.keys()), splitter_yaml)
    else:
        train_logs, val_logs = assign_logs(list(groups.keys()), val_ratio, seed)

    # log-disjoint 硬断言
    overlap = train_logs & val_logs
    assert not overlap, f"log-disjoint 违背：{len(overlap)} 个 log 同时在 train/val：{list(overlap)[:3]}"

    train_root, val_root, n_train, n_val = materialize_split(
        groups, train_logs, val_logs, out_dir, force)

    summary = {
        "data_dir": str(data_dir),
        "manifest": str(manifest_path),
        # 🔴 Q64：划分来源必须能一眼看出，否则两条管线是否同源无法事后核验
        "split_source": "splitter_yaml" if splitter_yaml is not None else "rng_shuffle",
        "splitter_yaml": str(splitter_yaml) if splitter_yaml is not None else None,
        "seed": seed if splitter_yaml is None else None,
        "val_ratio": val_ratio if splitter_yaml is None else None,
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
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="按 log 数计的 val 比例（--splitter_yaml 模式下忽略）")
    p.add_argument("--seed", type=int, default=42,
                   help="log 洗牌随机种子（--splitter_yaml 模式下忽略）")
    p.add_argument("--splitter_yaml", type=str, default=None,
                   help="🔴 Q64 第二轮口径：读 PlanTF 的 planTF/config/splitter/gnn_scaling_L{n}.yaml，"
                        "两条管线共用同一份 train/val 划分（不再自己抽样）")
    p.add_argument("--force", action="store_true", help="覆盖已存在的 train/ val/ 子目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else data_dir / "log_manifest.json"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_dir.parent / f"{data_dir.name}_logsplit"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在：{manifest_path}（预处理是否跑过 E1 版？）")
    splitter_yaml = Path(args.splitter_yaml).resolve() if args.splitter_yaml else None
    if splitter_yaml is not None and not splitter_yaml.exists():
        raise FileNotFoundError(f"splitter yaml 不存在：{splitter_yaml}")
    run_split(data_dir, manifest_path, out_dir, args.val_ratio, args.seed, args.force,
              splitter_yaml)


if __name__ == "__main__":
    main()
