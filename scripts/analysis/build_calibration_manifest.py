"""S4-b 第一步：构建**固定、可哈希、train-only** 的 BN calibration manifest。

阶段⑳ §2.2 冻结的规则（本脚本逐条实现）：

| 项 | 规则 | 实现 |
|---|---|---|
| 数据来源 | 只来自 F1 **training split**；禁用 validation / locked test | `--train_root` 只接受 split3 的 `train/`，并与 `split3_manifest.json` 的 `train_root` 逐字符核对 |
| manifest | 固定、可哈希、按 log/场景类型覆盖；初始目标 N=1600 | 480 个 train log 全覆盖（最大余数法分配配额），log 内按 **history-only 速度分位** 均匀取样 |
| 标签 | calibration 不使用 future label 做选择 | 只读 `EGO_HISTORY`；`GROUND_TRUTH` 一次都不读 |
| 顺序/批量/dtype | 固定 | manifest 内 token 顺序即喂入顺序；断言 N % batch_size == 0 |

⚠️ **没有 scenario_type 可用**：F1 的 `log_manifest.json` 只有 `{token, log_name}` 两个字段
（实测 30000 条全部如此），nuPlan 的 scenario_type 标签不在特征侧。因此"场景类型覆盖"用
**history-only 的运动学代理**实现：log 内按平均速度排序后取等距分位点，保证低速/中速/高速都进 manifest。
这是一个明确记录在案的口径妥协，不是遗漏。

用法::

    python3 scripts/analysis/build_calibration_manifest.py \
        --split_manifest data/processed_features_magnet_e1_none_v3_split3/split3_manifest.json \
        --n_target 1600 --batch_size 32 \
        --out experiments/calibration/calib_manifest_f1_train_n1600.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

#: 特征 pkl 的采样间隔（database_interval=0.1 ⇒ 10 Hz，见执行契约）
DT_SECONDS = 0.1
#: 从文件名解析 token：features_<token>.pkl
FILE_PREFIX = "features_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建 train-only BN calibration manifest（S4-b）")
    p.add_argument("--split_manifest", required=True, type=Path,
                   help="split3_manifest.json（提供 train_root 与 log 归属，防止误用 val/test）")
    p.add_argument("--n_target", type=int, default=1600, help="目标样本数（阶段⑳ 初始目标 1600）")
    p.add_argument("--batch_size", type=int, default=32, help="calibration batch size；N 必须被它整除")
    p.add_argument("--out", required=True, type=Path, help="输出 manifest JSON")
    return p.parse_args()


def _git_provenance(repo: Path) -> Dict[str, str]:
    """记录代码版本（D0#2/#5：任何 formal 产物都要能回溯到 commit）。"""
    def _run(*args: str) -> str:
        try:
            return subprocess.run(args, cwd=repo, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "UNKNOWN"
    return {"commit": _run("git", "rev-parse", "HEAD"),
            "dirty": "yes" if _run("git", "status", "--porcelain") else "no"}


def load_token_to_log(split_manifest: Path) -> Tuple[Path, Dict[str, str], List[str]]:
    """读 split manifest，返回 (train_root, token->log_name, train_logs)。"""
    sm = json.loads(split_manifest.read_text())
    train_root = Path(sm["train_root"])
    parent_manifest = Path(sm["manifest"])
    entries = json.loads(parent_manifest.read_text())
    token_to_log = {e["token"]: e["log_name"] for e in entries}
    train_logs = list(sm["logs"]["train"])
    logger.info("split manifest: train_root=%s, train logs=%d, token 表=%d",
                train_root, len(train_logs), len(token_to_log))
    return train_root, token_to_log, train_logs


def scan_train_tokens(train_root: Path, token_to_log: Dict[str, str],
                      train_logs: List[str]) -> Dict[str, List[str]]:
    """扫 train split 的 pkl，按 log 归组；任何 token 缺归属都直接报错（不静默丢样本）。"""
    files = sorted(train_root.glob(f"{FILE_PREFIX}*.pkl"))
    if not files:
        raise RuntimeError(f"{train_root} 下没有 {FILE_PREFIX}*.pkl")
    train_log_set = set(train_logs)
    by_log: Dict[str, List[str]] = defaultdict(list)
    missing: List[str] = []
    leaked: List[str] = []
    for f in files:
        token = f.stem[len(FILE_PREFIX):]
        log = token_to_log.get(token)
        if log is None:
            missing.append(token)
            continue
        if log not in train_log_set:
            leaked.append(token)      # log-disjoint 被破坏 = 数据链路出问题，必须暴露
            continue
        by_log[log].append(token)
    if missing:
        raise RuntimeError(f"{len(missing)} 个 token 在 log_manifest 里查不到归属，例：{missing[:3]}")
    if leaked:
        raise RuntimeError(f"{len(leaked)} 个 token 的 log 不在 train 列表里（split 泄漏），例：{leaked[:3]}")
    logger.info("train split: %d 样本 / %d log（manifest 声明 %d log）",
                len(files), len(by_log), len(train_logs))
    return {log: sorted(tokens) for log, tokens in by_log.items()}


def mean_speed_from_history(pkl_path: Path) -> float:
    """由 `EGO_HISTORY`（[20,2] 逐帧增量，10 Hz）算平均速率 m/s。

    **只读 history**，不碰 `GROUND_TRUTH` —— calibration 选样不得使用 future label。
    第 0 行恒为 0（原点），故从第 1 行起算。
    """
    df = pd.read_pickle(pkl_path)
    hist = df["EGO_HISTORY"].values[0]
    if hist is None or not isinstance(hist, np.ndarray) or hist.shape[0] < 2:
        return 0.0
    step = np.linalg.norm(np.asarray(hist[1:], dtype=np.float64), axis=1)
    return float(step.mean() / DT_SECONDS)


def allocate_quota(by_log: Dict[str, List[str]], n_target: int) -> Dict[str, int]:
    """按各 log 样本数比例分配配额（最大余数法），每个 log 至少 1，且不超过其样本数。"""
    logs = sorted(by_log)
    counts = np.array([len(by_log[g]) for g in logs], dtype=np.float64)
    if n_target > counts.sum():
        raise ValueError(f"n_target={n_target} 超过 train 样本总数 {int(counts.sum())}")
    if n_target < len(logs):
        raise ValueError(f"n_target={n_target} 小于 log 数 {len(logs)}，无法保证 log 全覆盖")

    exact = counts / counts.sum() * n_target
    quota = np.maximum(1, np.floor(exact)).astype(int)
    quota = np.minimum(quota, counts.astype(int))

    # 最大余数法补齐/削减到恰好 n_target；tie 由 log 名字典序决定，保证可复现
    while quota.sum() != n_target:
        if quota.sum() < n_target:
            room = quota < counts
            order = np.argsort(-(exact - quota) + np.where(room, 0.0, np.inf))
            quota[order[0]] += 1
        else:
            room = quota > 1
            order = np.argsort((exact - quota) + np.where(room, 0.0, np.inf))
            quota[order[0]] -= 1
    return {g: int(q) for g, q in zip(logs, quota)}


def pick_by_speed_quantile(tokens_sorted_by_speed: List[str], k: int) -> List[str]:
    """在按速度排序的 token 列表上取 k 个**中点分位** ((i+0.5)/k) 样本。

    用中点而非端点包含的 `linspace(0, n-1, k)`：后者每个 log 必取最慢与最快两个极端样本，
    实测会把选中速度均值从 4.775 抬到 5.148 m/s（+8%）。BN 的 running statistics 估计的是
    **部署输入分布**，选样偏移会直接偏移统计量，故这里要的是分层无偏取样，覆盖是副产品。
    """
    n = len(tokens_sorted_by_speed)
    if k >= n:
        return list(tokens_sorted_by_speed)
    idx = np.floor((np.arange(k) + 0.5) / k * n).astype(int)
    idx = np.unique(np.clip(idx, 0, n - 1))
    for cand in range(n):                    # 极端重复时向后补空位（确定性）
        if len(idx) >= k:
            break
        if cand not in set(idx.tolist()):
            idx = np.unique(np.concatenate([idx, np.array([cand], dtype=int)]))
    return [tokens_sorted_by_speed[i] for i in idx[:k]]


def main() -> None:
    args = parse_args()
    if args.n_target % args.batch_size != 0:
        raise ValueError(f"n_target={args.n_target} 必须能被 batch_size={args.batch_size} 整除"
                         "（momentum=None 的累积平均要求等大小 batch）")

    repo = Path(__file__).resolve().parents[2]
    train_root, token_to_log, train_logs = load_token_to_log(args.split_manifest)
    by_log = scan_train_tokens(train_root, token_to_log, train_logs)

    logger.info("读 EGO_HISTORY 计算速度代理（history-only，不读 GROUND_TRUTH）...")
    speed: Dict[str, float] = {}
    for log in sorted(by_log):
        for token in by_log[log]:
            speed[token] = mean_speed_from_history(train_root / f"{FILE_PREFIX}{token}.pkl")

    quota = allocate_quota(by_log, args.n_target)
    selected: List[Dict[str, object]] = []
    for log in sorted(by_log):
        ordered = sorted(by_log[log], key=lambda t: (speed[t], t))
        for token in sorted(pick_by_speed_quantile(ordered, quota[log])):
            selected.append({"token": token, "log_name": log,
                             "ego_speed_mps": round(speed[token], 6)})

    if len(selected) != args.n_target:
        raise RuntimeError(f"选出 {len(selected)} 条，与目标 {args.n_target} 不符")

    # manifest 顺序 = (log_name, token) 字典序 = calibration 喂入顺序
    selected.sort(key=lambda r: (r["log_name"], r["token"]))
    tokens = [str(r["token"]) for r in selected]
    manifest_hash = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()

    sel_speeds = np.array([r["ego_speed_mps"] for r in selected], dtype=np.float64)
    all_speeds = np.array(list(speed.values()), dtype=np.float64)
    doc = {
        "contract": "S4-b BN calibration manifest（阶段⑳ §2.2 冻结）",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "split_manifest": str(args.split_manifest.resolve()),
        "train_root": str(train_root),
        "source_split": "train",
        "uses_future_label": False,
        "selection_rule": "log 全覆盖（最大余数法配额）+ log 内 history-only 速度中点分位 (i+0.5)/k",
        "scenario_type_available": False,
        "scenario_type_note": "log_manifest.json 只有 {token, log_name}；用 history-only 速度分位作场景覆盖代理",
        "n_selected": len(tokens),
        "batch_size": args.batch_size,
        "n_batches": len(tokens) // args.batch_size,
        "n_logs_covered": len(by_log),
        "n_train_samples_total": int(sum(len(v) for v in by_log.values())),
        "quota_min": int(min(quota.values())),
        "quota_max": int(max(quota.values())),
        "speed_selected": {"mean": float(sel_speeds.mean()), "sd": float(sel_speeds.std(ddof=1)),
                           "min": float(sel_speeds.min()), "max": float(sel_speeds.max()),
                           "p50": float(np.median(sel_speeds))},
        "speed_train_all": {"mean": float(all_speeds.mean()), "sd": float(all_speeds.std(ddof=1)),
                            "min": float(all_speeds.min()), "max": float(all_speeds.max()),
                            "p50": float(np.median(all_speeds))},
        "manifest_hash_sha256": manifest_hash,
        "code_version": _git_provenance(repo),
        "entries": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    logger.info("写出 %s", args.out)
    logger.info("N=%d  logs=%d  hash=%s", len(tokens), len(by_log), manifest_hash[:16])
    logger.info("选中速度 mean=%.3f sd=%.3f [%.3f, %.3f] m/s（全 train mean=%.3f sd=%.3f）",
                sel_speeds.mean(), sel_speeds.std(ddof=1), sel_speeds.min(), sel_speeds.max(),
                all_speeds.mean(), all_speeds.std(ddof=1))


if __name__ == "__main__":
    main()
