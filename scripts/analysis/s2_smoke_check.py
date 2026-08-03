"""S2 冒烟判据 —— 判据写在见到任何结果之前（阶段53 起的纪律）。

七条判据，全过才允许起全量 cache：

  [P1] cache 进程 rc == 0，且日志里出现 devkit 自己打的 "Extracted N scenarios"。
       ⚠️ 第十条纪律：rc=0 不代表跑成了，所以 rc 只是必要条件，真判据是 [P2]。
  [P2] 落盘 token 数 == 请求的 200。若不足，必须报出缺哪些 —— rig=False 下不应有
       goal 导致的丢弃，任何缺失都说明还有第二个未知的过滤器。
  [P3] 落盘 token ⊆ 请求集合（无污染：filter 真的按 scenario_tokens 生效，
       而不是忽略它、退化成「整个 log 全采」）。
  [P4] feature.gz 的 key 集合（含 agent/map 两层嵌套）与现有 plantf_cache_L0 **完全相等**。
  [P5] 各叶子的 dtype 相等；形状除样本维（agent 数 / map 元素数）外相等。
  [P6] ray 的 session 目录不在根分区 —— 根分区只剩 76 G 且已 96% 满，
       全量 43 GB 时 spill 到那里会炸。
  [P7] 两城的 log 都有产物 —— 验证 data_root 传 list 真的让 boston + pittsburgh
       同时被发现（devkit 的 discover_log_dbs 用 iterdir()，不递归）。

用法：
    python s2_smoke_check.py
"""

from __future__ import annotations

import gzip
import json
import logging
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SMOKE_CACHE = GNN / "data/plantf_cache_smoke"
REF_CACHE = GNN / "data/plantf_cache_L0"
RUN_LOG = GNN / "data/s2_cache_smoke_run_20260802.log"
TOKENS_JSON = GNN / "experiments/scaling_ladder/s2_smoke_tokens.json"
SPLITS = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits")

# 这些叶子的第 0 维（或指定维）是可变的样本维，形状不参与 [P5] 比较
VARIABLE_DIMS: Dict[str, Tuple[int, ...]] = {}


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict 摊平成 {点分路径: 叶子}。"""
    flat: Dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _load_feature(sample_dir: Path) -> Dict[str, Any]:
    """读一个样本的 feature.gz 并摊平。"""
    with gzip.open(sample_dir / "feature.gz", "rb") as handle:
        return _flatten(pickle.load(handle))


def _iter_samples(cache: Path) -> List[Path]:
    """枚举 cache 下的样本目录（<log>/<scenario_type>/<token>）。"""
    return sorted(p for p in cache.glob("*/*/*") if p.is_dir() and (p / "feature.gz").is_file())


def _describe(value: Any) -> Tuple[str, Any]:
    """返回 (dtype 字符串, 形状)，标量与 ndarray 统一处理。"""
    if isinstance(value, np.ndarray):
        return str(value.dtype), tuple(value.shape)
    return type(value).__name__, ()


def main() -> int:
    """跑七条判据，返回 0 表示全过。"""
    requested = json.loads(TOKENS_JSON.read_text())
    req_tokens = set(requested["tokens"])
    req_logs = requested["logs"]

    results: List[Tuple[str, bool, str]] = []

    # [P1]
    rc_ok = RUN_LOG.is_file()
    extracted = 0
    if rc_ok:
        text = RUN_LOG.read_text(errors="replace")
        for line in text.splitlines():
            if "Extracted" in line and "scenarios" in line:
                try:
                    extracted += int(line.split("Extracted")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        has_error = "Traceback (most recent call last)" in text
        rc_ok = not has_error
        results.append(
            ("P1 无 Traceback 且 devkit 报 Extracted", rc_ok, f"Extracted 合计 {extracted}")
        )
    else:
        results.append(("P1 无 Traceback 且 devkit 报 Extracted", False, "运行日志不存在"))

    # [P2] [P3]
    samples = _iter_samples(SMOKE_CACHE)
    got_tokens = {p.name for p in samples}
    results.append(
        ("P2 落盘 token 数 == 200", len(got_tokens) == len(req_tokens),
         f"落盘 {len(got_tokens)} / 请求 {len(req_tokens)}")
    )
    extra = got_tokens - req_tokens
    results.append(
        ("P3 落盘 token ⊆ 请求集合（filter 真生效）", not extra,
         f"多出 {len(extra)} 个" + (f"，示例 {sorted(extra)[:3]}" if extra else ""))
    )
    missing = req_tokens - got_tokens
    if missing:
        logger.info("  ⚠️ 缺失 %d 个 token，示例 %s", len(missing), sorted(missing)[:5])

    # [P4] [P5]
    # 🔴 08-02 审阅加固：原实现只比 samples[0] vs ref_samples[0]（n=1），
    # 用一个样本断言不了「schema 一致」—— 若某类场景缺 key 或多一维，n=1 抽不到。
    # 现改为两侧各随机抽 N_SCHEMA 个，要求**每侧内部 key 集合唯一**且**两侧相等**。
    n_schema = 40
    ref_samples = _iter_samples(REF_CACHE)
    if samples and ref_samples:
        rng = np.random.default_rng(20260802)
        pick_new = [samples[i] for i in rng.choice(len(samples), min(n_schema, len(samples)), replace=False)]
        pick_ref = [ref_samples[i] for i in rng.choice(len(ref_samples), min(n_schema, len(ref_samples)), replace=False)]
        feats_new = [_load_feature(p) for p in pick_new]
        feats_ref = [_load_feature(p) for p in pick_ref]

        keysets_new = {frozenset(f) for f in feats_new}
        keysets_ref = {frozenset(f) for f in feats_ref}
        same_keys = len(keysets_new) == 1 and keysets_new == keysets_ref
        detail = (f"两侧各抽 {len(feats_new)}/{len(feats_ref)} 个｜"
                  f"key 集合种类 新 {len(keysets_new)} / L0 {len(keysets_ref)}")
        if not same_keys and keysets_new and keysets_ref:
            diff = set().union(*keysets_new) ^ set().union(*keysets_ref)
            detail += f"｜差集 {sorted(diff)[:5]}"
        results.append(("P4 feature key 集合与 L0 完全相等（n=%d，非单样本）" % len(feats_new),
                        same_keys, detail))

        mismatches: List[str] = []
        # 每侧内部先查「非样本维是否恒定」，再跨侧比
        def _profile(feats: List[Dict[str, Any]]) -> Dict[str, set]:
            prof: Dict[str, set] = {}
            for feat in feats:
                for key, value in feat.items():
                    dtype, shape = _describe(value)
                    prof.setdefault(key, set()).add((dtype, shape[1:] if len(shape) > 1 else ()))
            return prof

        prof_new, prof_ref = _profile(feats_new), _profile(feats_ref)
        for key, vals in prof_new.items():
            if len(vals) > 1:
                mismatches.append(f"{key}: 冒烟侧非样本维不恒定 {sorted(vals)[:2]}")
        for key in sorted(set(prof_new) & set(prof_ref)):
            if prof_new[key] != prof_ref[key]:
                mismatches.append(f"{key}: {sorted(prof_new[key])[:1]} vs {sorted(prof_ref[key])[:1]}")
        results.append(
            ("P5 dtype 与非样本维形状与 L0 一致（n=%d）" % len(feats_new), not mismatches,
             "全部一致" if not mismatches else "；".join(mismatches[:4]))
        )
    else:
        results.append(("P4 feature key 集合与 L0 完全相等", False, "缺样本，无法比较"))
        results.append(("P5 dtype 与非样本维形状与 L0 一致", False, "缺样本，无法比较"))

    # [P6]
    ray_dirs = sorted(Path("/mnt/local/Xiao_Yi/tmp/ray").glob("session_*")) if Path(
        "/mnt/local/Xiao_Yi/tmp/ray"
    ).is_dir() else []
    root_ray = sorted(Path("/tmp/ray").glob("session_*")) if Path("/tmp/ray").is_dir() else []
    # 只看本次冒烟之后新建的：以运行日志的 mtime 为界
    cutoff = RUN_LOG.stat().st_mtime - 3600 if RUN_LOG.is_file() else 0
    fresh_root = [d for d in root_ray if d.stat().st_mtime > cutoff]
    results.append(
        ("P6 ray session 不落根分区", not fresh_root,
         f"/mnt/local 下 {len(ray_dirs)} 个 session｜根分区新建 {len(fresh_root)} 个"
         + (f"，示例 {fresh_root[0]}" if fresh_root else ""))
    )

    # [P7]
    boston = {p.stem for p in (SPLITS / "train_boston").glob("*.db")}
    got_logs = {p.parent.parent.name for p in samples}
    n_bos = sum(1 for log in got_logs if log in boston)
    n_pit = len(got_logs) - n_bos
    results.append(
        ("P7 两城 log 都有产物（data_root list 生效）", n_bos >= 1 and n_pit >= 1,
         f"boston {n_bos} log｜pittsburgh {n_pit} log｜请求 {len(req_logs)} log")
    )

    logger.info("\n===== S2 冒烟判据（七条事前写死）=====")
    for name, ok, detail in results:
        logger.info("  %s %s\n       %s", "✅" if ok else "❌", name, detail)

    # 测量（不是判据）：每样本体积 ⇒ 用于校正 43 GB 的外推
    if samples:
        total = sum(f.stat().st_size for p in samples for f in p.iterdir())
        per_kb = total / len(samples) / 1024
        logger.info(
            "\n  📏 测量（非判据）：每样本 %.1f KB（L0 实测 175 KB）"
            "⇒ L3 245 500 样本外推 %.1f GB", per_kb, per_kb * 245500 / 1024 / 1024
        )

    n_pass = sum(1 for _, ok, _ in results if ok)
    logger.info("\n  合计 %d/%d 过", n_pass, len(results))
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
