#!/usr/bin/env python3
"""E13 产物完整性 / provenance 校验（4 拓扑 × 3 seed = 12 格 test eval）。

## 为什么需要这个脚本

test split **只开一次**（`.TEST_SPLIT_FROZEN`，L-15），跑完不能重来。2026-08-18 在开跑前 8 h
截住的第三处缺陷（阶段221）正是「一路 `rc=0`、读数已被污染」的形态：守卫把**格子名**
`f_nearby` 当 `--dataset_config` 传，而 `get_adjacency_config()` 对未注册名**静默回退
DefaultBuilder** ⇒ 用 30 m 近邻图训的三格被放到完全图上评（实测 ADE +0.381 m / +11.5%）。
⇒ 判读之前必须**逐格核对产物 provenance**，而不是看退出码。任一格不符 ⇒ 当场停、先查再判读。

## 校验项（全部来自 `eval_test_set.py` 写进 `test_metrics.json` 的 provenance 字段）

| # | 检查 | 拦的是什么 |
|---|---|---|
| 1 | 12 格产物齐、且**没有第 13 个** `test_metrics.json` | 半套数据判读 / 回归门产物被误计 |
| 2 | `dataset_config` == 该格 ckpt `train_meta.json` 的 `dataset_config` | 阶段221 邻接错配 |
| 3 | `dataset_config` == 期望映射（`f_nearby→nearby_only` 等） | 目录名与 builder 名的错位 |
| 4 | `dataset_config_source == "ckpt_train_meta"` | 有人手动传参绕过 sidecar |
| 5 | `follow_batch == ["gnn_x"]` | D0#7 同族：逐样本池化退化成整批池化 |
| 6 | `batch_size == 8` | BN(track_running_stats=False) 使读数随 batch 变，跨格不可比 |
| 7 | `n_samples == 5943` 且 12 格**全相等** | 建图期 skip 数不同 ⇒ 逐样本配对失效 |
| 8 | `split_manifest` 的 sha16 / token 数 | 评的不是那份冻结 split |
| 9 | `train_seed` == 目录名里的 seed；`checkpoint` 指向该格 `best_model_default.pth` | 张冠李戴 |
| 10 | `model_class == MagnetImitationPooled`、`gnn_pooling_fix_effective` 为真 | 裸建 TransformerPredictor 的旧缺陷 |
| 11 | `test_per_scene.csv` 行数 == `n_samples` | 产物半写 |

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/e13_verify_provenance.py
    # 退出码 0 = 全过；1 = 有不符项（当场停，不要进判读）；2 = 产物不齐
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
DEFAULT_ROOT = GNN / "experiments" / "test_eval_20260818"
KNOBS = GNN / "experiments" / "knobs"

#: 格子目录名 → 该格训练时的 builder 名（= 期望的 `dataset_config`）。
#: 🔴 `f_nearby → nearby_only` 是阶段221 那处缺陷的落点，不得简化成「同名」。
CONFIG_TO_BUILDER = {
    "fr_base": "default",
    "ego_only": "ego_only",
    "no_ego_road": "no_ego_road",
    "f_nearby": "nearby_only",
}
SEEDS = ("42", "1", "2")
EXPECT_N_TOKENS = 5943
EXPECT_SHA16 = "43b31f84d8d42048"
EXPECT_BATCH = 8
EXPECT_MODEL_CLASS = "MagnetImitationPooled"


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_cell(root: Path, cfg: str, seed: str) -> Tuple[List[str], Dict]:
    """返回 (不符项列表, 该格摘要)。摘要在全过时供判读使用。"""
    bad: List[str] = []
    cell = root / f"{cfg}_seed{seed}"
    mpath = cell / "test_metrics.json"
    if not mpath.exists():
        return [f"缺产物 {mpath}"], {}
    m = _load(mpath)

    ckpt = KNOBS / f"{cfg}_L1" / f"seed{seed}" / "models" / "best_model_default.pth"
    meta_path = ckpt.parent / "train_meta.json"
    train_meta = _load(meta_path) if meta_path.exists() else {}

    expect_builder = CONFIG_TO_BUILDER[cfg]
    got_cfg = m.get("dataset_config")
    if got_cfg != expect_builder:
        bad.append(f"dataset_config={got_cfg!r} ≠ 期望 {expect_builder!r}")
    meta_cfg = train_meta.get("dataset_config")
    if meta_cfg != got_cfg:
        bad.append(f"dataset_config={got_cfg!r} 与 ckpt sidecar {meta_cfg!r} 不一致")
    if m.get("dataset_config_source") != "ckpt_train_meta":
        bad.append(f"dataset_config_source={m.get('dataset_config_source')!r} ≠ 'ckpt_train_meta'")

    if m.get("follow_batch") != ["gnn_x"]:
        bad.append(f"follow_batch={m.get('follow_batch')!r} ≠ ['gnn_x']（逐样本池化会退化）")
    if m.get("batch_size") != EXPECT_BATCH:
        bad.append(f"batch_size={m.get('batch_size')!r} ≠ {EXPECT_BATCH}")
    if m.get("n_samples") != EXPECT_N_TOKENS:
        bad.append(f"n_samples={m.get('n_samples')!r} ≠ {EXPECT_N_TOKENS}")

    sm = m.get("split_manifest") or {}
    if sm.get("test_logs_sha256_16") != EXPECT_SHA16:
        bad.append(f"split sha16={sm.get('test_logs_sha256_16')!r} ≠ {EXPECT_SHA16!r}")
    if sm.get("n_tokens_test") != EXPECT_N_TOKENS:
        bad.append(f"split n_tokens_test={sm.get('n_tokens_test')!r} ≠ {EXPECT_N_TOKENS}")

    if str(m.get("train_seed")) != seed:
        bad.append(f"train_seed={m.get('train_seed')!r} ≠ 目录名的 {seed}")
    if Path(m.get("checkpoint", "")).resolve() != ckpt.resolve():
        bad.append(f"checkpoint={m.get('checkpoint')!r} ≠ {ckpt}")

    if m.get("model_class") != EXPECT_MODEL_CLASS:
        bad.append(f"model_class={m.get('model_class')!r} ≠ {EXPECT_MODEL_CLASS!r}")
    arch = m.get("architecture_resolved") or {}
    if not arch.get("gnn_pooling_fix_effective"):
        bad.append(f"architecture_resolved.gnn_pooling_fix_effective={arch.get('gnn_pooling_fix_effective')!r} 非真")
    # subgraph_agg：fr_base 的 sidecar 缺该键（回退 mean），其余三格显式 mean —— 两种都必须解析成 mean
    if m.get("subgraph_agg") != "mean":
        bad.append(f"subgraph_agg={m.get('subgraph_agg')!r} ≠ 'mean'")

    csv_path = cell / "test_per_scene.csv"
    n_rows = None
    if not csv_path.exists():
        bad.append(f"缺 {csv_path}")
    else:
        with open(csv_path, newline="", encoding="utf-8") as f:
            n_rows = sum(1 for _ in csv.reader(f)) - 1  # 去表头
        if n_rows != m.get("n_samples"):
            bad.append(f"test_per_scene.csv {n_rows} 行 ≠ n_samples {m.get('n_samples')}")

    summary = {
        "cell": f"{cfg}/seed{seed}", "dataset_config": got_cfg,
        "ade_mean": m.get("ade_mean"), "fde_mean": m.get("fde_mean"),
        "n_samples": m.get("n_samples"), "git_rev": (m.get("git_rev") or "")[:8],
        "created": m.get("created"), "eval_seconds": m.get("eval_seconds"), "csv_rows": n_rows,
    }
    return bad, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="E13 产物根目录（绝对路径）")
    a = ap.parse_args()
    root: Path = a.root

    # 检查 1：没有第 13 个 test_metrics.json（回归门产物已被守卫改名，若又冒出来说明通路变了）
    found = sorted(root.rglob("test_metrics.json")) if root.exists() else []
    print(f"产物根目录：{root}")
    print(f"找到 test_metrics.json：{len(found)} 个（期望 12）")

    all_bad: Dict[str, List[str]] = {}
    rows: List[Dict] = []
    missing = 0
    for cfg in CONFIG_TO_BUILDER:
        for seed in SEEDS:
            bad, summ = check_cell(root, cfg, seed)
            if not summ:
                missing += 1
            else:
                rows.append(summ)
            if bad:
                all_bad[f"{cfg}/seed{seed}"] = bad

    expected_paths = {root / f"{c}_seed{s}" / "test_metrics.json"
                      for c in CONFIG_TO_BUILDER for s in SEEDS}
    extra = [p for p in found if p not in expected_paths]
    if extra:
        all_bad["__extra__"] = [f"非期望的产物：{p}" for p in extra]

    # 检查 7 的跨格部分：12 格 n_samples 必须全相等（否则逐样本口径不可并列）
    ns = {r["n_samples"] for r in rows if r.get("n_samples") is not None}
    if len(ns) > 1:
        all_bad["__cross__"] = [f"12 格 n_samples 不一致：{sorted(ns)}（建图期 skip 数不同 ⇒ 不可并列）"]

    print()
    print(f"{'格子':<20}{'builder':<14}{'ADE':>9}{'FDE':>9}{'n':>7}  {'git':<10}{'created'}")
    for r in rows:
        print(f"{r['cell']:<20}{str(r['dataset_config']):<14}"
              f"{(r['ade_mean'] or float('nan')):>9.4f}{(r['fde_mean'] or float('nan')):>9.4f}"
              f"{r['n_samples']:>7}  {r['git_rev']:<10}{r['created']}")

    print()
    if missing:
        print(f"🔴 {missing} 格产物缺失 ⇒ 不进判读")
    if all_bad:
        print("🔴 provenance 不符项：")
        for k, v in all_bad.items():
            for line in v:
                print(f"  [{k}] {line}")
        return 1
    if missing:
        return 2
    print("✅ 12 格 provenance 全部通过（邻接/池化/batch/split/seed/ckpt/行数）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
