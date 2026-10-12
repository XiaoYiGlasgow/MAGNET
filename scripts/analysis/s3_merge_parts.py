#!/usr/bin/env python
"""把 S3 多进程切分的 `part_*/` 合并成一份冻结产物，并**校验覆盖完整**。

为什么要单独一个脚本：S3 必须跑 CPU（GPU 前向不可复现，§8.2；实测 GPU 侧空操作门读数是
**重尾**的 —— 4300 场景 p99 仅 1.4e-04，却撞到过 2.377e-03），而 CPU 上单进程 ≈0.26 场景/s
⇒ 只能多进程切分。切分后必须证明：**每个场景恰好被处理一次**，不漏不重。

🔴 三道校验，任何一条不过就拒绝合并（宁可没有产物，也不要一份看起来正常的残缺标签）：

1. **口径一致**：全部 part 的 teacher sha 列表、`err_metric`、`dose`、`placebo`、`n_placebo`、
   `seed`、`adjacency_config`、`version`、`root` 必须逐字相同；
2. **覆盖完整**：各 part 的 `scene_range` 排序后必须**首尾相接**且并集恰为 `[0, n_dataset)`；
3. **切分不变**：S3 的 RNG 已改为 `default_rng([seed, scene_index])` 逐场景派生
   （已由 `tests/test_s3_edge_utility_labels.py` 验过切分拼接与整段跑**逐位相同**），
   本脚本只校验 part 之间的 `scene_range` 不重叠 —— 重叠会让同一场景出现两次。

🔴 **合并必须在「数组级」做，不能逐场景切视图**（2026-08-26 实测事故的直接产物）：
`np.load` 返回的 `NpzFile` **每次取键都整份解压**，而 `z[k][sl]` 是**视图** —— 它把整份解压
数组一起钉在内存里。原实现每场景驻留 **56.0 MiB**（`scenes[0]["U"].base.size` = 整个 part 的
对数），train 单 part 外推 **154 GiB**、七 part **1081 GiB** ⇒ 被 OOM killer SIGKILL，
**连 traceback 都没有**（val 侥幸跑通只是因为它每 part 只有约 15.5 万对）。
现改为：每个 shard 的每个键**只解压一次**，按 part 顺序拼成全局扁平数组 + 全局 `scene_ptr`，
再按 `shard_size` 切片写盘。峰值内存约 **1 GiB**，且输出与原实现**逐键逐位相同**
（回归门：val 的 7 个 part 用新代码重跑，与 08-26 00:52 落盘的旧产物 `array_equal` 全过）。

跑法::

    python scripts/analysis/s3_merge_parts.py --root data/edge_labels_s3_20260825/train
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("s3_merge")

#: 必须逐字相同的口径字段（不含随 part 变化的 scene_range / n_scenes_* / n_shards）。
CALIBER_KEYS = ("version", "teacher_sha256", "err_metric", "dose", "placebo", "n_placebo",
                "seed", "adjacency_config", "root", "device", "noop_tol", "n_dataset",
                "teacher_aggregation", "interv_chunk", "rng")
#: 逐对的扁平数组字段（沿 0 轴拼接）。
PAIR_FIELDS = ("src", "dst", "U", "I", "S", "placebo", "placebo_sd", "U_sd_teacher",
               "err_base_dst", "n_edges_cut", "dst_stratum", "src_stratum")
#: 逐场景字段。
SCENE_FIELDS = ("n_entity", "n_veh", "n_lane", "n_placebo", "noop_max",
                "cut_d2c_mean", "cut_p2c_mean")
FROZEN_MARK = ".S3_LABELS_FROZEN"


class MergeError(RuntimeError):
    """校验失败 —— 一律抛出，绝不产出残缺标签。"""


def _load_part(d: Path) -> Dict:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    shards = sorted(d.glob("labels_*.npz"))
    return {"dir": d, "meta": meta, "shards": shards}


def _check_caliber(parts: List[Dict]) -> Dict:
    ref = parts[0]["meta"]
    for p in parts[1:]:
        bad = [k for k in CALIBER_KEYS if ref.get(k) != p["meta"].get(k)]
        if bad:
            raise MergeError(
                f"{p['dir'].name} 与 {parts[0]['dir'].name} 的口径不同：{bad}"
                f"（例：{bad[0]} = {p['meta'].get(bad[0])!r} vs {ref.get(bad[0])!r}）"
                f" —— 不同口径的标签合并会产出一份没人能解释的产物")
    return ref


def _check_coverage(parts: List[Dict], n_dataset: int) -> None:
    rng = sorted((tuple(p["meta"]["scene_range"]), p["dir"].name) for p in parts)
    cur = 0
    for (lo, hi), name in rng:
        if lo < cur:
            raise MergeError(f"{name} 的区间 [{lo},{hi}) 与前一段重叠（前一段到 {cur}）⇒ 场景会重复计入")
        if lo > cur:
            raise MergeError(f"场景 [{cur},{lo}) 没有任何 part 覆盖（下一段是 {name} 的 [{lo},{hi})）")
        cur = hi
    if cur != n_dataset:
        raise MergeError(f"覆盖到 {cur} 但数据集有 {n_dataset} 个场景 —— 尾部 {n_dataset - cur} 个漏了")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="合并 S3 的多进程切分产物并校验覆盖")
    p.add_argument("--root", required=True, help="含 part_*/ 的目录；合并结果写在它下面")
    p.add_argument("--shard_size", type=int, default=4000, help="合并后每个 npz 放几个场景")
    a = p.parse_args()

    root = Path(a.root)
    dirs = sorted(d for d in root.glob("part_*") if (d / "meta.json").exists())
    if not dirs:
        raise MergeError(f"{root} 下没有任何带 meta.json 的 part_*")
    parts = [_load_part(d) for d in dirs]
    ref = _check_caliber(parts)
    _check_coverage(parts, int(ref["n_dataset"]))
    logger.info("✅ 口径一致（%d 个 part）且覆盖 [0, %d) 完整无重叠", len(parts), ref["n_dataset"])

    parts.sort(key=lambda q: q["meta"]["scene_range"][0])

    # ── 读入：每个 shard 的每个键只解压一次，整块留着（不切视图）──────────────
    pair_chunks: Dict[str, List[np.ndarray]] = {f: [] for f in PAIR_FIELDS}
    scene_chunks: Dict[str, List[np.ndarray]] = {f: [] for f in SCENE_FIELDS}
    tokens: List[str] = []
    counts: List[int] = []                    # 逐场景的对数，用来重建 scene_ptr
    for q in parts:
        for f in q["shards"]:
            with np.load(f) as z:
                ptr = np.asarray(z["scene_ptr"])
                n_sc = int(ptr.size) - 1
                if n_sc <= 0:                 # `_flush` 不写空 shard，这里只是兜底
                    continue
                n_pr = int(ptr[-1])
                tokens.extend(str(t) for t in z["token"])
                counts.extend(int(c) for c in np.diff(ptr))
                for k in PAIR_FIELDS:
                    pair_chunks[k].append(np.asarray(z[k])[:n_pr])
                for k in SCENE_FIELDS:
                    scene_chunks[k].append(np.asarray(z[k])[:n_sc])
    n_scenes = len(counts)
    if n_scenes == 0:
        raise MergeError("全部 part 加起来一个场景都没有")

    # 逐字段拼接后立刻释放该字段的碎块，避免峰值翻倍。
    pair_all: Dict[str, np.ndarray] = {}
    for k in PAIR_FIELDS:
        pair_all[k] = np.concatenate(pair_chunks[k])
        pair_chunks[k] = []
    scene_all: Dict[str, np.ndarray] = {}
    for k in SCENE_FIELDS:
        scene_all[k] = np.concatenate(scene_chunks[k])
        scene_chunks[k] = []
    tok_all = np.array(tokens, dtype="U64")
    # 全局 scene_ptr：off[i] = 第 i 个场景在扁平数组里的起点。
    off = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64)))).astype(np.int64)

    # ── 写盘：按 shard_size 个场景切片；场景顺序与 part/shard 顺序逐位一致 ────
    k = 0
    for st in range(0, n_scenes, a.shard_size):
        en = min(st + a.shard_size, n_scenes)
        lo, hi = int(off[st]), int(off[en])
        np.savez_compressed(
            root / f"labels_{k:04d}.npz",
            scene_ptr=(off[st:en + 1] - off[st]).astype(np.int64),
            token=tok_all[st:en],
            **{f: pair_all[f][lo:hi] for f in PAIR_FIELDS},
            **{f: scene_all[f][st:en] for f in SCENE_FIELDS})
        k += 1

    n_pair = int(off[-1])
    noop = float(scene_all["noop_max"].max())
    meta = {**{q: ref[q] for q in CALIBER_KEYS if q in ref},
            "teachers": ref.get("teachers"), "n_teachers": ref.get("n_teachers"),
            "merged_from": [q["dir"].name for q in parts], "n_merged_parts": len(parts),
            "n_scenes_labelled": n_scenes, "n_pairs": n_pair, "n_shards": k,
            "n_skipped": sum(int(q["meta"].get("n_skipped", 0)) for q in parts),
            "noop_max_observed": noop, "caveats": ref.get("caveats", []),
            "script_sha256": None}
    import hashlib
    meta["script_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / FROZEN_MARK).write_text("\n".join(ref["teacher_sha256"]) + "\n", encoding="utf-8")
    if noop != 0.0:
        logger.warning("⚠️ 空操作门实测 max %.3e ≠ 0 —— CPU 档本应逐位恒等，请查是否误跑了 GPU", noop)
    logger.info("✅ 合并完成 %s：%d 场景 / %d 对 / %d 分片 / skip %d / 空操作门 max %.1e",
                root, n_scenes, n_pair, k, meta["n_skipped"], noop)


if __name__ == "__main__":
    try:
        main()
    except MergeError as exc:
        print(f"🔴 {exc}", file=sys.stderr)
        sys.exit(2)
