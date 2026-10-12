#!/usr/bin/env python
"""`s3_merge_parts.py` / `s3_report.py` 的合并与读数判据（2026-08-26 立）。

为什么单独立一个文件：`test_s3_edge_utility_labels.py` 管的是**标签怎么算**，
这里管的是**多进程切分产物怎么拼回去、拼回去以后怎么读**。两次真实事故都出在后者，
而且**都只在「多分片」时才现形** —— 单分片的 val 全程正常，train 一跑就炸：

* 2026-08-26 07:08 **合并被 OOM killer 杀掉，日志连 traceback 都没有**。
  根因：`np.load` 的 `NpzFile` 每取一次键就整份解压，而 `z[k][sl]` 是**视图**，
  把整份解压数组一起钉住。实测每场景驻留 **56.0 MiB** ⇒ 单 part 外推 154 GiB。
* 同日 10:38 `s3_report.py` 把 train 的场景数报成 **7,915,962**（真值 19,826）。
  根因：`base = len(scene_of)` 取的是**已累计的对数**而不是已累计的场景数；
  单分片时 `base` 恒为 0 ⇒ 侥幸正确。

判据
----

[M1] **多 part × 多 shard 的合并等价** —— 与「逐场景朴素参考实现」逐键 dtype+shape+逐位相同，
     且场景数不整除 `shard_size`（专门覆盖尾分片）。
[M2] **合并的峰值内存与「每分片对数」解耦** —— 造一个 200 场景 × 4000 对的分片（33.6 MiB），
     旧实现每场景钉住整份分片 ⇒ 驻留约 **6.7 GiB**；新实现实测 **0.07 GiB**。
     实测子进程峰值 RSS 必须低于硬上界。
     🔴 这条不是性能测试，是**正确性测试**：它锁的是「不得逐场景切视图」这个实现约束。
[M3] **读数侧的场景计数在多分片下正确** —— `report.json` 的 `n_scenes` 必须等于
     `meta.json` 的 `n_scenes_labelled`，且远小于 `n_pairs`。
[M4] **三道校验仍然抓得住漏段与重叠** —— 各注入一次，必须非零退出并给出可读原因。

跑法::

    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    /home/team/anaconda3/envs/nuplan/bin/python tests/test_s3_merge_report.py
"""
from __future__ import annotations

import json
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for _p in (str(_GNN), str(_GNN / "scripts" / "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PY = "/home/team/anaconda3/envs/nuplan/bin/python"
MERGE = _GNN / "scripts/analysis/s3_merge_parts.py"
REPORT = _GNN / "scripts/analysis/s3_report.py"

#: 与 `s3_merge_parts.py` 逐字一致（写死在这里是故意的：脚本改了字段，本测试必须一起改）。
PAIR_FIELDS = ("src", "dst", "U", "I", "S", "placebo", "placebo_sd", "U_sd_teacher",
               "err_base_dst", "n_edges_cut", "dst_stratum", "src_stratum")
SCENE_FIELDS = ("n_entity", "n_veh", "n_lane", "n_placebo", "noop_max",
                "cut_d2c_mean", "cut_p2c_mean")
PAIR_DTYPE = {"src": np.int32, "dst": np.int32, "n_edges_cut": np.int32,
              "dst_stratum": np.uint8, "src_stratum": np.uint8}
SCENE_DTYPE = {"n_entity": np.int64, "n_veh": np.int64, "n_lane": np.int64,
               "n_placebo": np.int64, "noop_max": np.float64,
               "cut_d2c_mean": np.float32, "cut_p2c_mean": np.float32}
#: [M2] 的硬上界。旧实现在该规模下驻留约 6.7 GiB，新实现实测 0.07 GiB ⇒ 阈值放在中间，
#: 上方 4.5 倍、下方 20 倍余量，不是在噪声上追阈值（§8.2「不得在重尾上追阈值」的同一条纪律）。
MEM_LIMIT_GIB = 1.5

_FAILED: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


# ── 造数据 ──────────────────────────────────────────────────────────────────
def _base_meta(scene_range: Tuple[int, int], n_dataset: int, n_skipped: int = 0) -> Dict:
    """一份口径字段齐全的 part meta（CALIBER_KEYS 全覆盖，否则 `_check_caliber` 会误判）。"""
    return {"version": "s3-test", "teacher_sha256": ["aa" * 32], "err_metric": "minade",
            "dose": "D2c", "placebo": "P2c", "n_placebo": 4, "seed": 20260826,
            "adjacency_config": "default", "root": "/dev/null", "device": "cpu",
            "noop_tol": 0.0, "n_dataset": n_dataset, "teacher_aggregation": "median",
            "interv_chunk": 8, "rng": "default_rng([seed, scene_index])",
            "teachers": ["t1"], "n_teachers": 1, "caveats": ["测试用"],
            "scene_range": list(scene_range), "n_scenes_labelled": 0,
            "n_skipped": n_skipped, "n_shards": 0}


def _make_part(d: Path, scene_range: Tuple[int, int], per_scene: List[int],
               shard_size: int, rng: np.random.Generator) -> List[Dict]:
    """写出一个 part（含若干 shard），并返回逐场景的真值记录，供朴素参考实现比对。"""
    d.mkdir(parents=True, exist_ok=True)
    truth: List[Dict] = []
    k = 0
    for st in range(0, len(per_scene), shard_size):
        grp = per_scene[st:st + shard_size]
        recs = []
        for j, n in enumerate(grp):
            r = {"token": f"{d.name}_s{st + j:05d}"}
            for f in PAIR_FIELDS:
                dt = PAIR_DTYPE.get(f, np.float32)
                r[f] = (rng.integers(0, 60, size=n).astype(dt) if np.issubdtype(dt, np.integer)
                        else rng.standard_normal(n).astype(dt))
            for f in SCENE_FIELDS:
                dt = SCENE_DTYPE[f]
                r[f] = (dt(rng.integers(1, 90)) if np.issubdtype(dt, np.integer)
                        else dt(rng.standard_normal()))
            r["noop_max"] = np.float64(0.0)          # CPU 档必须逐位恒等
            recs.append(r)
        ptr = np.cumsum([0] + [int(r["src"].size) for r in recs]).astype(np.int64)
        np.savez_compressed(
            d / f"labels_{k:04d}.npz", scene_ptr=ptr,
            token=np.array([r["token"] for r in recs], dtype="U64"),
            **{f: np.concatenate([r[f] for r in recs]) for f in PAIR_FIELDS},
            **{f: np.array([r[f] for r in recs], dtype=SCENE_DTYPE[f]) for f in SCENE_FIELDS})
        truth.extend(recs)
        k += 1
    m = _base_meta(scene_range, 0)
    m["n_scenes_labelled"] = len(per_scene)
    m["n_shards"] = k
    (d / "meta.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return truth


def _finalize_n_dataset(root: Path, n_dataset: int) -> None:
    """`n_dataset` 属口径字段，必须全 part 一致 —— 造完全部 part 后统一回填。"""
    for f in sorted(root.glob("part_*/meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        m["n_dataset"] = n_dataset
        f.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def _run_merge(root: Path, shard_size: int = 4000) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(MERGE), "--root", str(root), "--shard_size", str(shard_size)],
                          capture_output=True, text=True)


def _naive_reference(truth: List[Dict], shard_size: int) -> List[Dict]:
    """朴素参考实现：按 shard_size 切场景，逐场景拼接 —— 与被测脚本完全独立的代码路径。"""
    out = []
    for st in range(0, len(truth), shard_size):
        grp = truth[st:st + shard_size]
        out.append({
            "scene_ptr": np.cumsum([0] + [int(g["src"].size) for g in grp]).astype(np.int64),
            "token": np.array([g["token"] for g in grp], dtype="U64"),
            **{f: np.concatenate([g[f] for g in grp]) for f in PAIR_FIELDS},
            **{f: np.array([g[f] for g in grp], dtype=SCENE_DTYPE[f]) for f in SCENE_FIELDS}})
    return out


# ── [M1] 多 part × 多 shard 的合并等价 ──────────────────────────────────────
def test_m1_merge_equivalence() -> None:
    print("\n[M1] 多 part × 多 shard 合并等价（含尾分片）")
    rng = np.random.default_rng(20260826)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        truth: List[Dict] = []
        lo = 0
        # 三个 part，每个 2 个 shard；场景数 7/5/9 = 21，shard_size=4 ⇒ 6 个输出分片且尾分片不满
        for i, n_sc in enumerate((7, 5, 9)):
            per = [int(x) for x in rng.integers(3, 40, size=n_sc)]
            truth += _make_part(root / f"part_{i:02d}", (lo, lo + n_sc), per, shard_size=4, rng=rng)
            lo += n_sc
        _finalize_n_dataset(root, lo)

        r = _run_merge(root, shard_size=4)
        check(r.returncode == 0, f"[M1] 合并成功退出（rc={r.returncode}）{r.stderr[-200:]}")
        if r.returncode != 0:
            return

        ref = _naive_reference(truth, shard_size=4)
        got = sorted(root.glob("labels_*.npz"))
        check(len(got) == len(ref) == 6, f"[M1] 分片数 {len(got)} == 参考 {len(ref)} == 6（21/4 向上取整）")
        n_bad = 0
        for a, b in zip(got, ref):
            z = np.load(a)
            for k in ("scene_ptr", "token", *PAIR_FIELDS, *SCENE_FIELDS):
                x, y = z[k], b[k]
                if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                    n_bad += 1
                    print(f"       🔴 {a.name}:{k} dtype {x.dtype}/{y.dtype} shape {x.shape}/{y.shape}")
        check(n_bad == 0, f"[M1] 全部 {len(got)} 分片 × 21 个键 dtype+shape+逐位与参考实现相同")

        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        check(meta["n_scenes_labelled"] == 21, f"[M1] meta 场景数 {meta['n_scenes_labelled']} == 21")
        check(meta["n_pairs"] == sum(int(g["src"].size) for g in truth), "[M1] meta 对数与真值一致")
        check(meta["noop_max_observed"] == 0.0, "[M1] 空操作门 max 逐位 0.0（CPU 档）")


# ── [M2] 峰值内存与「每分片对数」解耦 ───────────────────────────────────────
def test_m2_memory() -> None:
    print(f"\n[M2] 合并峰值内存 < {MEM_LIMIT_GIB} GiB（旧实现在该规模下驻留约 6.7 GiB）")
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        n_sc, n_pair = 200, 4000            # 单个 shard 内 80 万对 ≈ 33.6 MiB
        _make_part(root / "part_00", (0, n_sc), [n_pair] * n_sc, shard_size=n_sc, rng=rng)
        _finalize_n_dataset(root, n_sc)
        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        r = _run_merge(root, shard_size=n_sc)
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        check(r.returncode == 0, f"[M2] 合并成功退出（rc={r.returncode}）")
        gib = peak / 1024 / 1024
        # 旧实现每个场景都钉住**整份分片数组**（n_sc × n_pair 对）⇒ 驻留 ≈ n_sc × 分片体积。
        naive = n_sc * (n_sc * n_pair * 42) / 1024 ** 3
        check(gib < MEM_LIMIT_GIB,
              f"[M2] 子进程峰值 RSS {gib:.3f} GiB < {MEM_LIMIT_GIB} GiB"
              f"（逐场景切视图的实现在此规模下需约 {naive:.1f} GiB）")
        _ = before


# ── [M3] 读数侧的场景计数在多分片下正确 ─────────────────────────────────────
def test_m3_report_scene_count() -> None:
    print("\n[M3] 多分片下 report 的 n_scenes 正确（2026-08-26 报成 7,915,962 的那个 bug）")
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        n_sc = 30
        _make_part(root / "part_00", (0, n_sc), [int(x) for x in rng.integers(20, 60, size=n_sc)],
                   shard_size=n_sc, rng=rng)
        _finalize_n_dataset(root, n_sc)
        r = _run_merge(root, shard_size=7)      # 30/7 ⇒ 5 个分片（末片 2 个场景）
        check(r.returncode == 0, f"[M3] 合并成功（rc={r.returncode}）")
        n_shard = len(list(root.glob("labels_*.npz")))
        check(n_shard == 5, f"[M3] 确实产出多分片（{n_shard} 个）—— 单分片测不出这个 bug")

        out = root / "report.json"
        rr = subprocess.run([PY, str(REPORT), "--root", str(root), "--out", str(out)],
                            capture_output=True, text=True)
        check(rr.returncode == 0, f"[M3] 报告成功（rc={rr.returncode}）{rr.stderr[-300:]}")
        if rr.returncode != 0:
            return
        rep = json.loads(out.read_text(encoding="utf-8"))
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        check(rep["n_scenes"] == meta["n_scenes_labelled"] == n_sc,
              f"[M3] report n_scenes {rep['n_scenes']} == meta {meta['n_scenes_labelled']} == {n_sc}")
        check(rep["n_pairs"] == meta["n_pairs"], "[M3] report n_pairs == meta n_pairs")

        # 交叉校验门本身必须有牙：篡改 meta 后报告必须拒绝产出
        meta["n_scenes_labelled"] = n_sc + 1
        (root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        rr2 = subprocess.run([PY, str(REPORT), "--root", str(root)], capture_output=True, text=True)
        check(rr2.returncode != 0 and "不一致" in (rr2.stdout + rr2.stderr),
              "[M3] 篡改 meta 后交叉校验门当场停（不是只打个 warning）")


# ── [M4] 覆盖校验仍抓得住漏段与重叠 ─────────────────────────────────────────
def test_m4_coverage_guards() -> None:
    print("\n[M4] 漏段 / 重叠 / 口径不一致 —— 三道校验必须各自拦下")
    rng = np.random.default_rng(3)
    for name, mutate, kw in (
        ("漏段", lambda ms: ms[1].__setitem__("scene_range", [12, 20]), "没有任何 part 覆盖"),
        ("重叠", lambda ms: ms[1].__setitem__("scene_range", [8, 20]), "重叠"),
        ("口径不一致", lambda ms: ms[1].__setitem__("err_metric", "top1"), "口径不同"),
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _make_part(root / "part_00", (0, 10), [5] * 10, 10, rng)
            _make_part(root / "part_01", (10, 20), [5] * 10, 10, rng)
            _finalize_n_dataset(root, 20)
            metas = []
            for f in sorted(root.glob("part_*/meta.json")):
                metas.append(json.loads(f.read_text(encoding="utf-8")))
            mutate(metas)
            for f, m in zip(sorted(root.glob("part_*/meta.json")), metas):
                f.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            r = _run_merge(root, shard_size=10)
            txt = r.stdout + r.stderr
            check(r.returncode != 0 and kw in txt,
                  f"[M4] {name}：非零退出（rc={r.returncode}）且报出「{kw}」")
            check(not list(root.glob("labels_*.npz")),
                  f"[M4] {name}：拒绝产出残缺标签（根层无 labels_*.npz）")


if __name__ == "__main__":
    for f in (MERGE, REPORT):
        if not f.exists():
            print(f"🔴 缺 {f}")
            sys.exit(1)
    test_m1_merge_equivalence()
    test_m2_memory()
    test_m3_report_scene_count()
    test_m4_coverage_guards()
    if _FAILED:
        print(f"\n🔴 {len(_FAILED)} 项失败:")
        for f in _FAILED:
            print("   -", f)
        sys.exit(1)
    print("\n✅ test_s3_merge_report 4/4 通过")
