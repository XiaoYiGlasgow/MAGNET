#!/usr/bin/env python3
"""Q74 2×2 通路分解 —— k0 套扩到三个 training seed（42/1/2）的合并读数（Q96 补强①）。

判据仍是 🧊 `预注册判据/Q74_2x2通路分解_预注册判据_20260810_冻结.md` §3，**一字不改**；
本脚本只是把 `q74_readout_20260812.py`（单套 ckpt）推广到多 training seed，
聚合层级逐字复用 `paired_usage_delta.py` 的四层口径（audit seed→token 中位 / token 配对
两侧非塌陷 / **training seed→token 中位（seed-matched，不堆 token）** / token→中位 + log-cluster CI）。

产出（每个数字都带 pooled + per-training-seed，第七十九/八十七条：3 点不做区间宣称，
seed 间符号不一致只写「pooled X，seed 间不一致」）：
- **per-training-seed**：三套各自完整跑一遍 `q74_readout_20260812.py` 的读数（含 per-audit-seed），
  seed42 那份须与既有 `q74_k0_seed42/q74_readout_20260812.json` 的 pooled 块**逐位一致**（回归门）；
- **pooled(3 training seed)**：token 在三套 × 三格全非塌陷的公共集上，先对 training seed 取中位，
  再走 `readout_block`（Q74-1 G4 使用度 / Q74-2 G3/G2 份额 / Q74-3 可加性只报不判）；
- **Δ(G3−G2)、Δ(G4−G2) 三 seed 配对差**：直接调 `PUD.paired_cell`（seed-matched），
  自带 per-training-seed δ 与 `sign_consistent_across_seeds`；
- **符号一致性表**：G3/G2>1、Δ(G3−G2)>0、Δ(G4−G2)>0、Q74-3 d 的符号在三 seed 上是否一致。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/q74_readout_3seed_20260817.py                 # 42,1,2 全合并
    $PY scripts/analysis/q74_readout_3seed_20260817.py --seeds 42      # 只跑 seed42（回归门自检）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
AUDIT = GNN / "experiments" / "interaction_audit"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, GNN / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PUD = _load("paired_usage_delta", "scripts/analysis/paired_usage_delta.py")
Q = _load("q74_readout_20260812", "scripts/analysis/q74_readout_20260812.py")
CELLS = Q.CELLS
LOG_MANIFEST = Q.LOG_MANIFEST
EXPECT = Q.SUITES["k0"]          # model_class=MagnetImitationPooled, gnn_pooling_fix=True（三 seed 同一期望）
SEED_DIR = {42: AUDIT / "q74_k0_seed42", 1: AUDIT / "q74_k0_seed1", 2: AUDIT / "q74_k0_seed2"}


def single_seed_readout(suite_dir: Path, t2l: Dict[str, str], n_boot: int, boot_seed: int) -> Dict:
    """逐字重放 q74_readout_20260812.main() 的单套读数（不改口径），另把 per-token 中间量带出供 3-seed 池化。"""
    _, g1_meta = Q.load_cell_shards(suite_dir, "G1", EXPECT)
    recs, meta = {}, {"G1": g1_meta}
    for cell in CELLS:
        recs[cell], meta[cell] = Q.load_cell_shards(suite_dir, cell, EXPECT)
    if any(m["max_noop_disp"] != 0.0 for m in meta.values()):
        raise ValueError(f"🔴 {suite_dir.name} 空操作证据非零：{meta}")
    pt = {cell: PUD.per_token(recs[cell]) for cell in CELLS}
    common = sorted(set.intersection(*(set(v) for v in pt.values())))
    common = [k for k in common if not any(pt[c][k]["collapsed"] for c in CELLS)]
    unmapped = [k for k in common if k not in t2l]
    if unmapped:
        raise ValueError(f"🔴 {suite_dir.name}: {len(unmapped)}/{len(common)} token 不在 log_manifest，拒绝出数")
    logs = np.array([t2l[k] for k in common])
    u = {cell: np.array([pt[cell][k]["ratio"] for k in common]) for cell in CELLS}
    rom = {cell: (float(np.median([pt[cell][k]["mean_disp"] for k in common]))
                  / float(np.median([pt[cell][k]["base_scale"] for k in common]))) for cell in CELLS}
    pooled = Q.readout_block(u, logs, rom, n_boot, boot_seed)
    ps = {cell: Q.per_token_per_seed(recs[cell]) for cell in CELLS}
    per_audit = {}
    for s in Q.AUDIT_SEEDS:
        ks = [k for k in common if all(k in ps[c][s] for c in CELLS)]
        lg = np.array([t2l[k] for k in ks])
        us = {cell: np.array([ps[cell][s][k] for k in ks]) for cell in CELLS}
        per_audit[str(s)] = Q.readout_block(us, lg, {c: None for c in CELLS}, n_boot, boot_seed)
    return {"suite_dir": str(suite_dir),
            "n_collapsed_head": {c: meta[c]["n_collapsed_head"] for c in meta},
            "pooled": pooled, "per_audit_seed": per_audit,
            "_pt": pt, "_common": common}


def crosscheck_existing(suite_dir: Path, pooled: Dict) -> Dict:
    """与该 seed 目录下既有 paired_usage_delta 产物（守卫出的 *_paired_*.json）逐位对账。"""
    out = {}
    for treat in ("G3", "G4"):
        cands = sorted(suite_dir.glob(f"q74_{treat}_vs_G2_paired_*.json"))
        if not cands:
            out[f"{treat}_vs_G2"] = {"file": None, "checks": "无既有配对产物（守卫未出或未跑），跳过对账"}
            continue
        f = max(cands, key=lambda p: p.stat().st_mtime)
        ref = json.loads(f.read_text())["median_of_ratios"]
        n_ref = json.loads(f.read_text())["n_paired"]
        checks = {"n_paired": (pooled["n_paired"], n_ref),
                  "base(G2)": (pooled["usage_G2"]["median_of_ratios"], ref["base"]),
                  f"treat({treat})": (pooled[f"usage_{treat}"]["median_of_ratios"], ref["treat"])}
        for name, (a, b) in checks.items():
            if abs(float(a) - float(b)) >= 1e-12:
                raise ValueError(f"🔴 对账失败 {suite_dir.name}/{f.name}:{name} 重算 {a!r} vs 已落盘 {b!r}")
        out[f"{treat}_vs_G2"] = {"file": f.name, "checks": "逐位一致(<1e-12)",
                                 "paired_delta_median": ref["paired_delta_median"],
                                 "paired_delta_ci95_logcluster": ref["paired_delta_ci95_logcluster"],
                                 "excludes_zero": ref["excludes_zero"]}
    return out


def pooled_over_training_seeds(per_seed: Dict[int, Dict], t2l: Dict[str, str],
                               n_boot: int, boot_seed: int) -> Dict:
    """第 3 层：training seed → token 取中位（seed-matched，只留三套三格全非塌陷的公共 token）。"""
    seeds = sorted(per_seed)
    common = sorted(set.intersection(*(set(per_seed[s]["_common"]) for s in seeds)))
    if not common:
        raise ValueError("三 seed 公共配对 token 为空，拒绝出数")
    logs = np.array([t2l[k] for k in common])
    u = {cell: np.array([float(np.median([per_seed[s]["_pt"][cell][k]["ratio"] for s in seeds]))
                         for k in common]) for cell in CELLS}
    rom = {}
    for cell in CELLS:
        md = float(np.median([np.median([per_seed[s]["_pt"][cell][k]["mean_disp"] for s in seeds]) for k in common]))
        bs = float(np.median([np.median([per_seed[s]["_pt"][cell][k]["base_scale"] for s in seeds]) for k in common]))
        rom[cell] = md / bs if bs else float("nan")
    blk = Q.readout_block(u, logs, rom, n_boot, boot_seed)
    blk["n_training_seeds"] = len(seeds)
    blk["training_seeds"] = seeds
    # 同一公共 token 集上的 per-training-seed 读数（与 pooled 同 n，便于像对像比较；
    # 各 seed 自己全集上的读数在 per_training_seed 块，两者 n 不同，勿混）
    blk["per_training_seed_on_common_tokens"] = {}
    for s in seeds:
        us = {cell: np.array([per_seed[s]["_pt"][cell][k]["ratio"] for k in common]) for cell in CELLS}
        blk["per_training_seed_on_common_tokens"][str(s)] = Q.readout_block(
            us, logs, {c: None for c in CELLS}, n_boot, boot_seed)
    return blk


def paired_delta_3seed(seeds: Sequence[int], treat: str, t2l: Dict[str, str],
                       n_boot: int, boot_seed: int) -> Dict:
    """Δ(treat − G2) 的三 seed 配对读数：直接调冻结实现 PUD.paired_cell（seed-matched）。"""
    base_dirs = [SEED_DIR[s] / "G2" for s in seeds]
    treat_dirs = [SEED_DIR[s] / treat for s in seeds]
    res = PUD.paired_cell(base_dirs, treat_dirs, t2l, n_boot, boot_seed)
    # PUD 的 pair 名只有 "G3−G2"（cell 目录名不含 seed），补上 training seed 标签
    for s, ps in zip(seeds, res["per_training_seed"]):
        ps["training_seed"] = s
    return res


def sign_table(per_seed: Dict[int, Dict], d3: Dict[str, Dict]) -> Dict:
    """三 training seed 上各主读数的符号一致性（只报，不改判据）。"""
    seeds = sorted(per_seed)
    rows = {}
    rows["G3_over_G2_gt_1"] = {s: bool(per_seed[s]["pooled"]["q74_2_share_G3_over_G2"]["ratio_of_medians_pt"] > 1) for s in seeds}
    rows["G3_over_G2_ci_excludes_1"] = {
        s: bool(min(per_seed[s]["pooled"]["q74_2_share_G3_over_G2"]["ratio_of_medians_ci95"]) > 1) for s in seeds}
    for k in ("G3", "G4"):
        rows[f"delta_{k}_minus_G2_positive"] = {
            ps["training_seed"]: bool(ps["paired_delta_median"] > 0) for ps in d3[k]["per_training_seed"]}
        rows[f"delta_{k}_minus_G2_excludes_zero"] = {
            ps["training_seed"]: bool(ps["excludes_zero"]) for ps in d3[k]["per_training_seed"]}
    rows["q74_3_d_negative(report_only)"] = {
        s: bool(per_seed[s]["pooled"]["q74_3_additivity_report_only"]["median_token_G4_minus_G2plusG3"] < 0) for s in seeds}
    return {name: {"per_seed": {str(s): v for s, v in vals.items()},
                   "consistent": len(set(vals.values())) == 1} for name, vals in rows.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Q74 2×2 三 training seed 合并读数（判据 §3 不改）")
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--n_boot", type=int, default=PUD.N_BOOT_DEFAULT)
    p.add_argument("--boot_seed", type=int, default=PUD.BOOT_SEED_DEFAULT)
    p.add_argument("--out", type=Path, default=AUDIT / "q74_readout_3seed_20260817.json")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    t2l = {e["token"]: e["log_name"] for e in json.loads(LOG_MANIFEST.read_text())}

    per_seed: Dict[int, Dict] = {}
    for s in seeds:
        per_seed[s] = single_seed_readout(SEED_DIR[s], t2l, a.n_boot, a.boot_seed)
        per_seed[s]["crosscheck_paired_files"] = crosscheck_existing(SEED_DIR[s], per_seed[s]["pooled"])

    # 回归门：seed42 与既有 08-12 读数逐位一致
    regress = {}
    if 42 in per_seed:
        ref_f = SEED_DIR[42] / "q74_readout_20260812.json"
        ref = json.loads(ref_f.read_text())["pooled"]
        new = per_seed[42]["pooled"]
        diffs = []
        for cell in CELLS:
            for key in ("median_of_ratios", "ratio_of_medians"):
                diffs.append(abs(ref[f"usage_{cell}"][key] - new[f"usage_{cell}"][key]))
            diffs.append(max(abs(x - y) for x, y in zip(ref[f"usage_{cell}"]["ci95_logcluster"], new[f"usage_{cell}"]["ci95_logcluster"])))
        diffs.append(abs(ref["q74_2_share_G3_over_G2"]["ratio_of_medians_pt"] - new["q74_2_share_G3_over_G2"]["ratio_of_medians_pt"]))
        mx = max(diffs)
        regress = {"ref": str(ref_f), "max_abs_diff": mx, "n_paired": (ref["n_paired"], new["n_paired"]),
                   "pass": bool(mx < 1e-12 and ref["n_paired"] == new["n_paired"])}
        if not regress["pass"]:
            raise ValueError(f"🔴 回归门失败：seed42 重放与 08-12 读数不一致 {regress}")

    result: Dict = {
        "suite": "k0", "training_seeds": seeds, "n_boot": a.n_boot, "boot_seed": a.boot_seed,
        "criteria": "预注册判据/Q74_2x2通路分解_预注册判据_20260810_冻结.md §3（一字未改）",
        "convention": ("四层聚合逐字复用 paired_usage_delta.py：audit seed→token 中位 → token 两侧非塌陷配对 → "
                       "training seed→token 中位（seed-matched，不堆 token）→ token 中位 + log-cluster bootstrap；"
                       "🔴 三标注：旧档 L1 低密度档 + batch=1 CPU + k0 fix=True；G2/G3 通路级不得与 PlanTF 13.80% 并列，G4 模型级可；"
                       "3 点不做区间宣称，seed 间符号不一致只写「pooled X，seed 间不一致」"),
        "regression_seed42_vs_20260812": regress,
        "per_training_seed": {str(s): {k: v for k, v in per_seed[s].items() if not k.startswith("_")} for s in seeds},
    }
    if len(seeds) >= 2:
        result["pooled_3seed"] = pooled_over_training_seeds(per_seed, t2l, a.n_boot, a.boot_seed)
        d3 = {t: paired_delta_3seed(seeds, t, t2l, a.n_boot, a.boot_seed) for t in ("G3", "G4")}
        result["paired_delta_3seed"] = {f"{t}_minus_G2": d3[t] for t in d3}
        result["sign_consistency"] = sign_table(per_seed, d3)

    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    # 打印
    print("=" * 78)
    print(f"Q74 2×2 k0 套 training seed {seeds}  回归门 seed42: {regress.get('pass')}")
    for s in seeds:
        b = per_seed[s]["pooled"]
        cells = "  ".join(f"{c}={b[f'usage_{c}']['median_of_ratios']*100:.4f}%"
                          f"[{b[f'usage_{c}']['ci95_logcluster'][0]*100:.3f},{b[f'usage_{c}']['ci95_logcluster'][1]*100:.3f}]" for c in CELLS)
        r = b["q74_2_share_G3_over_G2"]
        q3 = b["q74_3_additivity_report_only"]
        print(f"  seed{s:>2} (n={b['n_paired']}/{b['n_logs']} log; 塌陷 {per_seed[s]['n_collapsed_head']}): {cells}")
        print(f"          G3/G2={r['ratio_of_medians_pt']:.3f}×[{r['ratio_of_medians_ci95'][0]:.3f},{r['ratio_of_medians_ci95'][1]:.3f}]"
              f"  Q74-3 d={q3['median_token_G4_minus_G2plusG3']*100:+.4f} pt[{q3['ci95_logcluster'][0]*100:+.3f},{q3['ci95_logcluster'][1]*100:+.3f}]")
    if "pooled_3seed" in result:
        b = result["pooled_3seed"]
        cells = "  ".join(f"{c}={b[f'usage_{c}']['median_of_ratios']*100:.4f}%"
                          f"[{b[f'usage_{c}']['ci95_logcluster'][0]*100:.3f},{b[f'usage_{c}']['ci95_logcluster'][1]*100:.3f}]" for c in CELLS)
        r = b["q74_2_share_G3_over_G2"]; q3 = b["q74_3_additivity_report_only"]
        print(f"  —— 公共 token 集（n={b['n_paired']}）上的 per-seed 水位（像对像）：")
        for s in seeds:
            bb = b["per_training_seed_on_common_tokens"][str(s)]
            print(f"     seed{s:>2}: " + "  ".join(f"{c}={bb[f'usage_{c}']['median_of_ratios']*100:.4f}%" for c in CELLS)
                  + f"  G3/G2={bb['q74_2_share_G3_over_G2']['ratio_of_medians_pt']:.3f}×")
        print(f"  POOLED({len(seeds)} seed, n={b['n_paired']}/{b['n_logs']} log): {cells}")
        print(f"          G3/G2={r['ratio_of_medians_pt']:.3f}×[{r['ratio_of_medians_ci95'][0]:.3f},{r['ratio_of_medians_ci95'][1]:.3f}]"
              f"  逐token比 {r['median_token_ratio_pt']:.3f}×[{r['median_token_ratio_ci95'][0]:.3f},{r['median_token_ratio_ci95'][1]:.3f}]"
              f"  Q74-3 d={q3['median_token_G4_minus_G2plusG3']*100:+.4f} pt[{q3['ci95_logcluster'][0]*100:+.3f},{q3['ci95_logcluster'][1]*100:+.3f}]")
        for t in ("G3", "G4"):
            d = result["paired_delta_3seed"][f"{t}_minus_G2"]["median_of_ratios"]
            ps = result["paired_delta_3seed"][f"{t}_minus_G2"]["per_training_seed"]
            print(f"  Δ({t}−G2) pooled {d['paired_delta_median']*100:+.4f} pt CI[{d['paired_delta_ci95_logcluster'][0]*100:+.3f},{d['paired_delta_ci95_logcluster'][1]*100:+.3f}]"
                  f" 排0={d['excludes_zero']}  per-seed: " + ", ".join(f"s{x['training_seed']} {x['paired_delta_median']*100:+.4f}[{'排0' if x['excludes_zero'] else '含0'}]" for x in ps)
                  + f"  符号一致={result['paired_delta_3seed'][f'{t}_minus_G2']['sign_consistent_across_seeds']}")
        print("  符号一致性表: " + "; ".join(f"{k}={'✅' if v['consistent'] else '🔴不一致'}" for k, v in result["sign_consistency"].items()))
    print(f"已写 {a.out}")


if __name__ == "__main__":
    main()
