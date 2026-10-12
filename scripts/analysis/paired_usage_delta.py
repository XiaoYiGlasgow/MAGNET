#!/usr/bin/env python3
"""跨 config 的**配对使用度差 ΔS** —— R1 主判据的落盘实现（第八十八条）。

背景：`usage_paired_20260804.json` 里已经有 `paired_delta_ci95_logcluster` 这个键，
但**全项目 grep 不到产生它的代码** —— 算法只存在于一次性的临时代码里。
⇒ R1 主判据在本脚本落盘之前**不可复现**。本脚本把口径定死并提供逐位回归门。

---

## 口径（冻结，见任何 R1 数字之前写死）

**度量**：使用度 ``U = mean_disp / base_scale``，干预 = ``count_sat.remove_agent``
（删光全部 agent）。与 §8「使用度的口径」逐字一致，**不得换口径**。

**四层聚合，顺序不可交换**：

1. **audit seed → token**：同 token 内对 audit seed（1000/1001/1002）取**中位数**。
   audit seed 只是「随机挑中哪几个实体」的随机性，不是独立样本。
2. **token 配对**：只保留在 treat 与 base **两侧都存在**、且**两侧都非塌陷**的 token。
   🔴 塌陷场景任何干预 response 恒为 0，混进来会把 estimand 稀释成假的零效应；
   **必须两侧都非塌陷** —— 只按一侧过滤会让两格的 token 集不同，配对就不成立了。
3. **training seed → token**（多 seed 时）：seed-matched 配对（treat_seedX ↔ base_seedX），
   同 token 内对 training seed 取**中位数**。
   🔴 **不把不同 training seed 的 token 堆在一起** —— 那会让 n 虚增 3 倍
   （第六十八条：有效 n 是簇数，不是观测数）。
4. **token → 读数**：点估计取**中位数**（分布右偏），区间用**按 log 聚类的 bootstrap**
   （第七十七条：分析单位 = log，不是 token）。

**主判据**：`paired_delta_median` 的 log-cluster 95% CI **排除 0 且方向为升**。
**次判据**：per-training-seed 的 δ **符号是否一致**（三 seed 全同号才算稳健）。

🔴 **同时报 `ratio_of_medians`**：`median(mean_disp) / median(base_scale)`。
它与主口径 `median_of_ratios` 是同一个相对量的**两种合法算法**，历史上「21×」
就是两个数字各选了不同的一组自由度造出来的（第八十三条）。**两个都报，不许挑一个用。**

🔴 **通路级限制（阶段85 已查实，写作时必须同报）**：`run_interaction_audit.py:202/209`
的 `_clone_with` 只覆盖 `gnn_x`/`gnn_edge_index`/`gnn_edge_attr`，`data.x` 原封不动
⇒ 本脚本测的是 **GNN 通路级**使用度，**不是模型级**，
**不得**与 PlanTF 的 13.80%（单通路、全模型切除）并列。

---

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    # 回归门：逐位复现 usage_paired_20260804.json
    $PY scripts/analysis/paired_usage_delta.py --regress

    # R1 vs L1 基线（三 seed，seed-matched）
    $PY scripts/analysis/paired_usage_delta.py \
        --audit_dir experiments/interaction_audit \
        --treat r1_seed42,r1_seed1,r1_seed2 \
        --base   l1_seed42,l1_seed1,l1_seed2 \
        --log_manifest experiments/scaling_ladder/s4_log_manifest_from_s1.json \
        --out experiments/interaction_audit/r1_paired_usage.json

🔴 **L1 侧的 log_manifest 只有一个正确答案**（08-05 实测）：
`experiments/scaling_ladder/s4_log_manifest_from_s1.json`（227 344 条，覆盖 L1 val 2400/2400，
映射出的 24 个 log 与 `split_manifest.json` 的 `val_logs` 完全一致）。
❌ `data/processed_features_scaling_L1_logsplit/` **没有** `log_manifest.json`；
❌ `data/processed_features_scaling_L3/log_manifest.json` 只有 100 条，覆盖 L1 val **0/2400**。
给错了不会报错，只会让每个 token 各自成簇 ⇒ CI 偏窄 ⇒ 主判据假阳性
（本脚本已把这条变成**当场停**，见 `--allow_unmapped`）。

⚠️ **L1 val 只有 24 个 log 簇**（08-01 那批审计是 59 个）⇒ cluster bootstrap 的有效 n
少一半以上（第六十八条：有效 n 是簇数不是观测数）。**判读 R1 的 CI 宽度时必须记住这一点。**
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("paired_usage")

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")

# 冻结口径：干预格与度量（换任何一个都不再是 §8 的「使用度」）
DOSE, OP_SIDE = "count_sat", "remove_agent"

#: 🆕 08-25：多模态模型需要**选口径**。旧 audit 的 shard 只有 `mean_disp`/`base_scale`
#: （单输出模型，三口径恒等）⇒ **默认 `argmax` 与历史行为逐位相同、回归门不受影响**。
#: `modes` = 逐模态自比后平均（**完全不依赖选模**），是 §6.2 预注册的**主口径**
#: —— 因为 joint 模型的干预下**模态翻转率 53–57%**，argmax 口径有一半「响应」是选模抖动，
#: 而 legacy 是 M=1（翻转率恒 0、三口径恒等）⇒ 只有 `modes` 在两侧是同一个 estimand。
METRIC_FIELDS = {
    "argmax": ("mean_disp", "base_scale"),
    "modes": ("mean_disp_modes", "base_scale_modes"),
    "fixed": ("mean_disp_fixed", "base_scale"),
}
_NUM, _DEN = METRIC_FIELDS["argmax"]          # 模块级默认；`main()` 按 --metric 改写
N_BOOT_DEFAULT, BOOT_SEED_DEFAULT = 10000, 20260731


def load_records(config_dir: Path) -> List[Dict]:
    """读一个 config 目录下全部 shard；缺 shard 只警告不静默丢。"""
    shards = sorted(glob.glob(str(config_dir / "shard_*.json")))
    if not shards:
        raise FileNotFoundError(f"{config_dir} 下没有 shard_*.json")
    recs: List[Dict] = []
    for s in shards:
        with open(s, encoding="utf-8") as f:
            recs.extend(json.load(f)["records"])
    logger.info("%s: %d shard，%d 条记录", config_dir.name, len(shards), len(recs))
    return recs


def per_token(records: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """第 1 层聚合：audit seed → token（取中位数），并带出塌陷标记。

    `mean_disp` / `base_scale` 各自单独取中位数（供 `ratio_of_medians` 用），
    比值 `ratio` 也在 seed 层先算再取中位数（供主口径 `median_of_ratios` 用）。
    """
    buf: Dict[str, Dict[str, List[float]]] = {}
    collapsed: Dict[str, bool] = {}
    for r in records:
        dd = r.get("doses", {}).get(DOSE)
        if not isinstance(dd, dict) or "skipped" in dd:
            continue
        cell = dd.get(OP_SIDE)
        if not isinstance(cell, dict) or _NUM not in cell:
            continue
        bs = cell.get(_DEN)
        if not bs:                      # None / 0 —— 比值无定义，不能静默当 0
            continue
        tok = r["token"]
        b = buf.setdefault(tok, {"ratio": [], "mean_disp": [], "base_scale": []})
        b["ratio"].append(float(cell[_NUM]) / float(bs))
        b["mean_disp"].append(float(cell[_NUM]))
        b["base_scale"].append(float(bs))
        collapsed[tok] = collapsed.get(tok, False) or bool(r.get("collapsed_head"))
    return {
        tok: {"ratio": float(np.median(v["ratio"])),
              "mean_disp": float(np.median(v["mean_disp"])),
              "base_scale": float(np.median(v["base_scale"])),
              "collapsed": collapsed[tok],
              "n_audit_seeds": len(v["ratio"])}
        for tok, v in buf.items()
    }


def cluster_bootstrap_ci(values: np.ndarray, clusters: np.ndarray,
                         n_boot: int, boot_seed: int) -> Tuple[float, float]:
    """按 cluster（log）重抽的 bootstrap 95% CI —— 同一 log 内的场景不是独立样本。"""
    if values.size == 0:
        return float("nan"), float("nan")
    uniq = np.unique(clusters)
    if uniq.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(boot_seed)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        stats[b] = np.median(values[np.concatenate([idx[c] for c in pick])])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _seed_of(name: str) -> Optional[str]:
    """从 config 目录名里抠出 training seed（`..._seed42` → `42`）。抠不出返回 None。"""
    m = re.search(r"seed(\d+)", name)
    return m.group(1) if m else None


def paired_cell(base_dirs: Sequence[Path], treat_dirs: Sequence[Path],
                token_to_log: Dict[str, str], n_boot: int, boot_seed: int,
                allow_unmapped: bool = False) -> Dict:
    """完整读数：seed-matched 配对 → 两个口径的点估计 + CI + per-seed 符号一致性。

    :param base_dirs: 对照格的 config 目录（按 training seed 顺序）。
    :param treat_dirs: 处理格的 config 目录，**必须与 base_dirs 逐位 seed 对齐**。
    :param allow_unmapped: 允许 token 不在 log_manifest 里。🔴 默认 False，见下方硬门。
    """
    if len(base_dirs) != len(treat_dirs):
        raise ValueError(f"seed-matched 配对要求两侧 config 数相同："
                         f"base {len(base_dirs)} vs treat {len(treat_dirs)}")

    # 🔴 静默类防线：目录名里的 seed 必须逐位对上。写反顺序（如
    # `--treat r1_seed42,r1_seed1 --base l1_seed1,l1_seed42`）会静默做成跨 seed 配对，
    # 产出一个看着正常、其实混了训练随机性的 ΔS。
    for bd, td in zip(base_dirs, treat_dirs):
        sb, st = _seed_of(bd.name), _seed_of(td.name)
        if sb is not None and st is not None and sb != st:
            raise ValueError(
                f"🔴 seed 未对齐：base={bd.name}(seed{sb}) vs treat={td.name}(seed{st})。"
                "seed-matched 配对要求两侧逐位同 seed —— 请检查 --base/--treat 的顺序。")

    # 每对 (base, treat) 各自做第 1–2 层聚合
    per_pair: List[Dict[str, Dict[str, float]]] = []
    pair_names: List[str] = []
    for bd, td in zip(base_dirs, treat_dirs):
        B, T = per_token(load_records(bd)), per_token(load_records(td))
        # 第 2 层：两侧都存在 + 两侧都非塌陷
        keys = sorted(k for k in set(B) & set(T)
                      if not B[k]["collapsed"] and not T[k]["collapsed"])
        per_pair.append({k: {"base": B[k], "treat": T[k]} for k in keys})
        pair_names.append(f"{td.name}−{bd.name}")
        logger.info("%s: 配对 token %d（base 侧 %d / treat 侧 %d）",
                    pair_names[-1], len(keys), len(B), len(T))

    # 第 3 层：training seed → token（只保留所有 seed 都有的 token，否则配对不齐）
    common = sorted(set.intersection(*(set(p) for p in per_pair)))
    if not common:
        raise ValueError("所有 seed 的公共配对 token 为空 ⇒ 无法配对，拒绝出数")

    delta = np.array([float(np.median([p[k]["treat"]["ratio"] - p[k]["base"]["ratio"]
                                       for p in per_pair])) for k in common])
    base_u = np.array([float(np.median([p[k]["base"]["ratio"] for p in per_pair])) for k in common])
    treat_u = np.array([float(np.median([p[k]["treat"]["ratio"] for p in per_pair])) for k in common])
    logs = np.array([token_to_log.get(k, f"__unknown_{k}") for k in common])
    n_unknown = int(sum(1 for k in common if k not in token_to_log))
    # 🔴🔴 静默类防线（本脚本最危险的一条）：token 查不到 log 时会**各自成一簇**，
    # cluster bootstrap 退化成 token 级 bootstrap ⇒ CI 大幅偏窄 ⇒ 可能假阳性地
    # 「排除 0」，而输出看上去完全正常。给错 manifest（比如把 e1_none_v3 的清单用到
    # L1 logsplit 的 token 上）就是这个形态 ⇒ **默认当场停，不降级**。
    if n_unknown and not allow_unmapped:
        raise ValueError(
            f"🔴 {n_unknown}/{len(common)} 个 token 不在 log_manifest 里。"
            "它们会各自成簇 ⇒ CI 偏窄 ⇒ 主判据可能假阳性。"
            "请确认 --log_manifest 与审计所用 --raw_dir 是同一份数据；"
            "确实要带着跑请显式加 --allow_unmapped。")
    if n_unknown:
        logger.warning("🔴 %d/%d 个 token 不在 log_manifest 里 ⇒ 各自成簇，CI 会偏窄"
                       "（已由 --allow_unmapped 放行）", n_unknown, len(common))

    d_lo, d_hi = cluster_bootstrap_ci(delta, logs, n_boot, boot_seed)
    b_lo, b_hi = cluster_bootstrap_ci(base_u, logs, n_boot, boot_seed)
    t_lo, t_hi = cluster_bootstrap_ci(treat_u, logs, n_boot, boot_seed)

    # ratio_of_medians：同一相对量的第二种合法算法（第八十三条：两个都报）
    def _rom(side: str) -> float:
        md = float(np.median([np.median([p[k][side]["mean_disp"] for p in per_pair]) for k in common]))
        bs = float(np.median([np.median([p[k][side]["base_scale"] for p in per_pair]) for k in common]))
        return md / bs if bs else float("nan")

    # per-seed δ（次判据：符号一致性）
    per_seed = []
    for name, p in zip(pair_names, per_pair):
        ks = [k for k in common if k in p]
        d = np.array([p[k]["treat"]["ratio"] - p[k]["base"]["ratio"] for k in ks])
        lg = np.array([token_to_log.get(k, f"__unknown_{k}") for k in ks])
        lo, hi = cluster_bootstrap_ci(d, lg, n_boot, boot_seed)
        per_seed.append({"pair": name, "n_paired": len(ks),
                         "paired_delta_median": float(np.median(d)),
                         "paired_delta_ci95_logcluster": [lo, hi],
                         "excludes_zero": bool(lo > 0 or hi < 0)})

    signs = {np.sign(s["paired_delta_median"]) for s in per_seed}
    return {
        "n_paired": len(common),
        "n_logs": int(np.unique(logs).size),
        "n_tokens_not_in_manifest": n_unknown,
        "n_training_seeds": len(per_pair),
        "median_of_ratios": {
            "base": float(np.median(base_u)), "base_ci95": [b_lo, b_hi],
            "treat": float(np.median(treat_u)), "treat_ci95": [t_lo, t_hi],
            "paired_delta_median": float(np.median(delta)),
            "paired_delta_ci95_logcluster": [d_lo, d_hi],
            "excludes_zero": bool(d_lo > 0 or d_hi < 0),
            "direction": "up" if float(np.median(delta)) > 0 else "down",
        },
        "ratio_of_medians": {"base": _rom("base"), "treat": _rom("treat")},
        "per_training_seed": per_seed,
        "sign_consistent_across_seeds": len(signs) == 1,
        "n_delta_positive": int((delta > 0).sum()),
        "n_delta_negative": int((delta < 0).sum()),
        "n_delta_zero": int((delta == 0).sum()),
    }


# --------------------------------------------------------------------------- #
# 回归门：逐位复现 usage_paired_20260804.json（该 json 由无脚本的临时代码产出）
# --------------------------------------------------------------------------- #
REGRESS_REF = GNN / "experiments" / "interaction_audit" / "usage_paired_20260804.json"
REGRESS_CASES = {
    "gate1": ("B_seed42_raw", "gate1solo_seed42"),
    "c2": ("B_seed42_raw", "c2_ego_query_seed42"),
}
REGRESS_MANIFEST = GNN / "data" / "processed_features_magnet_e1_none_v3" / "log_manifest.json"


def run_regression() -> bool:
    if not REGRESS_REF.exists():
        logger.error("回归门：%s 不存在", REGRESS_REF)
        return False
    ref = json.loads(REGRESS_REF.read_text())
    with open(REGRESS_MANIFEST, encoding="utf-8") as f:
        t2l = {e["token"]: e["log_name"] for e in json.load(f)}
    audit = GNN / "experiments" / "interaction_audit"
    ok = True
    print("\n=== 回归门（重算 vs 已落盘 usage_paired_20260804.json）===")
    for case, (base, treat) in REGRESS_CASES.items():
        got = paired_cell([audit / base], [audit / treat], t2l,
                          N_BOOT_DEFAULT, BOOT_SEED_DEFAULT)
        r = ref[case]
        checks: List[Tuple[str, float, float]] = [
            ("n_paired", got["n_paired"], r["n_paired"]),
            ("n_logs", got["n_logs"], r["n_logs"]),
            ("mor.base", got["median_of_ratios"]["base"], r["median_of_ratios"]["base"]),
            ("mor.treat", got["median_of_ratios"]["treat"], r["median_of_ratios"]["treat"]),
            ("mor.base_ci_lo", got["median_of_ratios"]["base_ci95"][0], r["median_of_ratios"]["base_ci95"][0]),
            ("mor.base_ci_hi", got["median_of_ratios"]["base_ci95"][1], r["median_of_ratios"]["base_ci95"][1]),
            ("mor.treat_ci_lo", got["median_of_ratios"]["treat_ci95"][0], r["median_of_ratios"]["treat_ci95"][0]),
            ("mor.treat_ci_hi", got["median_of_ratios"]["treat_ci95"][1], r["median_of_ratios"]["treat_ci95"][1]),
            ("paired_delta", got["median_of_ratios"]["paired_delta_median"], r["median_of_ratios"]["paired_delta_median"]),
            ("paired_ci_lo", got["median_of_ratios"]["paired_delta_ci95_logcluster"][0], r["median_of_ratios"]["paired_delta_ci95_logcluster"][0]),
            ("paired_ci_hi", got["median_of_ratios"]["paired_delta_ci95_logcluster"][1], r["median_of_ratios"]["paired_delta_ci95_logcluster"][1]),
            ("rom.base", got["ratio_of_medians"]["base"], r["ratio_of_medians"]["base"]),
            ("rom.treat", got["ratio_of_medians"]["treat"], r["ratio_of_medians"]["treat"]),
        ]
        print(f"\n  --- {case} ({treat} − {base}) ---")
        for name, a, b in checks:
            hit = abs(float(a) - float(b)) < 1e-12
            ok &= hit
            print(f"    {'✅' if hit else '❌'} {name:16s} 重算 {a!r:24s} vs 文档 {b!r}")
        hit = got["median_of_ratios"]["excludes_zero"] == r["median_of_ratios"]["excludes_zero"]
        ok &= hit
        print(f"    {'✅' if hit else '❌'} excludes_zero    重算 "
              f"{got['median_of_ratios']['excludes_zero']} vs 文档 {r['median_of_ratios']['excludes_zero']}")
    print(f"\n  回归门：{'✅ 全过 —— 算法已落盘，R1 主判据可复现' if ok else '🔴 有不一致'}")
    return ok


def print_report(res: Dict, label: str) -> None:
    mor = res["median_of_ratios"]
    print("\n" + "=" * 78)
    print(f"配对使用度差 ΔS —— {label}")
    print("=" * 78)
    print(f"  配对 token {res['n_paired']}｜log 簇 {res['n_logs']}｜"
          f"training seed {res['n_training_seeds']}")
    print(f"\n  🎯 主口径 median_of_ratios（U = mean_disp / base_scale）")
    print(f"    base  {mor['base']*100:.4f}%  CI[{mor['base_ci95'][0]*100:.4f}%,"
          f"{mor['base_ci95'][1]*100:.4f}%]")
    print(f"    treat {mor['treat']*100:.4f}%  CI[{mor['treat_ci95'][0]*100:.4f}%,"
          f"{mor['treat_ci95'][1]*100:.4f}%]")
    print(f"    🔴 配对 ΔS = {mor['paired_delta_median']*100:+.4f} pt  "
          f"CI[{mor['paired_delta_ci95_logcluster'][0]*100:+.4f},"
          f"{mor['paired_delta_ci95_logcluster'][1]*100:+.4f}] pt")
    print(f"    ⇒ CI {'排除 0' if mor['excludes_zero'] else '含 0'}，方向 {mor['direction']}")
    rom = res["ratio_of_medians"]
    print(f"\n  第二种合法算法 ratio_of_medians（第八十三条：两个都报，不许挑）")
    print(f"    base {rom['base']*100:.4f}%   treat {rom['treat']*100:.4f}%")
    print(f"\n  逐 token 方向：+{res['n_delta_positive']} / −{res['n_delta_negative']} "
          f"/ 0 {res['n_delta_zero']}")
    if res["n_training_seeds"] > 1:
        print(f"\n  次判据 —— per training seed δ（符号"
              f"{'一致 ✅' if res['sign_consistent_across_seeds'] else '不一致 ⚠️'}）")
        for s in res["per_training_seed"]:
            ci = s["paired_delta_ci95_logcluster"]
            print(f"    {s['pair']:32s} δ={s['paired_delta_median']*100:+.4f} pt "
                  f"CI[{ci[0]*100:+.4f},{ci[1]*100:+.4f}] "
                  f"{'排除0' if s['excludes_zero'] else '含0'} (n={s['n_paired']})")
    # 🔴 干预覆盖范围**随 shard 而定**，不能写死（08-25：`joint_usage_probe.py` 产的是模型级）。
    #    旧 audit 的 shard 里没有 `mean_disp_modes` ⇒ 用它作判别，比按目录名猜可靠。
    _model_level = any(
        isinstance(r.get("doses", {}).get(DOSE, {}).get(OP_SIDE), dict)
        and "mean_disp_modes" in r["doses"][DOSE][OP_SIDE]
        for d in bd + td for r in (load_records(d) if d.exists() else []))
    if _model_level:
        print("\n  🔴 **模型级**读数（真移除节点 + 全部边 + 其向量行）⇒ 对照基准是 **G4 = 6.8697%**，"
              "\n     **不是** 0.72%（那是 GNN 通路级 G2）；亦不得与 PlanTF 13.80% 直接并列（不同模型族）。")
    else:
        print("\n  🔴 通路级限制：本读数只切 GNN 通路（`data.x` 未动）⇒ 不是模型级依赖，"
              "\n     不得与 PlanTF 的 13.80%（全模型切除）并列。")


def main() -> None:
    p = argparse.ArgumentParser(description="跨 config 配对使用度差 ΔS（R1 主判据）")
    p.add_argument("--regress", action="store_true", help="只跑回归门")
    p.add_argument("--audit_dir", type=Path, default=GNN / "experiments" / "interaction_audit")
    p.add_argument("--base", type=str, help="对照格 config 目录名，逗号分隔（按 seed 顺序）")
    p.add_argument("--treat", type=str, help="处理格 config 目录名，与 --base 逐位 seed 对齐")
    p.add_argument("--log_manifest", type=Path, help="token→log 映射（cluster bootstrap 用）")
    p.add_argument("--n_boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--boot_seed", type=int, default=BOOT_SEED_DEFAULT)
    p.add_argument("--allow_unmapped", action="store_true",
                   help="🔴 允许 token 不在 log_manifest 里（会各自成簇、CI 偏窄）")
    p.add_argument("--label", type=str, default="treat − base")
    p.add_argument("--out", type=Path)
    p.add_argument("--metric", choices=tuple(METRIC_FIELDS), default="argmax",
                   help="🔴 多模态模型必须显式选：argmax=部署口径（含模态翻转）｜"
                        "modes=**§6.2 预注册主口径**（逐模态自比，不依赖选模）｜fixed=沿用 base 模态。"
                        "默认 argmax ⇒ 与历史行为逐位相同（旧 shard 只有该字段）")
    a = p.parse_args()

    if a.regress:
        raise SystemExit(0 if run_regression() else 1)
    # 🔴 按 --metric 切换读哪两个字段。回归门在上面已 return ⇒ **回归门恒用 argmax，不受影响**。
    global _NUM, _DEN
    _NUM, _DEN = METRIC_FIELDS[a.metric]
    logger.info("度量口径 = %s（分子 %s / 分母 %s）", a.metric, _NUM, _DEN)
    if not (a.base and a.treat and a.log_manifest):
        raise SystemExit("需要 --base / --treat / --log_manifest（或用 --regress）")

    with open(a.log_manifest, encoding="utf-8") as f:
        t2l = {e["token"]: e["log_name"] for e in json.load(f)}
    bd = [a.audit_dir / c.strip() for c in a.base.split(",") if c.strip()]
    td = [a.audit_dir / c.strip() for c in a.treat.split(",") if c.strip()]

    res = paired_cell(bd, td, t2l, a.n_boot, a.boot_seed, a.allow_unmapped)
    res["metric"] = a.metric
    res["metric_fields"] = {"numerator": _NUM, "denominator": _DEN}
    res["convention"] = (
        f"U = {_NUM}/{_DEN}（--metric {a.metric}），干预 count_sat.remove_agent；聚合顺序 "
        "audit-seed→token(median) → 两侧都非塌陷的配对 token → training-seed(median) → "
        "token(median) + log-cluster bootstrap；主判据 = 配对 ΔS 的 CI 排除 0 且为升。"
        "🔴 **干预覆盖范围随 shard 而定**：旧 `interaction_audit` 的 shard 是**通路级**"
        "（`_clone_with` 不动 `data.x`）⇒ 不得与 PlanTF 13.80% 并列；"
        "`joint_usage_probe.py` 产的 shard 是**模型级**（真移除节点+边+向量行）⇒ 对照基准是 G4=6.8697%")
    res["base_configs"] = [d.name for d in bd]
    res["treat_configs"] = [d.name for d in td]
    res["n_boot"], res["boot_seed"] = a.n_boot, a.boot_seed

    print_report(res, a.label)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n已写 {a.out}")


if __name__ == "__main__":
    main()
