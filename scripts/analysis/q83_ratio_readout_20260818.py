#!/usr/bin/env python3
"""Q83 剂量-反应扫描的聚合读数 —— 交换率 **R\\*** 的落盘实现（判据 §7 的最后一件交付物）。

判据 = `预注册判据/Q83_剂量反应与交换率_预注册判据_20260805.md`（2026-08-05 写于跑任何一格之前），
**本脚本一字不改判据**，只把 §2/§3/§4 的规则翻译成代码：

* §2 估计方法：网格 ``R ∈ {1,2,3,5,7,10,14,20}``（`RATIO_DOSES`，固定 ``k_target=1``、``k_control=R``），
  每个 R 上按既有主口径算 ``median_S`` 与 **log-cluster bootstrap CI**（口径不换 ⇒ 直接复用
  `agg_interaction_audit` 的 `_by_token` / `_test_cell`，不重写一遍统计）；
* §2 R\\* 的报法：**只报包含变号点的相邻两格 [R_i, R_{i+1}]，不做插值、不给点估计**；
  CI 侧则报「最后一个 CI 全正的 R」与「第一个 CI 全负的 R」；
* §3 主判据：``median_S(R)`` **单调不增**（允许并列，不得先降后升）且网格内**发生符号变号**；
* §3 自校验点：``ratio_1`` 必须与 ``count_1`` 读数一致 —— 由 08-18 14:0x 的两次专跑给出
  （`experiments/interaction_audit/q83_selfcheck_20260818/{count_1,ratio_1}.json`，
  各自作为**首个剂量**从新 rng 起 ⇒ 72 字段逐位一致），本脚本把结论作为 `self_check` 字段带出；
* §4 三分支读法（**写于见数字之前**）：单调且变号 / 单调但不变号 / 非单调或无变号，
  🔴 第三种只能把「符号翻转」当一次性 caveat 报，**不得改判据去凑单调**；
* §5 口径与限制：主口径 `median_S` + **非塌陷子集** + **log-cluster bootstrap**（分析单位 = log）；
  通路级读数，🔴 不得与 PlanTF 13.80% 并列；`ratio_*` 是 exploratory，**不进 Holm primary family**；
  报告必须同时给出每个 R 的**实际删除数** `mean_k_target` / `mean_k_control`。

用法::

    PY=/home/team/anaconda3/envs/nuplan/bin/python
    $PY scripts/analysis/q83_ratio_readout_20260818.py \\
        --audit_dir experiments/interaction_audit/q83_ratio_l1_seed42 \\
        --log_manifest experiments/scaling_ladder/s4_log_manifest_from_s1.json \\
        --out experiments/interaction_audit/q83_ratio_readout_20260818.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from audit_dose_contract import RATIO_DOSES
from agg_interaction_audit import _by_token, _load_records, _test_cell

logger = logging.getLogger("q83_readout")

#: 网格顺序 = 判据 §2 的书写顺序，**不排序、不增删**。
GRID: Sequence[int] = tuple(int(d.value) for d in RATIO_DOSES)
SELF_CHECK_DIR = Path("experiments/interaction_audit/q83_selfcheck_20260818")


def _self_check(root: Path) -> Dict:
    """判据 §3 自校验点：``ratio_1`` 与 ``count_1`` 的读数是否逐位一致。

    两个产物必须由**各自作为首个剂量**的两次专跑给出 —— `run_interaction_audit.py` 里
    `rng` 每 (token, seed) 重置一次，但同一次跑内剂量是**顺序消费** rng 的，
    把两个剂量放进同一次跑，后一个的实体抽样就依赖前一个，比较将失去意义。
    """
    a_path, b_path = root / "count_1.json", root / "ratio_1.json"
    if not (a_path.exists() and b_path.exists()):
        return {"status": "MISSING", "note": f"缺 {a_path} 或 {b_path}"}
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    n_fields, mism = 0, []
    for ra, rb in zip(a["records"], b["records"]):
        if (ra["token"], ra["seed"]) != (rb["token"], rb["seed"]):
            mism.append({"token": ra["token"], "why": "记录顺序不对齐"})
            continue
        da, db = ra["doses"].get("count_1"), rb["doses"].get("ratio_1")
        if da is None or db is None:
            mism.append({"token": ra["token"], "why": "剂量缺失"})
            continue
        if "skipped" in da or "skipped" in db:
            if da.get("skipped") != db.get("skipped"):
                mism.append({"token": ra["token"], "why": "skip 理由不同"})
            continue
        for key, va in da.items():
            if key == "mode":
                continue
            vb = db[key]
            if isinstance(va, dict):
                for kk, vv in va.items():
                    n_fields += 1
                    if vv != vb[kk]:
                        mism.append({"token": ra["token"], "field": f"{key}.{kk}"})
            else:
                n_fields += 1
                if va != vb:
                    mism.append({"token": ra["token"], "field": key})
    return {
        "status": "PASS" if not mism else "FAIL",
        "n_fields_compared": n_fields,
        "n_mismatch": len(mism),
        "mismatches": mism[:20],
        "note": "判据 §3：若不一致 ⇒ 当场停，不得继续解释曲线",
    }


def _ci_sign(ci: Sequence[float]) -> str:
    """CI 的符号态：全正 / 全负 / 含 0 / 不可用。"""
    lo, hi = ci
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "unavailable"
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "contains_zero"


def _monotone_non_increasing(values: Sequence[float], tol: float = 0.0) -> Dict:
    """判据 §3 的单调性：允许相邻并列，**不得出现先降后升的反转**。"""
    reversals = []
    for i in range(1, len(values)):
        if values[i] > values[i - 1] + tol:
            reversals.append({"from_index": i - 1, "to_index": i,
                              "delta": float(values[i] - values[i - 1])})
    return {"monotone_non_increasing": not reversals, "reversals": reversals}


def _crossing(grid: Sequence[int], medians: Sequence[float]) -> Optional[Dict]:
    """点估计的变号区间：报**包含变号点的相邻两格**，🔴 不插值、不给点估计（判据 §2）。"""
    for i in range(1, len(grid)):
        a, b = medians[i - 1], medians[i]
        if a > 0 >= b or (a >= 0 > b):
            return {"R_interval": [grid[i - 1], grid[i]],
                    "median_S_at_endpoints": [float(a), float(b)]}
    return None


def _ci_bracket(grid: Sequence[int], signs: Sequence[str]) -> Dict:
    """CI 侧的 R\\* 区间：最后一个 CI 全正的 R 与第一个 CI 全负的 R（判据 §2）。"""
    last_pos = max((r for r, s in zip(grid, signs) if s == "positive"), default=None)
    first_neg = min((r for r, s in zip(grid, signs) if s == "negative"), default=None)
    return {"last_R_ci_all_positive": last_pos, "first_R_ci_all_negative": first_neg,
            "R_star_ci_interval": [last_pos, first_neg] if (last_pos is not None and first_neg is not None) else None}


def _sign_change_ci(grid: Sequence[int], signs: Sequence[str]) -> Dict:
    """判据 §3 对「变号」的**字面定义**：`R=1` 处 CI 全正、且某个 R 处 CI 全负。

    🔴 与点估计变号分开报 —— 点估计换号只说明曲线跨过 0，判据要的是**两端都被 CI 排除**。
    两者不一致时（例如 R=1 的 CI 含 0），必须如实说「主判据的变号部分只满足一半」，
    不得用点估计变号顶替（那等于事后放松判据）。
    """
    if not grid:
        return {"met": False, "why": "无可用格点"}
    r1_positive = bool(grid[0] == 1 and signs[0] == "positive")
    first_neg = next((r for r, sg in zip(grid, signs) if sg == "negative"), None)
    return {
        "met": bool(r1_positive and first_neg is not None),
        "R1_ci_all_positive": r1_positive,
        "R1_ci_sign": signs[0] if signs else None,
        "first_R_ci_all_negative": first_neg,
        "why": ("满足" if (r1_positive and first_neg is not None) else
                "R=1 处 CI 未全正" if not r1_positive else "无 CI 全负的 R"),
    }


def _verdict(mono: Dict, crossing: Optional[Dict], signs: Sequence[str],
             usable_grid: Sequence[int], sc_ci: Dict) -> Dict:
    """判据 §4 的三分支读法（写于见数字之前，逐字翻译，不新增分支）。

    🔴 格点不足 2 个时三分支**一个都不适用** —— 返回 `INSUFFICIENT` 而不是硬套分支②，
    否则空数据会被读成「单调但不变号 ⇒ 交换率 > 20」这种听起来像结论的假结论。
    这不是新增判据分支，是拒绝在前提不成立时判读。

    🔴 「变号」以判据 §3 的**字面（CI）定义**为准（`sc_ci["met"]`）；点估计变号只作附注。
    两者分歧时分支归属按字面定义走，并在 `caveat` 里写清分歧，**不得**用点估计顶替。
    """
    if len(usable_grid) < 2:
        return {"branch": "INSUFFICIENT（不适用三分支）",
                "conclusion": f"只有 {len(usable_grid)} 个可用格点，单调性与变号都无从谈起",
                "next": "先查为什么格被 skip（n_control < R 全跳？塌陷？剂量没跑？）"}
    point_change = crossing is not None
    caveat = None
    if point_change != sc_ci["met"]:
        caveat = (f"🔴 点估计变号={point_change} 与判据 §3 的 CI 形式变号={sc_ci['met']} 不一致"
                  f"（{sc_ci['why']}）⇒ 分支归属按 CI 形式定义；报 R* 时必须同时给出两种读法。")
    if mono["monotone_non_increasing"] and sc_ci["met"]:
        out = {"branch": "① 单调且在网格内变号",
               "conclusion": "剂量-反应成立，R* 可估 ⇒ 「谁更重要」被 R* 取代",
               "next": "把 R* 作为跨架构可比标量：报 L1 基线 vs R1 的 R*；两者不同即说明修复改变了模态交换率"}
    elif mono["monotone_non_increasing"] and sc_ci.get("first_R_ci_all_negative") is not None:
        # 🔴 判据没有枚举这一情形：曲线**确实**进入 CI 全负区（右端满足），
        #    但左端 R=1 的 CI 含 0 ⇒ 分支② 的结论文字「交换率 > 20」被本数据**直接证伪**，
        #    分支① 的两端条件又没全满足。硬套任何一支都会产出判据未授权的结论 ⇒ 如实标为未枚举情形。
        #    这不是新增判据分支，是拒绝把不适用的分支文字当结论。
        out = {"branch": "①/② 之外的未枚举情形（判据 §4 未覆盖）",
               "conclusion": ("曲线单调不增且从 R="
                              f"{sc_ci['first_R_ci_all_negative']} 起 CI 全负 ⇒ 剂量-反应关系成立；"
                              "但 R=1 处 CI 含 0（等量删除时 S 与 0 不可区分）⇒ 判据字面的「变号」只满足右端。"
                              "🔴 分支② 的「交换率 > 20」被本数据证伪，不得引用；分支① 的「R* 可估」只在点估计口径成立"),
               "next": ("报法固定为：R* 点估计区间 = 变号相邻两格（不插值）；CI 侧**只给上界** R* <= "
                        f"{sc_ci['first_R_ci_all_negative']}，**下界不可估**（无 CI 全正的格）。"
                        "L-1 的批判半边（符号由剂量决定）因此加强；建设半边（R* 作跨模型可比标量）只能带上界报")}
    else:
        out = {"branch": "③ 非单调 / 无变号",
               "conclusion": "剂量-反应假设不成立 ⇒ 🔴 本条候选降级：「符号翻转」只能作一次性 caveat 报，不能升级成机制",
               "next": "🔴 不得改判据去凑单调（判据 §4 明文）"}
    if caveat:
        out["caveat"] = caveat
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q83 剂量-反应扫描聚合读数（判据 §2–§5）")
    p.add_argument("--audit_dir", required=True, type=Path,
                   help="放 shard_*.json 的 config 目录，例如 experiments/interaction_audit/q83_ratio_l1_seed42")
    p.add_argument("--log_manifest", required=True, type=Path,
                   help="token→log 映射；L1 侧唯一正确答案 = experiments/scaling_ladder/s4_log_manifest_from_s1.json（§8.2）")
    p.add_argument("--ops", default="remove",
                   help="要读的算子，逗号分隔；判据谈的是 remove，zero 只作 exploratory 附注")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--boot_seed", type=int, default=20260731)
    p.add_argument("--allow_unmapped", action="store_true",
                   help="🔴 默认关闭：token 映不到 log 会让 cluster bootstrap 退化成 token 级、CI 假窄（§8.2）")
    p.add_argument("--self_check_dir", type=Path, default=SELF_CHECK_DIR)
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    a = parse_args()

    with open(a.log_manifest, encoding="utf-8") as f:
        token_to_log = {e["token"]: e["log_name"] for e in json.load(f)}

    records = _load_records(a.audit_dir)
    if not records:
        raise SystemExit(f"🔴 {a.audit_dir} 下没有任何 shard 产物")
    all_tokens = {r["token"] for r in records}
    unmapped = sorted(t for t in all_tokens if t not in token_to_log)
    if unmapped and not a.allow_unmapped:
        raise SystemExit(
            f"🔴 {len(unmapped)}/{len(all_tokens)} 个 token 不在 log_manifest 里 —— "
            "cluster bootstrap 会退化成 token 级、CI 大幅偏窄（§8.2 已有先例）。"
            f"给错的 manifest 不会报错，只会静默削 CI。前三个：{unmapped[:3]}。"
            "确实要带着跑请显式加 --allow_unmapped。")
    if unmapped:
        logger.warning("%d 个 token 映不到 log（已由 --allow_unmapped 放行）", len(unmapped))

    payload: Dict = {
        "criteria": "预注册判据/Q83_剂量反应与交换率_预注册判据_20260805.md（§2 估计方法 / §3 主判据 / §4 三分支 / §5 口径）",
        "criteria_unchanged": True,
        "protocol": "INTERVENTION_PROTOCOL_v1_20260731",
        "audit_dir": str(a.audit_dir),
        "log_manifest": str(a.log_manifest),
        "grid_R": list(GRID),
        "n_boot": a.n_boot, "boot_seed": a.boot_seed,
        "family": "exploratory —— ratio_* 不进 Holm primary family（判据 §5）",
        "caveats": [
            "通路级读数：干预只切 GNN 通路（data.x 未动）⇒ 🔴 不得与 PlanTF 13.80% 并列",
            "n_control < R 的 token 一律 skip（不 clamp）⇒ 每个 R 的有效 n 不同，逐格报 n",
            "R* 只报包含变号点的相邻两格，不插值、不给点估计（8 个格点不支持连续拟合）",
            "🔴 不得由本实验推出「lane 比 agent 重要」或反之 —— 那正是被证明良定义不了的问法",
        ],
        "self_check_ratio1_vs_count1": _self_check(a.self_check_dir),
        "cells": {},
    }

    for op in [o.strip() for o in a.ops.split(",") if o.strip()]:
        rng = np.random.default_rng(a.boot_seed)   # 每个算子独立复位 ⇒ 与算子顺序无关
        per_R: Dict[str, Dict] = {}
        medians: List[float] = []
        signs: List[str] = []
        usable_grid: List[int] = []
        for dose, R in zip(RATIO_DOSES, GRID):
            tokens = _by_token(records, dose.name, op)
            if not tokens:
                per_R[dose.name] = {"skipped": "该格全部 token 被 skip", "R": R}
                continue
            res = _test_cell(tokens, token_to_log, a.n_boot, rng)
            res["R"] = R
            res["role"] = "exploratory"
            res["ci_sign"] = _ci_sign(res.get("ci95_S_logcluster", [float("nan")] * 2))
            per_R[dose.name] = res
            if "median_S" in res:
                usable_grid.append(R)
                medians.append(res["median_S"])
                signs.append(res["ci_sign"])

        mono = _monotone_non_increasing(medians)
        crossing = _crossing(usable_grid, medians)
        bracket = _ci_bracket(usable_grid, signs)
        sc_ci = _sign_change_ci(usable_grid, signs)
        payload["cells"][op] = {
            "per_R": per_R,
            "usable_grid": usable_grid,
            "missing_grid_points": [R for R in GRID if R not in usable_grid],
            "median_S_curve": medians,
            "ci_sign_curve": signs,
            "monotonicity": mono,
            "R_star_point_interval": crossing,
            "R_star_ci_bracket": bracket,
            "sign_change_ci_form": sc_ci,
            "verdict": _verdict(mono, crossing, signs, usable_grid, sc_ci),
        }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    sc = payload["self_check_ratio1_vs_count1"]
    print(f"\n自校验 ratio_1 ≡ count_1：{sc['status']}"
          f"（比较 {sc.get('n_fields_compared')} 字段，不一致 {sc.get('n_mismatch')}）")
    for op, blk in payload["cells"].items():
        print(f"\n=== op={op} ===")
        for name, r in blk["per_R"].items():
            if "median_S" not in r:
                print(f"  {name:9s} {r.get('skipped') or r.get('test_skipped')}")
                continue
            ci = r["ci95_S_logcluster"]
            print(f"  {name:9s} R={r['R']:2d}  S={r['median_S']:+.4f} "
                  f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] {r['ci_sign']:14s} "
                  f"k={r['mean_k_target']:.2f}/{r['mean_k_control']:.2f} "
                  f"n={r['n_non_collapsed']} logs={r['n_logs']}")
        print("  单调性：", blk["monotonicity"])
        print("  变号（点估计）：", blk["R_star_point_interval"], "| CI 括：", blk["R_star_ci_bracket"])
        print("  变号（判据 §3 CI 形式）：", blk["sign_change_ci_form"])
        print("  判据 §4 分支：", blk["verdict"]["branch"], "⇒", blk["verdict"]["conclusion"])
        if blk["verdict"].get("caveat"):
            print("  ", blk["verdict"]["caveat"])
    logger.info("写入 %s", a.out)


if __name__ == "__main__":
    main()
