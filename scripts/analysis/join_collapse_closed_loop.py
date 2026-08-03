"""Q46 —— 把 planner 塌陷探针的逐步读数与**官方闭环分数**按场景 join，满足 §7 硬门。

硬门是什么
----------
主控表 §7：**塌陷在与闭环失效逐 token 关联之前，不得写进任何结论。**
2026-08-01 实测：离线审计集（val 3000 token / 60 log）与闭环集（heldout 256 场景 / 117 log）
**token 与 log 交集均为 0** ⇒ 事后关联建不起来。
出路是让塌陷读数**产生在闭环本身**（`gnn_trans_planner.py` 的 `_CollapseProbe`），
本脚本负责把它与官方 aggregator 的分数 join 起来。

join key
--------
探针 JSONL 每行有 `log_name` + `token`；官方 aggregator parquet 有 `log_name` + `scenario`（=token）。
两边都取**非汇总行**（parquet 里 `log_name` 为空的是 scenario_type 汇总行与 `final_score` 行，
这正是 2026-07-31 把 269 误当场景数的那个坑）。

口径**在见到任何真实探针数据之前定死**（防事后挑口径）
------------------------------------------------------
2026-08-01 用合成数据 dry run 时发现：`any_collapse`（≈150 步里有 1 步塌陷即为真）
在任何非零塌陷概率下都会**饱和到 1.0**，分组直接退化、比较无法进行。
因此口径定为：

* **primary = 连续量**：`collapse_step_frac`（该场景塌陷步占比）与 `score` / 各失效通道的
  **Spearman 相关**，按 log 聚类 bootstrap 95% CI。连续量不需要任何阈值，也就没有挑阈值的空间。
* **secondary = 二分**：阈值**预先定为 0.5**（多数步塌陷），报组间均值差 + 同款聚类 bootstrap CI。
* `any_collapse` 只作**描述性**统计，**不作分组变量**。

⚠️ 以上三条写在真实探针数据产生**之前**。若日后要改，必须走协议 §11 变更记录。

报什么、不报什么
----------------
* 报：上述关联 + 塌陷步占比分布 + 首次塌陷时刻 + join 覆盖率；
* 🔴 **不报因果**。这是同一次运行内的**关联**，不是干预。塌陷与低分可能同源于「场景难」。
  要主张因果需要另做干预（例如强制绕开塌陷），本脚本不提供，也不得在文中暗示。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 官方失效通道（与 §7 硬门里「接到官方失效通道」的那几列一致）
FAILURE_CHANNELS = (
    "drivable_area_compliance",
    "no_ego_at_fault_collisions",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
    "speed_limit_compliance",
)


def load_probe(probe_dir: Path) -> pd.DataFrame:
    """把一堆 `<log_name>__<token>.jsonl` 汇成每场景一行。

    :returns: 列 = token / log_name / n_steps / n_collapsed / collapse_step_frac /
        first_collapse_step / any_collapse。
    """
    rows: List[Dict] = []
    for f in sorted(probe_dir.glob("*.jsonl")):
        steps = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                steps.append(json.loads(line))
        if not steps:
            logger.warning("空探针文件，跳过: %s", f.name)
            continue
        flags = [s.get("collapsed") for s in steps]
        known = [b for b in flags if b is not None]
        if len(known) != len(flags):
            # 不静默：没捕获到激活的步会让分母失真，必须让使用者看到
            logger.warning("%s 有 %d/%d 步未捕获到激活", f.name, len(flags) - len(known), len(flags))
        first = next((s["iteration"] for s in steps if s.get("collapsed")), None)
        rows.append({
            "token": steps[0].get("token"),
            "log_name": steps[0].get("log_name"),
            "scenario_type": steps[0].get("scenario_type"),
            "n_steps": len(steps),
            "n_collapsed": int(sum(1 for b in known if b)),
            "collapse_step_frac": round(sum(1 for b in known if b) / len(known), 6) if known else None,
            "first_collapse_step": first,
            "any_collapse": bool(any(known)),
        })
    return pd.DataFrame(rows)


def load_scores(group_dir: Path) -> pd.DataFrame:
    """读官方 aggregator parquet 的**逐场景**行。

    ⚠️ `log_name` 为空的行是 scenario_type 汇总行 + `final_score` 行，**必须滤掉** ——
    把它们当成场景正是 07-31 查出的 269/256 口径错误的成因。
    """
    files = sorted(group_dir.rglob("aggregator_metric/*.parquet"))
    if not files:
        raise SystemExit(f"{group_dir} 下找不到 aggregator_metric/*.parquet")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if "log_name" not in df.columns or "scenario" not in df.columns:
            continue
        frames.append(df[df["log_name"].notna()].copy())
    if not frames:
        raise SystemExit("aggregator parquet 里没有逐场景行")
    out = pd.concat(frames, ignore_index=True)
    logger.info("读到逐场景行 %d 条 <- %d 个 parquet", len(out), len(files))
    return out.rename(columns={"scenario": "token"})


#: secondary 二分阈值。**在见到真实数据之前定死**（见模块 docstring）。
DICHOTOMY_THRESHOLD = 0.5


def _safe_round(x, nd: int = 4):
    """NaN -> None。`json.dump` 会把 NaN 写成裸 `NaN`，那不是合法 JSON，下游会静默读错。"""
    return None if x is None or np.isnan(x) else round(float(x), nd)


def _log_cluster_indices(sub: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """按 **log 聚类**重抽一次，返回行下标。

    按 log 重抽而不是按场景，因为同一 log 里的场景不独立（同一段路、同一批他车）。
    """
    by_log = sub.groupby("log_name").indices          # {log: array(行位置)}
    logs = np.array(list(by_log.keys()))
    pick = rng.choice(logs, size=len(logs), replace=True)
    return np.concatenate([by_log[lg] for lg in pick])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman = 秩上的 Pearson。自己算避免为一个数引入 scipy 依赖。"""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_bootstrap_assoc(df: pd.DataFrame, col: str, n_boot: int = 10000,
                            seed: int = 1000) -> Optional[Dict[str, object]]:
    """primary（连续 Spearman）+ secondary（阈值 0.5 二分均值差），都带 log 聚类 bootstrap CI。"""
    sub = df[["log_name", "collapse_step_frac", col]].dropna().reset_index(drop=True)
    if len(sub) < 3:
        return None
    frac = sub["collapse_step_frac"].to_numpy(dtype=float)
    val = sub[col].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    rho_boot, diff_boot = [], []
    for _ in range(n_boot):
        idx = _log_cluster_indices(sub, rng)
        r = _spearman(frac[idx], val[idx])
        if not np.isnan(r):
            rho_boot.append(r)
        hi = frac[idx] >= DICHOTOMY_THRESHOLD
        if hi.any() and (~hi).any():
            diff_boot.append(val[idx][hi].mean() - val[idx][~hi].mean())

    hi = frac >= DICHOTOMY_THRESHOLD
    out: Dict[str, object] = {
        "primary_spearman_rho": round(_spearman(frac, val), 4),
        "primary_ci": ([round(float(np.percentile(rho_boot, 2.5)), 4),
                        round(float(np.percentile(rho_boot, 97.5)), 4)] if rho_boot else None),
        "n": int(len(sub)),
        "n_logs": int(sub["log_name"].nunique()),
        "secondary_threshold": DICHOTOMY_THRESHOLD,
        "secondary_n_high": int(hi.sum()),
        "secondary_n_low": int((~hi).sum()),
    }
    if hi.any() and (~hi).any():
        out["secondary_mean_high"] = round(float(val[hi].mean()), 4)
        out["secondary_mean_low"] = round(float(val[~hi].mean()), 4)
        out["secondary_diff"] = round(float(val[hi].mean() - val[~hi].mean()), 4)
        out["secondary_ci"] = ([round(float(np.percentile(diff_boot, 2.5)), 4),
                                round(float(np.percentile(diff_boot, 97.5)), 4)]
                               if diff_boot else None)
    else:
        out["secondary_note"] = f"阈值 {DICHOTOMY_THRESHOLD} 下有一侧为空，二分不可用（primary 仍有效）"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Q46：塌陷探针 × 官方闭环分数逐场景关联")
    p.add_argument("--probe_dir", required=True, help="探针 JSONL 目录（绝对路径）")
    p.add_argument("--group_dir", required=True, help="闭环 group 目录（绝对路径）")
    p.add_argument("--out", required=True, help="输出 JSON（绝对路径）")
    p.add_argument("--n_boot", type=int, default=10000)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    probe = load_probe(Path(a.probe_dir))
    scores = load_scores(Path(a.group_dir))
    logger.info("探针场景 %d 个 | 闭环场景 %d 个", len(probe), len(scores))

    merged = probe.merge(scores, on=["log_name", "token"], how="inner", suffixes=("", "_score"))
    # 🔴 join 覆盖率必须报出来：覆盖不全说明探针漏了场景，结论会有选择偏差
    coverage = {
        "n_probe": len(probe), "n_scores": len(scores), "n_joined": len(merged),
        "probe_unmatched": int(len(probe) - len(merged)),
        "scores_unmatched": int(len(scores) - len(merged)),
    }
    if len(merged) == 0:
        raise SystemExit(f"join 结果为空，检查 join key。{coverage}")

    out: Dict = {
        "coverage": coverage,
        "collapse_summary": {
            # any_collapse 只作描述，**不作分组变量**（≈150 步里 1 步为真即饱和到 1.0）
            "any_collapse_rate_descriptive_only": round(float(merged["any_collapse"].mean()), 4),
            # NaN 只在「所有场景都没捕获到激活」时出现（探针挂不上会先 raise，故基本不可达）；
            # 但 json.dump 会写出裸 NaN —— 那是**无效 JSON**，下游读取会静默出错，所以显式转 None
            "mean_collapse_step_frac": _safe_round(merged["collapse_step_frac"].mean()),
            "median_collapse_step_frac": _safe_round(merged["collapse_step_frac"].median()),
            "median_first_collapse_step": (
                float(merged["first_collapse_step"].median())
                if merged["first_collapse_step"].notna().any() else None
            ),
        },
        "association": {},
        "estimand": ("primary = collapse_step_frac 与该列的 Spearman 相关（log 聚类 bootstrap 95% CI）；"
                     f"secondary = 阈值 {DICHOTOMY_THRESHOLD} 二分的均值差。两者均在见到真实探针数据前定死。"),
        "caveat": ("同一次运行内的关联，不是干预 —— 塌陷与低分可能同源于场景难度。"
                   "不得在论文中写成因果链。"),
    }

    for col in ("score",) + FAILURE_CHANNELS:
        if col not in merged.columns:
            logger.warning("缺列 %s，跳过", col)
            continue
        res = cluster_bootstrap_assoc(merged, col, n_boot=a.n_boot)
        out["association"][col] = res if res is not None else "有效样本不足，无法比较"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    merged.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    logger.info("写出 %s（+ 同名 .csv 逐场景明细）", a.out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
