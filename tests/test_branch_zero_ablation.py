#!/usr/bin/env python
"""`scripts/analysis/p0_branch_zero_ablation.py` 的单测。

判据设计原则（第八十九条：扰动必须能穿透被测组件的不变性）——
本脚本的实质逻辑有三块，每块都必须构造出「做错了会给出不同答案」的输入：

* [A] `BranchTap` 的替换语义：置零 vs 常量替换必须**只动指定的 128 维**，
  其余两段逐位不变。若切片写错（如 ego/gnn 写反），[A] 会立刻炸。
* [B] `verdict` 的裁定逻辑：判据边界 0.169 必须是**严格小于**，
  且 S1 要求「zero 的 CI 下界 > 0 **且** mean 的 CI 上界 < δ」两个条件同时成立。
* [C] `paired_stats` 的分析单位必须是 **log**，不是 token —— 🔴 命门一：
  把同一 log 的 100 个 token 当独立样本会让 CI 大幅偏窄，判据假通过（第六十八条）。
* [D] CI 与符号检验必须落在**同一个估计量**上 —— 🔴 命门二（08-06 code review 查出）：
  `cluster_bootstrap_ci` 重抽的是中位数，而 `verdict` 当时用 `delta_mean` 判符号。
  本条同时钉死「中位数分支与已落盘实现逐位一致」，避免另立第二套 CI 口径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))

from scripts.analysis.p0_branch_zero_ablation import (  # noqa: E402
    DELTA_MAIN, SLICES, VERDICT_CI_KEY, VERDICT_POINT_KEY, BranchTap, _cluster_ci,
    paired_stats, verdict,
)
from scripts.analysis.paired_usage_delta import cluster_bootstrap_ci  # noqa: E402

_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


class _Dummy(torch.nn.Module):
    """最小可测模型：只有一个 `feature_fusion`，输出即融合层输入的恒等映射。"""

    def __init__(self) -> None:
        super().__init__()
        self.feature_fusion = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_fusion(x)


def test_a_tap_replacement_semantics() -> None:
    """[A] 替换只作用于指定分支的 128 维，其余两段逐位不变。"""
    print("[A BranchTap 替换语义]")
    m = _Dummy()
    tap = BranchTap(m)
    x = torch.arange(384, dtype=torch.float32).repeat(4, 1)

    tap.replace = {}
    check(torch.equal(m(x), x), "不设 replace 时输出逐位等于输入")

    tap.replace = {"gnn": None}
    out = m(x)
    lo, hi = SLICES["gnn"]
    check(bool((out[:, lo:hi] == 0).all()), "zero 臂把 gnn 段置零")
    check(torch.equal(out[:, :lo], x[:, :lo]), "zero 臂不动 ego+ctx 段")

    const = torch.full((128,), 7.0)
    tap.replace = {"ctx": const}
    out = m(x)
    clo, chi = SLICES["ctx"]
    check(bool((out[:, clo:chi] == 7.0).all()), "mean 臂把 ctx 段替换为常量")
    check(torch.equal(out[:, chi:], x[:, chi:]), "mean 臂不动 gnn 段")
    check(torch.equal(out[:, :clo], x[:, :clo]), "mean 臂不动 ego 段")

    # 三段切片必须首尾相接且恰好覆盖 384 —— 写错会让某几维永远测不到
    bounds = sorted(SLICES.values())
    check(bounds[0][0] == 0 and bounds[-1][1] == 384
          and all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)),
          "ego/ctx/gnn 三段无缝覆盖 [0,384)")

    tap.close()
    tap.replace = {"gnn": None}
    check(torch.equal(m(x), x), "close() 之后 hook 不再生效")


def test_b_verdict_boundaries() -> None:
    """[B] 判据边界：严格小于 δ；S1 需要两个条件同时成立。"""
    print("[B verdict 裁定逻辑]")

    def mk(mean_hi: float, zero_lo: float, mean_delta: float = 0.01) -> dict:
        return {"arms": {
            "gnn_mean": {VERDICT_CI_KEY: [-0.01, mean_hi], VERDICT_POINT_KEY: mean_delta},
            "gnn_zero": {VERDICT_CI_KEY: [zero_lo, zero_lo + 0.2], VERDICT_POINT_KEY: 0.3},
        }}

    v = verdict([mk(0.10, 0.05)])
    check(v["P0-1-M_主判据"]["passed"], "CI 上界 0.10 < 0.169 ⇒ 主判据通过")
    check(v["S1_D-6判决"]["passed"], "zero 下界 >0 且 mean 上界 <δ ⇒ S1 通过（作用=纯 bias）")

    check(not verdict([mk(DELTA_MAIN, 0.05)])["P0-1-M_主判据"]["passed"],
          "CI 上界恰等于 0.169 ⇒ 不通过（严格小于）")
    check(not verdict([mk(0.10, -0.01)])["S1_D-6判决"]["passed"],
          "zero 的 CI 含 0 ⇒ S1 不通过（不能判成纯 bias）")
    check(not verdict([mk(0.30, 0.05)])["S1_D-6判决"]["passed"],
          "mean 也劣化 ⇒ S1 不通过（该分支有实质功能贡献）")

    # 多 run：任意一个不过则整体不过
    check(not verdict([mk(0.10, 0.05), mk(0.30, 0.05)])["P0-1-M_主判据"]["passed"],
          "三 seed 中任一不过 ⇒ 主判据不过")
    check(not verdict([mk(0.10, 0.05, +0.01), mk(0.10, 0.05, -0.01)])["S2_符号一致性"]["passed"],
          "两 run 符号相反 ⇒ S2 不通过")
    check(verdict([mk(0.10, 0.05, +0.01), mk(0.12, 0.06, +0.02)])["S2_符号一致性"]["passed"],
          "两 run 同号 ⇒ S2 通过")

    # 🔴 NaN 必须记 invalid，不能因为 `nan < δ` 恒 False 就说成「不通过」——
    #    那会把「测不了」（簇数 <2 等退化情形）伪装成「测过了而且没通过」。
    nan_v = verdict([mk(float("nan"), 0.05)])
    check(nan_v["P0-1-M_主判据"]["invalid"] and not nan_v["P0-1-M_主判据"]["passed"],
          "CI 为 NaN ⇒ 标 invalid 且不判通过（不得伪装成「不通过」）")
    check(all(nan_v[k]["invalid"] for k in ("P0-1-M_主判据", "S1_D-6判决", "S2_符号一致性")),
          "NaN 会污染全部三条判据，三条都标 invalid")


def test_c_cluster_is_the_analysis_unit() -> None:
    """[C] 🔴 命门：分析单位必须是 log。同一 log 内的 token 不是独立样本。"""
    print("[C 分析单位 = log 而非 token]")
    rng = np.random.default_rng(0)
    # 构造 log 内强相关的数据：每个 log 有一个偏移，log 内 token 只有微小抖动
    # 🔴 簇数取 24 = L1 val 的真实 log 簇数。实测：簇数降到 6 时 bootstrap 分位点
    # 会**完全离散化**（换 boot_seed 给出逐位相同的 CI）⇒ 小簇场景下的 CI 不可当连续量读。
    n_logs, per_log = 24, 50
    offsets = rng.normal(0.0, 0.20, size=n_logs)
    delta = np.concatenate([offsets[i] + rng.normal(0, 1e-3, per_log) for i in range(n_logs)])
    logs = np.concatenate([np.full(per_log, f"log{i}") for i in range(n_logs)])
    tokens_as_clusters = np.array([f"t{i}" for i in range(delta.size)])

    by_log = paired_stats(delta, logs, n_boot=2000, boot_seed=20260731)
    by_token = paired_stats(delta, tokens_as_clusters, n_boot=2000, boot_seed=20260731)
    w_log = by_log[VERDICT_CI_KEY][1] - by_log[VERDICT_CI_KEY][0]
    w_tok = by_token[VERDICT_CI_KEY][1] - by_token[VERDICT_CI_KEY][0]

    check(by_log["n_logs"] == n_logs, f"簇数按 log 计 = {n_logs}（不是 {delta.size} 个 token）")
    check(w_log > 3 * w_tok,
          f"log 聚类 CI 宽 {w_log:.4f} ≫ token 级 {w_tok:.4f} ⇒ 错用 token 会假阳性")

    # 可复现性：同 seed 逐位一致，异 seed 不同（否则 bootstrap 根本没在抽）
    again = paired_stats(delta, logs, n_boot=2000, boot_seed=20260731)
    check(again[VERDICT_CI_KEY] == by_log[VERDICT_CI_KEY], "同 boot_seed 逐位可复现")
    other = paired_stats(delta, logs, n_boot=2000, boot_seed=1)
    check(other[VERDICT_CI_KEY] != by_log[VERDICT_CI_KEY], "异 boot_seed 给出不同 CI")

    # 点估计同报均值与中位数（第八十三条：不许只挑一个）
    check("delta_mean" in by_log and "delta_median" in by_log, "均值与中位数同报")


def test_d_estimand_is_named_and_vendored_path_is_bitwise() -> None:
    """[D] 🔴 命门二：CI 与符号检验必须落在**同一个**估计量上，且中位数那套逐位复用已落盘实现。

    08-06 code review 查出的问题：`cluster_bootstrap_ci` 重抽的是**中位数**，
    而 `verdict` 当时拿 `delta_mean` 做符号判定 ⇒ 区间与符号检验是两个量。
    """
    print("[D 估计量一致性]")
    rng = np.random.default_rng(7)
    n_logs, per_log = 24, 40
    off = rng.normal(0.05, 0.15, n_logs)
    # 右偏 + 少量离群：均值与中位数会明显分开，否则本条测了等于没测
    delta = np.concatenate([off[i] + rng.exponential(0.05, per_log) for i in range(n_logs)])
    delta[:5] += 8.0
    logs = np.concatenate([np.full(per_log, f"log{i}") for i in range(n_logs)])

    st = paired_stats(delta, logs, n_boot=1000, boot_seed=20260731)
    check(abs(st["delta_mean"] - st["delta_median"]) > 0.02,
          f"构造出的均值 {st['delta_mean']:.4f} 与中位数 {st['delta_median']:.4f} 确实分开")
    check(st["ci95_logcluster_median"] != st["ci95_logcluster_mean"],
          "两个估计量的 CI 不同 ⇒ 混用会得到不同结论")

    # 中位数那条必须与已落盘的 `cluster_bootstrap_ci` **逐位相同**（不另立口径）
    ref = cluster_bootstrap_ci(delta, logs, 1000, 20260731)
    check(list(ref) == st["ci95_logcluster_median"],
          "median 分支与 paired_usage_delta.cluster_bootstrap_ci 逐位一致")
    check(list(_cluster_ci(delta, logs, 1000, 20260731, "median")) == list(ref),
          "_cluster_ci(stat='median') 与 vendored 实现逐位一致")

    # verdict 认的键必须真的存在于 paired_stats 的产物里（改名漏改会当场炸）
    check(VERDICT_CI_KEY in st and VERDICT_POINT_KEY in st,
          f"verdict 用的键 {VERDICT_CI_KEY}/{VERDICT_POINT_KEY} 都在 paired_stats 产物里")


def main() -> int:
    for fn in (test_a_tap_replacement_semantics, test_b_verdict_boundaries,
               test_c_cluster_is_the_analysis_unit,
               test_d_estimand_is_named_and_vendored_path_is_bitwise):
        fn()
    n_fail = len(_FAILED)
    print(f"\n{'✅ 全部通过' if n_fail == 0 else f'❌ {n_fail} 条失败'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
