"""`agg_2x2.py` 统计侧改造的单测（Q11 / S5 / D0#14 / D0#15）。

覆盖判据
--------
A. one-sided exact sign-flip 在 n=6 全同号时给出下限 `1/64 = 0.015625`；
B. 方向反了（全部变差）时 one-sided p ≈ 1，不会被误判为通过；
C. Holm 只覆盖 confirmatory family（2 项），n=6 下限被抬到 `0.03125`；
D. paired contrasts 与 S5 的三条公式逐值一致，且主效应/交互项尺度差 2×（D0#15）；
E. δ 未签字时判定门不下结论（§12.2：不得看到结果再定门槛）。

跑法::

    python3 scripts/analysis/tests/test_agg_2x2_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agg_2x2 import (  # noqa: E402
    PRIMARY_METRIC, analyse_metric, gate_verdict, holm, _sign_flip_p_one_sided,
)

SEEDS = [42, 1, 2, 3, 4, 5]


def _report_from(matrix: dict) -> dict:
    res = analyse_metric(matrix, SEEDS, PRIMARY_METRIC)
    primary = {f"{PRIMARY_METRIC}.route_main_effect": res["route_main_effect"]["p_one_sided_prereg"],
               f"{PRIMARY_METRIC}.perturbation_main_effect":
                   res["perturbation_main_effect"]["p_one_sided_prereg"]}
    return {"metrics": {PRIMARY_METRIC: res}, "holm_adjusted_p_primary": holm(primary)}


def test_a_one_sided_floor() -> None:
    diffs = np.array([-0.20, -0.18, -0.25, -0.15, -0.22, -0.19])
    p = _sign_flip_p_one_sided(diffs)
    assert abs(p - 1.0 / 64.0) < 1e-12, f"期望 1/64，实得 {p}"
    print(f"A ✅ n=6 全改善 one-sided p = {p:.6f} = 1/64")


def test_b_wrong_direction() -> None:
    diffs = np.array([0.20, 0.18, 0.25, 0.15, 0.22, 0.19])      # 全部变差
    p = _sign_flip_p_one_sided(diffs)
    assert p > 0.99, f"方向相反却给出 p={p}"
    print(f"B ✅ 全变差 one-sided p = {p:.6f}（不会被误判通过）")


def test_c_holm_family() -> None:
    adj = holm({"ade.route_main_effect": 1 / 64, "ade.perturbation_main_effect": 1 / 64})
    assert abs(adj["ade.route_main_effect"] - 2 / 64) < 1e-12
    assert abs(adj["ade.perturbation_main_effect"] - 2 / 64) < 1e-12
    print(f"C ✅ Holm(2 primary) 下限 = {adj['ade.route_main_effect']:.6f} = 1/32")


def test_d_contrast_scale() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(2.0, 0.05, 6)
    r = b - 0.10                      # route 单独改善 0.10
    p = b - 0.06                      # perturb 单独改善 0.06
    rp = b - 0.10 - 0.06 - 0.04       # 额外 -0.04 的协同 = interaction
    res = analyse_metric({"B": b, "R": r, "P": p, "RP": rp}, SEEDS, PRIMARY_METRIC)

    assert abs(res["route_main_effect"]["mean"] - (-0.12)) < 1e-9, res["route_main_effect"]["mean"]
    assert abs(res["perturbation_main_effect"]["mean"] - (-0.08)) < 1e-9
    assert abs(res["interaction"]["mean"] - (-0.04)) < 1e-9
    assert "÷2" in res["route_main_effect"]["scale"]
    assert "difference-in-differences" in res["interaction"]["scale"]
    assert res["route_main_effect"]["role"] == "confirmatory"
    assert res["interaction"]["role"] == "exploratory"
    print("D ✅ per-seed paired contrasts 与 S5 公式一致；主效应 ÷2 / interaction 未除，"
          "尺度标注正确，role 分流正确")


def test_e_delta_gate() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(2.0, 0.02, 6)
    r, p, rp = b - 0.10, b - 0.06, b - 0.16
    report = _report_from({"B": b, "R": r, "P": p, "RP": rp})

    unsigned = gate_verdict(report, {"route": None, "perturb": None}, 0.05)
    assert all("未签字" in line for line in unsigned[:2]), unsigned
    signed_pass = gate_verdict(report, {"route": 0.05, "perturb": 0.05}, 0.05)
    assert "PASS" in signed_pass[0], signed_pass[0]
    signed_fail = gate_verdict(report, {"route": 0.50, "perturb": 0.50}, 0.05)
    assert "FAIL" in signed_fail[0], signed_fail[0]
    print("E ✅ δ 未签字 -> 不判定；δ=0.05 -> PASS；δ=0.50 -> FAIL（效应未达门槛）")


def main() -> None:
    tests: List = [test_a_one_sided_floor, test_b_wrong_direction, test_c_holm_family,
                   test_d_contrast_scale, test_e_delta_gate]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} 通过")


if __name__ == "__main__":
    main()
