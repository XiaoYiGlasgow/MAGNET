"""`c4_readout.py` 三条事前判据的回归测试（判据先行）。

🔴 写于 2026-08-01 21:1x —— **八个 C4 格一个都没跑完的时候**。
用合成读数把 P1/P2/P3 的每个分支都走一遍，确保 08-02 凌晨真数据落地时
读表是机械的、且不会因为判据本身写错而误判（坑#23：判据也可能测错对象）。

判据来源：`C4_规划形式化_设计_20260801.md` §4.4。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import c4_readout as rd  # noqa: E402


def _cell(final: float, progress: float, making: float, zero_frac: float = 0.5) -> dict:
    """造一个格的读数（只填判据用得到的字段）。"""
    return {
        "final_score": final,
        "progress": progress,
        "making_prog": making,
        "zero_frac": zero_frac,
        "n_scenario": 256,
        "n_log": 117,
    }


def _joined(lines) -> str:
    return "\n".join(lines)


# ── P1：progress 是不是主效应 ────────────────────────────────────────────
def test_p1_holds_when_progress_drop_dominates() -> None:
    """full 关掉 progress 后两项都掉、且掉得比其余三件都多 ⇒ P1 成立。"""
    rows = {
        "magnet_layer": _cell(0.30, 0.500, 0.600),
        "no_progress": _cell(0.20, 0.300, 0.350),   # 掉 0.200 / 0.250
        "no_dynamics": _cell(0.28, 0.480, 0.580),   # 掉 0.020 / 0.020
        "no_collision": _cell(0.29, 0.470, 0.590),
        "no_rules": _cell(0.27, 0.460, 0.570),
    }
    out = _joined(rd.judge_p1(rows))
    assert "P1 成立" in out and "不成立" not in out


def test_p1_fails_when_not_both_down() -> None:
    """只有一项下降（另一项反而升）⇒ 事前预测未复现，必须判不成立。"""
    rows = {
        "magnet_layer": _cell(0.30, 0.500, 0.600),
        "no_progress": _cell(0.20, 0.300, 0.650),   # making_prog 反而升
        "no_dynamics": _cell(0.28, 0.480, 0.580),
        "no_collision": _cell(0.29, 0.470, 0.590),
        "no_rules": _cell(0.27, 0.460, 0.570),
    }
    out = _joined(rd.judge_p1(rows))
    assert "P1 不成立" in out and "没有双双下降" in out


def test_p1_fails_when_another_ablation_drops_more() -> None:
    """双双下降但 −rules 掉得更多 ⇒ 「主效应」这半条不成立，不得放宽。"""
    rows = {
        "magnet_layer": _cell(0.30, 0.500, 0.600),
        "no_progress": _cell(0.25, 0.450, 0.550),   # 掉 0.05 / 0.05
        "no_dynamics": _cell(0.28, 0.480, 0.580),
        "no_collision": _cell(0.29, 0.470, 0.590),
        "no_rules": _cell(0.10, 0.200, 0.300),      # 掉 0.30 / 0.30，更大
    }
    out = _joined(rd.judge_p1(rows))
    assert "P1 不成立" in out and "未**大于" in out


def test_p1_partial_when_ablation_cells_missing() -> None:
    """三件消融没跑完时，只能判前半条，且必须显式说后半条判不了。"""
    rows = {
        "magnet_layer": _cell(0.30, 0.500, 0.600),
        "no_progress": _cell(0.20, 0.300, 0.350),
    }
    out = _joined(rd.judge_p1(rows))
    assert "前半条成立" in out
    assert "no_dynamics" in out and "判不了" in out


def test_p1_undecidable_without_full() -> None:
    out = _joined(rd.judge_p1({"no_progress": _cell(0.20, 0.30, 0.35)}))
    assert "无法判定" in out


# ── P2：有没有把分数抬离地板 ─────────────────────────────────────────────
def test_p2_holds_below_baseline() -> None:
    rows = {"magnet_layer": _cell(0.30, 0.5, 0.6, zero_frac=0.60)}
    out = _joined(rd.judge_p2(rows))
    assert "P2 成立" in out and "60.0%" in out


def test_p2_fails_at_or_above_baseline() -> None:
    """等于基线也算不成立（判据是严格小于），且必须带上「不得调权重」的告诫。"""
    rows = {"magnet_layer": _cell(0.19, 0.4, 0.5, zero_frac=rd.ZERO_FRAC_BASELINE)}
    out = _joined(rd.judge_p2(rows))
    assert "P2 不成立" in out
    assert "不得为了拿正数去调权重" in out
    assert "干预协议" in out


def test_p2_baseline_matches_master_plan() -> None:
    """判据基线必须是主控表 §8 的 78.5%，不许被悄悄改动。"""
    assert rd.ZERO_FRAC_BASELINE == 0.785


# ── P3：学习组件的额外贡献（诚实性支点）─────────────────────────────────
def test_p3_reports_absolute_delta_even_when_tiny() -> None:
    """§4.4 写死：无论多小都要报绝对值。"""
    rows = {
        "magnet_layer": _cell(0.2000, 0.4, 0.5),
        "cv_on": _cell(0.1998, 0.4, 0.5),
        "cv_off": _cell(0.0500, 0.1, 0.2),
    }
    out = _joined(rd.judge_p3(rows))
    assert "Δfinal = +0.0002" in out
    assert "0.0002" in out
    assert "无可测量的额外贡献" in out       # 措辞提示必须出现


def test_p3_refuses_when_cv_missing() -> None:
    """CV 那格缺失时，绝不能只报 MAGNET+层的绝对分数。"""
    out = _joined(rd.judge_p3({"magnet_layer": _cell(0.30, 0.5, 0.6)}))
    assert "无法判定" in out
    assert "不得只报 MAGNET+层的绝对分数" in out


def test_p3_negative_delta_is_reported_signed() -> None:
    """MAGNET 比 CV 还差时必须如实报负号（不得取绝对值掩盖方向）。"""
    rows = {"magnet_layer": _cell(0.150, 0.4, 0.5), "cv_on": _cell(0.200, 0.4, 0.5)}
    out = _joined(rd.judge_p3(rows))
    assert "Δfinal = -0.0500" in out


# ── 口径回归：零分比例的算法必须与主控表 §8 一致 ──────────────────────────
def test_zero_frac_reproduces_calibrated_baseline() -> None:
    """在真实产物上复算：g13a calibrated 的零分比例 = 78.5%（主控表 §8）。

    这条同时锁住「逐场景行只认 log_name 非空」的口径 —— 用旧写法会得到 86.6%。
    """
    got = rd.read_cell(rd.REUSED["magnet_off"][0])
    assert got is not None, "基准产物读不到，先查 experiments/closed_loop/g13a_20260730/calibrated"
    assert abs(float(got["zero_frac"]) - 0.7852) < 5e-4
    assert got["n_scenario"] == 256 and got["n_log"] == 117
    assert abs(float(got["final_score"]) - 0.1978) < 5e-4
