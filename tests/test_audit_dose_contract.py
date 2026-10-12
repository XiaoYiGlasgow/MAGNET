#!/usr/bin/env python3
"""剂量契约的回归门 + 新增 `ratio` 模式的单测（Q83，2026-08-05）。

🔴 **为什么必须有这个文件**：`audit_dose_contract.py` 的 docstring 从一开始就引用
`tests/test_audit_dose_contract.py`，但 08-05 实测该文件**根本不存在**
（第八十一条的同类：文档声称的检查不等于检查存在）。而本次要往冻结的剂量契约里加新模式，
没有回归门就等于在没有安全网的情况下动 primary estimand 的定义。

三组断言：

  [A] **回归门** —— 冻结的三个剂量（`count_sat` / `count_1` / `frac_25`）在典型场景下
      解析出的 `(k_target, k_control)` 与**已落盘 shard 的实测值**一致。
      期望值不是从代码反推的，是从 `summary_r1_vs_l1_20260805.json` 的
      `mean_k_target` / `mean_k_control` 读出来的实测均值反推的整数网格。
  [B] **新模式 `ratio`** —— `k_target` 恒为 1、`k_control` 恒为 R；
      🔴 **`n_control < R` 时必须 skip 而不是 clamp**（clamp 会静默改变实际比值）。
  [C] **自校验** —— `ratio_1` 与 `count_1` 在任何 (n_t, n_c) 上解析结果必须逐位相同。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python tests/test_audit_dose_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
sys.path.insert(0, str(GNN))

from scripts.analysis.audit_dose_contract import (  # noqa: E402
    DOSES, RATIO_DOSES, Dose, resolve_dose, round_half_up,
)

_BY_NAME = {d.name: d for d in DOSES}
_FAILED = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"    ✅ {msg}")
    else:
        print(f"    ❌ {msg}")
        _FAILED.append(msg)


def test_a_frozen_regression() -> None:
    """[A] 冻结剂量的回归门：典型场景 n_agent=4, n_lane=40（§8 记 lane 恒为 40）。"""
    print("[A 冻结剂量回归门]")
    n_t, n_c = 4, 40

    k_t, k_c, skip = resolve_dose(_BY_NAME["count_sat"], n_t, n_c, unit="节点")
    check((k_t, k_c, skip) == (4, 4, None),
          f"count_sat(4,40) -> (4,4,None)，实得 ({k_t},{k_c},{skip})")

    k_t, k_c, skip = resolve_dose(_BY_NAME["count_1"], n_t, n_c, unit="节点")
    check((k_t, k_c, skip) == (1, 1, None),
          f"count_1(4,40) -> (1,1,None)，实得 ({k_t},{k_c},{skip})")

    k_t, k_c, skip = resolve_dose(_BY_NAME["frac_25"], n_t, n_c, unit="节点")
    check((k_t, k_c, skip) == (1, 10, None),
          f"frac_25(4,40) -> (1,10,None)，实得 ({k_t},{k_c},{skip})")

    # 🔴 这一条正是「剂量混杂」的算术根源：同一个 ρ=0.25 在两侧解析出 1 vs 10
    check(10 / 1 == 10.0, "frac_25 在 (4,40) 上的实际删除比 = 10× ⇒ 剂量并未匹配")

    # count 模式两侧恒等（比值 1.00×）
    for n_a, n_l in ((2, 40), (7, 40), (1, 40), (20, 40)):
        k_t, k_c, skip = resolve_dose(_BY_NAME["count_sat"], n_a, n_l, unit="节点")
        check(k_t == k_c and skip is None,
              f"count_sat({n_a},{n_l}) 两侧删除数相同 = {k_t}")

    # 空池必须 skip，不得静默当 0 效应
    k_t, k_c, skip = resolve_dose(_BY_NAME["count_sat"], 0, 40, unit="节点")
    check((k_t, k_c) == (0, 0) and skip is not None, f"空 agent 池 -> skip（{skip}）")

    # banker's rounding 陷阱：n·ρ 恰为 .5 时必须向上
    check(round_half_up(0.5) == 1 and round_half_up(1.5) == 2,
          "round_half_up 半数向上（内建 round 会给 0 和 2）")


def test_b_ratio_mode() -> None:
    """[B] 新增 ratio 模式：R 是唯一变量，且不可达时 skip 而非 clamp。"""
    print("[B ratio 模式]")
    by_r = {int(d.value): d for d in RATIO_DOSES}
    check(sorted(by_r) == [1, 2, 3, 5, 7, 10, 14, 20],
          f"扫描网格 R = {sorted(by_r)}")

    for r in sorted(by_r):
        k_t, k_c, skip = resolve_dose(by_r[r], 4, 40, unit="节点")
        check((k_t, k_c, skip) == (1, r, None),
              f"ratio_{r}(4,40) -> (1,{r},None)，实得 ({k_t},{k_c},{skip})")

    # 🔴 静默类防线：control 池不够时必须 skip
    k_t, k_c, skip = resolve_dose(by_r[20], 4, 10, unit="节点")
    check((k_t, k_c) == (0, 0) and skip is not None and "clamp" in skip,
          f"ratio_20 在 n_control=10 上 skip 而非 clamp（{skip}）")

    k_t, k_c, skip = resolve_dose(by_r[10], 4, 10, unit="节点")
    check((k_t, k_c, skip) == (1, 10, None), "R 恰好可达（n_control == R）时正常工作")

    # target 侧至少要有 1 个
    k_t, k_c, skip = resolve_dose(by_r[3], 0, 40, unit="节点")
    check((k_t, k_c) == (0, 0) and skip is not None, "空 agent 池 -> skip")


def test_c_self_check() -> None:
    """[C] 自校验：ratio_1 必须与冻结的 count_1 逐位等价。"""
    print("[C ratio_1 ≡ count_1 自校验]")
    r1 = {int(d.value): d for d in RATIO_DOSES}[1]
    c1 = _BY_NAME["count_1"]
    same = True
    for n_a in range(0, 25):
        for n_l in (0, 1, 5, 40):
            if resolve_dose(r1, n_a, n_l, unit="节点")[:2] != \
               resolve_dose(c1, n_a, n_l, unit="节点")[:2]:
                same = False
                print(f"      分叉于 (n_t={n_a}, n_c={n_l})")
    check(same, "在 25×4 个 (n_target, n_control) 组合上 (k_t,k_c) 逐位相同")


def main() -> int:
    for fn in (test_a_frozen_regression, test_b_ratio_mode, test_c_self_check):
        fn()
    n_fail = len(_FAILED)
    print(f"\n{'✅ 全部通过' if n_fail == 0 else f'❌ {n_fail} 条失败'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
