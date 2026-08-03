"""剂量口径的回归测试 —— **Q47 已收口（2026-08-01）：两侧只剩一份定义**。

历史：两侧跑在不同 env（`plantf` 没有 `torch_geometric`），无法共享 import，
G19 运行期间也不能改 MAGNET 侧脚本 ⇒ 曾有两份等价副本并存，
本测试当时是它们不分叉的唯一保证（162 个组合逐点比对）。

G19 于 08-01 08:52 全部跑完后，MAGNET 侧已改为 import 共享模块并删掉副本，
本文件随之退化为**回归测试**，现在守三件事：

1. MAGNET 侧确实**复用**共享定义（而不是又长出一份副本）；
2. 🔴 **`unit="节点"` 没有被改掉** —— `skip_reason` 会被写进 audit shard 的 JSON 产物，
   已落盘的 G19 四 config 用的就是「节点」。这条一旦松动，那批产物即不可逐位复现；
3. `count_sat` 的池规模不对称性（协议 §3.2 第二档判读规则的存在理由）没被「顺手改对称」。

⚠️ 本测试必须在 **`nuplan` env** 跑（要 import MAGNET 侧脚本，它依赖 torch_geometric）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/tests/test_audit_dose_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[3]
for _p in (str(_GNN_ROOT), str(_THIS.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 与两个生产脚本走**同一条包路径** —— 否则同一文件被加载两次，
# 本文件的 `DOSES is M.DOSES` 断言会假失败（它正是要抓住双重加载）。
from scripts.analysis.audit_dose_contract import (  # noqa: E402
    DOSES, Dose, resolve_dose, round_half_up)


def test_magnet_reuses_shared_definition() -> None:
    """MAGNET 侧必须**复用**共享定义，而不是又长出一份副本（Q47 的全部意义）。"""
    import scripts.analysis.run_interaction_audit as M

    assert M.DOSES is DOSES, "MAGNET 侧的 DOSES 不是共享对象 —— 副本又长回来了"
    assert M.Dose is Dose, "MAGNET 侧的 Dose 类不是共享类"
    assert not hasattr(M, "_round_half_up"), \
        "MAGNET 侧又出现了自己的 _round_half_up 副本（Q47 已删，不应复活）"
    assert round_half_up(0.5) == 1, "0.5 必须进位到 1（Python 内建 round 会给 0）"
    print(f"[1] MAGNET 侧复用共享定义             PASS  ({[d.name for d in DOSES]}，无副本)")


def test_magnet_binding_locks_unit_to_node() -> None:
    """🔴 MAGNET 侧的 `skip_reason` 量词必须锁死为「节点」。

    `run_interaction_audit` 把 `skip_reason` 写进 shard JSON
    （`out[dose.name] = {"mode": ..., "skipped": skip}`），
    已落盘的 G19 四 config 用的就是「节点」。量词一改，那批产物即不可逐位复现。
    共享模块的默认量词是「实体」（PlanTF 侧在用），故这条绑定必须显式存在。
    """
    from scripts.analysis.run_interaction_audit import DOSE_UNIT, _resolve_dose as magnet_resolve

    assert DOSE_UNIT == "节点", f"MAGNET 侧量词被改成了 {DOSE_UNIT!r}，会让 G19 产物不可复现"

    dose = next(d for d in DOSES if d.name == "count_sat")
    _, _, skip_empty = magnet_resolve(dose, 0, 40)
    assert skip_empty == "缺少 agent 或 lane 节点（n_t=0, n_c=40）", \
        f"空池 skip 文本变了: {skip_empty!r}"

    frac = next(d for d in DOSES if d.mode == "fraction")
    _, _, skip_frac = magnet_resolve(frac, 1, 1)
    assert skip_frac == "fraction 剂量解析为 0 个节点（k_t=0, k_c=0）", \
        f"fraction skip 文本变了: {skip_frac!r}"

    # 共享模块的默认（PlanTF 侧）必须仍是「实体」，不被 MAGNET 的绑定污染
    _, _, shared_skip = resolve_dose(dose, 0, 40)
    assert shared_skip.endswith("实体（n_t=0, n_c=40）"), \
        f"共享模块默认量词被改了: {shared_skip!r}"
    print("[2] MAGNET 量词锁死「节点」            PASS  (共享默认仍为「实体」，两侧未串)")


def test_resolve_dose_matches_magnet() -> None:
    """在实体池大小的网格上逐点比对 `(k_target, k_control, 是否跳过)`。

    Q47 后 MAGNET 侧只是薄绑定，本测试守的是「绑定没把参数传错」
    （例如把 target/control 传反会让 count 口径的 `min` 取到另一侧）。
    """
    from scripts.analysis.run_interaction_audit import _resolve_dose as magnet_resolve

    n_checked = 0
    for dose in DOSES:
        for n_t in (0, 1, 2, 3, 5, 8, 13, 20, 40):
            for n_c in (0, 1, 2, 4, 8, 40):
                kt1, kc1, skip1 = resolve_dose(dose, n_t, n_c,
                                               target_name="agent", control_name="lane",
                                               unit="节点")
                kt2, kc2, skip2 = magnet_resolve(dose, n_t, n_c)
                assert (kt1, kc1, skip1) == (kt2, kc2, skip2), \
                    f"{dose.name} n_t={n_t} n_c={n_c}: ({kt1},{kc1},{skip1!r}) vs ({kt2},{kc2},{skip2!r})"
                n_checked += 1
    print(f"[3] 薄绑定逐点一致（含 skip 文本）    PASS  ({n_checked} 个 (剂量,n_t,n_c) 组合)")


def test_count_sat_asymmetry_is_visible() -> None:
    """G19 的关键读数性质：`count_sat` 在池规模悬殊时两侧删除**比例**必然不等。

    这不是 bug，是协议 §3.2 第二档判读规则存在的原因 —— 固化成回归测试，
    防止以后有人「顺手把它改对称」从而悄悄改掉 primary 的定义。
    """
    dose = next(d for d in DOSES if d.name == "count_sat")
    k_t, k_c, skip = resolve_dose(dose, n_target=2, n_control=40)
    assert skip is None and k_t == k_c == 2, f"期望两侧各删 2 个，实得 {k_t}/{k_c}"
    assert k_t / 2 == 1.0 and k_c / 40 == 0.05, "agent 侧删 100%、lane 侧只删 5% —— 这正是剂量不对等"
    print("[4] count_sat 池规模不对称性已固化    PASS  (agent 100% vs control 5%)")


def test_fraction_never_empties_pool() -> None:
    """fraction 口径禁止 ρ=1.0（两类实体全删 ⇒ 图为空）。"""
    for d in DOSES:
        if d.mode == "fraction":
            assert d.value < 1.0, f"{d.name} 的 ρ={d.value} 会把实体池清空"
    print("[5] fraction 剂量 ρ<1.0               PASS")


if __name__ == "__main__":
    tests = [
        test_magnet_reuses_shared_definition,
        test_magnet_binding_locks_unit_to_node,
        test_resolve_dose_matches_magnet,
        test_count_sat_asymmetry_is_visible,
        test_fraction_never_empties_pool,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[{t.__name__}] FAIL  {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} PASS")
    sys.exit(1 if failed else 0)
