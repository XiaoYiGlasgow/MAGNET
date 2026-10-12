#!/usr/bin/env python
"""`scripts/analysis/graph_edge_accounting.py` 的单测。

判据设计原则（第八十九条：扰动必须能穿透被测组件的不变性）——
本模块唯一的实质逻辑是「策略如何改变入图集合」，所以每条判据都必须构造出
**能让不同策略给出不同答案**的输入；否则测了等于没测。

🔴 [B] 是命门：它钉死「先取最近 N 再删静止」与「先删静止再取最近 N」的差，
也就是「更正 2」那 +47% 的全部来源。
"""
from __future__ import annotations

import sys
from pathlib import Path

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))

from scripts.analysis.graph_edge_accounting import (  # noqa: E402
    DEFAULT_POLICIES, RetentionPolicy, apply_policy, describe, n_choose_2,
    n_choose_2_continuous, summarize_counts,
)

_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


_BY_NAME = {p.name: p for p in DEFAULT_POLICIES}


def test_a_edge_count_basics() -> None:
    """[A] `C(n,2)` 与「实测中位场景」的逐位核对。"""
    print("[A 边数基本量]")
    check(n_choose_2(0) == 0 and n_choose_2(1) == 0, "n<2 时无边")
    check(n_choose_2(2) == 1 and n_choose_2(3) == 3 and n_choose_2(43) == 903, "C(2,2)=1 C(3,2)=3 C(43,2)=903")

    # 🔑 实测中位场景：2 个他车 + 40 条 lane + 1 个 ego = 43 节点 ⇒ 903 条边
    #    这与阶段109 实测的「总边数中位 903」逐位吻合 ⇒ 钉死图结构约定
    sc = apply_policy([True, True], n_lane_uncapped=40, policy=_BY_NAME["current"])
    check(sc.n_nodes == 43, f"节点数 = 1+2+40 = 43（实测 {sc.n_nodes}）")
    check(sc.edges_total == 903, f"总边数 = 903（实测 {sc.edges_total}）")
    check(sc.edges_agent_agent == 1, "他车↔他车边 = C(2,2) = 1")
    check(sc.edges_ego_agent == 2, "ego↔他车边 = k = 2")


def test_b_order_swap_is_the_whole_gain() -> None:
    """[B] 🔴 命门：过滤顺序调换必须、且只在「静止车挤占了槽位」时产生差异。"""
    print("[B 过滤顺序 —— 命门]")
    # 构造：最近的 20 个槽全是静止车，第 21–25 槽才是运动车。
    # 现状（先截断再删静止）⇒ 一辆运动车都进不来；调换顺序 ⇒ 5 辆全进。
    slots = [False] * 20 + [True] * 5
    cur = apply_policy(slots, 40, _BY_NAME["current"])
    swp = apply_policy(slots, 40, _BY_NAME["swap_order"])
    check(cur.m_moving == 0, f"现状：0 辆运动车进图（实测 {cur.m_moving}）")
    check(swp.m_moving == 5, f"调换顺序：5 辆全部进图（实测 {swp.m_moving}）")
    check(cur.edges_moving_moving == 0 and swp.edges_moving_moving == 10,
          "运动×运动边 0 → C(5,2)=10")

    # 反向判据：没有静止车时两策略必须**逐位相同**（否则说明改动泄漏到了不该改的路径）
    slots_clean = [True] * 7
    a = apply_policy(slots_clean, 40, _BY_NAME["current"])
    b = apply_policy(slots_clean, 40, _BY_NAME["swap_order"])
    check(a == b, "无静止车时两策略逐位相同（改动不泄漏）")

    # 单调性：任意输入下 swap 的运动车数 ≥ current
    import random
    rng = random.Random(20260806)
    mono = all(
        apply_policy(s, 40, _BY_NAME["swap_order"]).m_moving
        >= apply_policy(s, 40, _BY_NAME["current"]).m_moving
        for s in ([rng.random() < 0.25 for _ in range(rng.randint(0, 60))] for _ in range(500))
    )
    check(mono, "500 组随机输入上 swap 的运动车数恒 ≥ current")


def test_c_planb_keeps_static() -> None:
    """[C] 方案B 保留静止车，且上限确实放大。"""
    print("[C 方案B 保留静止]")
    slots = [False] * 30 + [True] * 3          # 33 辆车，最近 30 辆静止
    cur = apply_policy(slots, 40, _BY_NAME["current"])
    b48 = apply_policy(slots, 68, _BY_NAME["planB_48"])
    check(cur.k_agents == 0, f"现状入图他车 0（实测 {cur.k_agents}）")
    check(b48.k_agents == 33, f"方案B(48) 入图他车 33（实测 {b48.k_agents}）")
    check(b48.m_moving == 3, f"其中运动 3（实测 {b48.m_moving}）")
    # 🔑 方案B 多出来的边绝大部分是「停车×停车」—— 这正是「更正 1」的要点
    check(b48.edges_agent_agent == 528 and b48.edges_moving_moving == 3,
          f"他车↔他车 C(33,2)=528 但运动×运动仅 C(3,2)=3（实测 {b48.edges_agent_agent}/{b48.edges_moving_moving}）")

    # 截断确实在 agent_cap 处发生
    many = [True] * 100
    check(apply_policy(many, 68, _BY_NAME["planB_48"]).k_agents == 48, "planB_48 在 48 截断")
    check(apply_policy(many, 68, _BY_NAME["planB_64"]).k_agents == 64, "planB_64 在 64 截断")


def test_d_lane_cap_isolation() -> None:
    """[D] lane 上限只影响总边数，不得渗进任何 agent 侧读数。"""
    print("[D lane 上限隔离]")
    slots = [True, False, True]
    p68 = RetentionPolicy("t68", agent_cap=48, lane_cap=68, keep_static=True)
    p80 = RetentionPolicy("t80", agent_cap=48, lane_cap=80, keep_static=True)
    a = apply_policy(slots, n_lane_uncapped=75, policy=p68)
    b = apply_policy(slots, n_lane_uncapped=75, policy=p80)
    check((a.k_agents, a.m_moving, a.edges_agent_agent, a.edges_moving_moving)
          == (b.k_agents, b.m_moving, b.edges_agent_agent, b.edges_moving_moving),
          "lane 上限 68 vs 80 下 agent 侧四个量逐位相同")
    check(a.n_lane == 68 and b.n_lane == 75, "lane 截断生效：min(75,68)=68；min(75,80)=75")
    check(a.edges_total != b.edges_total, "但总边数不同（判据本身有穿透力）")


def test_e_jensen_direction() -> None:
    """[E] 精确均值必须 ≥ `C(均值,2)` 近似 —— 主控表那一列的偏差方向。"""
    print("[E Jensen 方向]")
    import random
    rng = random.Random(1)
    scenes = [[rng.random() < 0.3 for _ in range(rng.randint(0, 40))] for _ in range(800)]
    counts = [apply_policy(s, 40, _BY_NAME["planB_48"]) for s in scenes]
    s = summarize_counts(counts)
    j = s["jensen_check"]
    check(j["edges_moving_moving_exact_mean"] >= j["edges_moving_moving_approx_C_of_mean"],
          f"运动×运动：精确 {j['edges_moving_moving_exact_mean']} ≥ 近似 {j['edges_moving_moving_approx_C_of_mean']}")
    check(j["edges_agent_agent_exact_mean"] >= j["edges_agent_agent_approx_C_of_mean"],
          f"他车↔他车：精确 {j['edges_agent_agent_exact_mean']} ≥ 近似 {j['edges_agent_agent_approx_C_of_mean']}")
    check(j["edges_moving_moving_underestimate_x"] > 1.0, "低估倍数 > 1（近似确实偏低）")

    # 🔴 穿透性判据（第八十九条）：上面的随机样本量大，`round(均值)` 的偏差会被平均掉，
    #    测不出取整 bug。这里刻意构造**小样本 + 非整数均值**：
    #    m = [3, 4] ⇒ 精确 mean(3, 6) = 4.5；连续近似 3.5×2.5/2 = 4.375（✅ 方向对）；
    #    若误用 round ⇒ C(4,2) = 6 > 4.5（🔴 方向反）。08-06 冒烟实测就踩在这里。
    tiny = [apply_policy([True] * 3, 0, _BY_NAME["swap_order"]),
            apply_policy([True] * 4, 0, _BY_NAME["swap_order"])]
    jt = summarize_counts(tiny)["jensen_check"]
    check(jt["edges_moving_moving_exact_mean"] == 4.5, f"精确均值 4.5（实测 {jt['edges_moving_moving_exact_mean']}）")
    check(jt["edges_moving_moving_approx_C_of_mean"] == 4.375,
          f"连续近似 4.375、非 round 后的 6（实测 {jt['edges_moving_moving_approx_C_of_mean']}）")
    check(jt["edges_moving_moving_underestimate_x"] > 1.0, "小样本非整数均值下方向仍正确")

    check(n_choose_2_continuous(4.26) == 4.26 * 3.26 / 2, "连续式与主控表 §0.0-f 口径一致（4.26 ⇒ 6.94）")


def test_f_edges_and_guards() -> None:
    """[F] 边界输入与拒绝路径。"""
    print("[F 边界与拒绝]")
    empty = apply_policy([], 0, _BY_NAME["current"])
    check(empty.k_agents == 0 and empty.edges_total == 0 and empty.n_nodes == 1,
          "空场景：只剩 ego 一个节点、零边")
    all_static = apply_policy([False] * 15, 40, _BY_NAME["current"])
    check(all_static.k_agents == 0, "全静止场景在现状策略下入图为 0")
    kept_static = apply_policy([False] * 15, 40, _BY_NAME["planB_48"])
    check(kept_static.k_agents == 15 and kept_static.m_moving == 0,
          "同一场景在方案B 下入图 15 辆、运动 0 辆")

    try:
        summarize_counts([])
        ok = False
    except ValueError:
        ok = True
    check(ok, "空序列汇总当场抛 ValueError（不出假读数）")
    check(describe([])["median"] is None, "describe 空输入返回 None 而非报警")

    # 零边率
    s = summarize_counts([apply_policy([True], 40, _BY_NAME["current"]),
                          apply_policy([True, True], 40, _BY_NAME["current"])])
    check(s["zero_moving_edge_pct"] == 50.0, f"零运动边场景占比 50%（实测 {s['zero_moving_edge_pct']}%）")


def main() -> int:
    for fn in (test_a_edge_count_basics, test_b_order_swap_is_the_whole_gain,
               test_c_planb_keeps_static, test_d_lane_cap_isolation,
               test_e_jensen_direction, test_f_edges_and_guards):
        fn()
    n_fail = len(_FAILED)
    print(f"\n{'✅ 全部通过' if n_fail == 0 else f'❌ {n_fail} 条失败'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
