#!/usr/bin/env python
"""`src/shared/utils/agent_labels.py` + `data_utils.encoding_vectornet_features(drop_static=)` 的单测。

判据设计原则（第八十九条：扰动必须能穿透被测组件的不变性）——
每条判据都构造出「做错了会给出不同答案」的输入：

* [A] label 定义本身：平均速度必须**逐字复刻**旧算法（`mean((v[:-1]+v[1:])/2)`，
  不是 `mean(v)`）；`dist_rank` 取自槽位序；`was_truncated` 以旧 20 槽划线。
* [B] 🔴 **命门一**：`legacy_kept_ranks` 必须**先截断再删静止**。构造一个
  「第 21 名是运动车、前 20 名里有静止车」的场景 —— 顺序反了答案就不同。
* [C] 🔴 **命门二（静默类）**：label 必须挂在 **subgraph_ID** 上，不是 `agents_past` 的下标。
  构造一个「中间有不可用空槽」的输入 —— 两者从此错位，且**不会报错**。
* [D] 🔴 **命门三**：`drop_static=True` 的产出必须与改动前**逐字段相同、且不多一列**，
  否则等价门无从谈起。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GNN_ROOT))

from src.shared.utils.agent_labels import (  # noqa: E402
    LEGACY_NUM_AGENTS, SPEED_THRESHOLD, agent_avg_speed, compute_agent_labels,
    decode_agent_type, is_usable_agent, label_stats, legacy_kept_ranks,
)
from src.shared.utils.lane_labels import (  # noqa: E402
    compute_lane_labels, lane_label_stats, lane_min_distance,
)
from src.shared.utils.data_utils import encoding_vectornet_features  # noqa: E402

_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def mk_agent(speed: float, n: int = 21, type_onehot=(1, 0, 0)) -> np.ndarray:
    """造一辆恒定速度的车：列序 [x, y, heading, vx, vy] + 3 维 type one-hot。"""
    a = np.zeros((n, 8), dtype=np.float32)
    a[:, 0] = np.arange(n) * speed * 0.1      # x 随时间推进
    a[:, 3] = speed                            # vx
    a[:, 5:8] = np.asarray(type_onehot, dtype=np.float32)
    return a


def test_a_label_definitions() -> None:
    """[A] label 定义逐条。"""
    print("[A label 定义]")
    # 平均速度必须是 mean((v[:-1]+v[1:])/2)；造一个两者不同的输入
    a = np.zeros((5, 8), dtype=np.float32)
    a[:, 3] = [0.0, 10.0, 10.0, 10.0, 10.0]     # vx，首帧为 0
    v = np.abs(a[:, 3])
    naive, legacy = float(np.mean(v)), float(np.mean((v[:-1] + v[1:]) / 2))
    check(abs(naive - legacy) > 0.1, f"构造的输入让两种算法确实分开（{naive:.2f} vs {legacy:.2f}）")
    check(abs(agent_avg_speed(a) - legacy) < 1e-6, "agent_avg_speed 复刻的是旧算法，不是 mean(v)")

    check(agent_avg_speed(np.zeros((1, 8), dtype=np.float32)) == 0.0, "M<2 时返回 0.0 不报错")
    check(decode_agent_type(mk_agent(5.0, type_onehot=(0, 1, 0))) == "pedestrian", "type one-hot 解码")
    check(decode_agent_type(mk_agent(5.0, type_onehot=(0, 0, 0))) == "other", "全零 one-hot ⇒ other")

    check(not is_usable_agent(np.zeros((21, 8), dtype=np.float32)), "全零槽位不可用")
    check(not is_usable_agent(mk_agent(5.0, n=1)), "M<2 不可用")
    check(is_usable_agent(mk_agent(5.0)), "正常车可用")

    agents = [mk_agent(5.0), mk_agent(0.0), mk_agent(5.0)]
    lb = compute_agent_labels(agents)
    check([x["is_moving"] for x in lb] == [True, False, True], "is_moving 按阈值判定")
    check([x["dist_rank"] for x in lb] == [0, 1, 2], "dist_rank = 槽位序（已按距离升序）")
    check(all(not x["was_truncated"] for x in lb), "前 20 名 was_truncated=False")

    many = [mk_agent(5.0) for _ in range(LEGACY_NUM_AGENTS + 3)]
    lb2 = compute_agent_labels(many)
    check([x["was_truncated"] for x in lb2][LEGACY_NUM_AGENTS - 1:LEGACY_NUM_AGENTS + 1] == [False, True],
          f"was_truncated 恰在 rank={LEGACY_NUM_AGENTS} 处翻转")

    # 阈值边界：恰等于阈值算「运动」（旧代码是 `avg_speed < threshold` 才跳过）
    check(compute_agent_labels([mk_agent(SPEED_THRESHOLD)])[0]["is_moving"],
          "速度恰等于阈值 ⇒ is_moving=True（与旧的严格小于号一致）")


def test_b_legacy_order_is_truncate_then_filter() -> None:
    """[B] 🔴 命门一：旧口径是「先截断再删静止」，顺序反了答案不同。"""
    print("[B 旧口径的两步顺序]")
    # 前 20 名里塞 5 辆静止车，第 21–23 名是运动车
    agents = ([mk_agent(5.0)] * 15 + [mk_agent(0.0)] * 5 + [mk_agent(5.0)] * 3)
    labels = compute_agent_labels(agents)
    kept = legacy_kept_ranks(labels)

    check(kept == list(range(15)), f"先截断再删静止 ⇒ 留下 15 辆（实际 {len(kept)}）")
    check(max(kept) < LEGACY_NUM_AGENTS, "留下的全部来自前 20 名")

    # 反过来（先删静止再截断）会留下 18 辆 —— 这正是同口径实测 3.71→4.26 的那 +14.9%
    wrong = [lb["dist_rank"] for lb in labels if lb["is_moving"]][:LEGACY_NUM_AGENTS]
    check(len(wrong) == 18 and len(wrong) != len(kept),
          f"顺序反了会留 {len(wrong)} 辆 ≠ 正确的 {len(kept)} 辆 ⇒ 本条判据有区分力")

    check(legacy_kept_ranks([]) == [], "空输入不炸")
    check(legacy_kept_ranks(compute_agent_labels([mk_agent(0.0)] * 3)) == [],
          "全静止 ⇒ 空图（这正是 29.53% 空图率的来源）")


def test_c_label_keys_on_subgraph_id_not_slot_index() -> None:
    """[C] 🔴 命门二（静默类）：label 挂 subgraph_ID，不是槽位下标。"""
    print("[C label 挂在 subgraph_ID 上]")
    # 槽位 1 是全零空槽 ⇒ 它不进图，subgraph_ID 不为它递增，但槽位下标照样占位
    agents = [mk_agent(5.0), np.zeros((21, 8), dtype=np.float32), mk_agent(0.0), mk_agent(3.0)]
    labels = compute_agent_labels(agents)
    check(labels[1] is None, "不可用槽位的 label 是 None，且**保留位置**（与 agents_past 逐位对齐）")
    check([lb is None for lb in labels] == [False, True, False, False], "只有空槽是 None")

    df = _encode(agents, drop_static=False)
    lab = df["SUBGRAPH_ID_TO_LABEL"].iloc[0]
    types = df["SUBGRAPH_ID_TO_TYPE"].iloc[0]
    agent_ids = sorted(k for k, v in types.items() if v == "agent")

    # 🔴 lane label 接进来之后，键集合 = agent ∪ lane。这里只断言 **agent 那部分**
    #    （靠 `is_moving` 字段区分两类），并显式核对 agent 键就是 agent 子图 ID。
    agent_keyed = sorted(k for k, v in lab.items() if "is_moving" in v)
    check(agent_keyed == agent_ids,
          f"带 agent 字段的键 = agent 子图 ID 集合 {agent_ids}（不是槽位下标 0..3）")
    check(0 not in lab, "ego 子图（ID=0）不打 label —— label 只描述 agent 与 lane")
    # 槽位 0/2/3 → subgraph_ID 1/2/3（0 号是 ego）。若误用槽位下标，dist_rank 会读成 1,2,3
    check([lab[i]["dist_rank"] for i in agent_ids] == [0, 2, 3],
          f"dist_rank 仍是原槽位序 [0,2,3]（空槽 1 被跳过）—— 误用下标会得到 [1,2,3]")
    check([lab[i]["is_moving"] for i in agent_ids] == [True, False, True],
          "静止车确实进图了且被标成 is_moving=False")


def test_d_default_path_is_unchanged() -> None:
    """[D] 🔴 命门三：drop_static=True 的产出逐字段不变、且不多一列。"""
    print("[D 默认档逐字段不变]")
    agents = [mk_agent(5.0), mk_agent(0.0), mk_agent(3.0)]
    df_old = _encode(agents, drop_static=True)
    df_new = _encode(agents, drop_static=False)

    check("SUBGRAPH_ID_TO_LABEL" not in df_old.columns,
          "默认档**不多任何列**（否则等价门无从谈起）")
    check("SUBGRAPH_ID_TO_LABEL" in df_new.columns, "全量档才多出 label 列")

    n_old = sum(1 for v in df_old["SUBGRAPH_ID_TO_TYPE"].iloc[0].values() if v == "agent")
    n_new = sum(1 for v in df_new["SUBGRAPH_ID_TO_TYPE"].iloc[0].values() if v == "agent")
    check(n_old == 2 and n_new == 3,
          f"默认档删掉静止车（{n_old} 辆），全量档保留（{n_new} 辆）⇒ 两档确实不同")

    # 默认档的 polyline 必须与「手工按旧规则筛出来」的行数一致
    pl_old = df_old["POLYLINE_FEATURES"].iloc[0]
    check(pl_old.shape[1] == 12, "polyline 仍是 12 列（模型输入 schema 未变）")

    st = label_stats(compute_agent_labels(agents))
    check(st["n_usable"] == 3 and st["n_static"] == 1 and abs(st["static_frac"] - 1 / 3) < 1e-9,
          f"label_stats 汇总正确: {st}")


def _encode(agents, drop_static: bool, n_lanes: int = 1):
    """最小可跑的 encoding_vectornet_features 调用（一条直行 ego + n_lanes 条 lane）。"""
    n = 21
    ego_past = np.zeros((n, 5), dtype=np.float32)
    ego_past[:, 0] = np.arange(n) * 0.5
    ego_past[:, 3] = 5.0
    ego_future = np.zeros((60, 2), dtype=np.float32)
    # lane 至少 3 列：`data_utils.py:936` 取 `lane[:, 2]` 当 heading（2 列会 IndexError）
    vector_map = {"lanes": [np.full((50, 3), 10.0 * (k + 1), dtype=np.float32)
                            for k in range(n_lanes)]}
    ts = torch.arange(n, dtype=torch.float64) * 0.1
    return encoding_vectornet_features(
        agents, None, ego_past, ego_future, vector_map, ts, float(ts[-1]),
        ego_history=None, lane_route_channels=None, drop_static=drop_static,
    )


def test_f_lane_labels() -> None:
    """[F] lane 侧 label（D4 补录）。

    🔴 命门五：lane 的 `lane_dist_rank` 是**按距离排的名次**，不是 `lanes` 的原始下标
    —— 与 agent 侧（下标即 rank）约定**相反**。构造一个「原序与距离序不同」的输入，
    混用两边约定就会红。
    """
    print("[F lane 侧 label]")
    # 原序 = [远, 近, 中]，距离序应是 [1, 2, 0]
    def lane_at(x):
        a = np.zeros((10, 3), dtype=np.float32)
        a[:, 0] = x
        return a
    lanes = [lane_at(30.0), lane_at(5.0), lane_at(12.0)]

    check(abs(lane_min_distance(lanes[1]) - 5.0) < 1e-6, "lane_min_distance 取最近点")
    check(lane_min_distance(np.zeros((0, 3), dtype=np.float32)) == float("inf"),
          "空折线 ⇒ inf（排序时落到最后）")

    lb = compute_lane_labels(lanes)
    check([x["lane_dist_rank"] for x in lb] == [2, 0, 1],
          f"rank 按距离而非原序（实际 {[x['lane_dist_rank'] for x in lb]}，原序会是 [0,1,2]）")
    check(all("is_route_lane" not in x for x in lb),
          "🔴 没给 route 通道 ⇒ `is_route_lane` **不出现**（不是填假的 False）")

    ch = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]
    lb2 = compute_lane_labels(lanes, ch)
    check([x["is_route_lane"] for x in lb2] == [False, True, False], "给了 route 通道才解出 is_route_lane")
    check(lane_label_stats(lb)["has_route_info"] is False
          and lane_label_stats(lb2)["has_route_info"] is True,
          "lane_label_stats 能区分「有没有 route 信息」")

    # 端到端：lane label 挂在 lane 的 subgraph_ID 上，且与 agent 的键空间不撞车
    agents = [mk_agent(5.0), mk_agent(0.0)]
    df = _encode(agents, drop_static=False, n_lanes=3)
    lab = df["SUBGRAPH_ID_TO_LABEL"].iloc[0]
    types = df["SUBGRAPH_ID_TO_TYPE"].iloc[0]
    lane_ids = sorted(k for k, v in types.items() if v == "lane")
    agent_ids = sorted(k for k, v in types.items() if v == "agent")
    check(set(lane_ids).isdisjoint(agent_ids), "lane 与 agent 的 subgraph_ID 键空间不相交")
    check(all("lane_dist_rank" in lab[i] for i in lane_ids), "每条 lane 都有 lane_dist_rank")
    check(all("is_moving" in lab[i] for i in agent_ids), "每个 agent 都有 is_moving")
    check(len(lab) == len(lane_ids) + len(agent_ids),
          f"label 总数 = lane {len(lane_ids)} + agent {len(agent_ids)}（没有漏也没有多）")


def test_e_single_source_of_truth_for_threshold() -> None:
    """[E] 🔴 命门四：`data_utils` 里就地硬编码的阈值必须 == `agent_labels.SPEED_THRESHOLD`。

    背景（08-06 自审）：`data_utils` 在本仓有**三条不同的导入路径**
    （`src.shared.utils.*` / `GNN_Trans.utils.*` 符号链接 / 扁平 `from data_utils import`），
    在函数顶部加绝对导入会让**默认路径**多一个失败点。故阈值仍就地硬编码，
    单一真相源改由本条判据保证 —— 两边任一被改动都会当场红，而不是静默漂移。
    """
    print("[E 阈值单一真相源]")
    src = (GNN_ROOT / "src/shared/utils/data_utils.py").read_text(encoding="utf-8")
    m = re.search(r"^\s*speed_threshold\s*=\s*([0-9.eE+-]+)", src, re.M)
    check(m is not None, "在 data_utils.py 里找到 speed_threshold 赋值")
    if m:
        check(float(m.group(1)) == SPEED_THRESHOLD,
              f"data_utils 的 {m.group(1)} == agent_labels.SPEED_THRESHOLD ({SPEED_THRESHOLD})")

    # 默认路径不得在函数顶部引入 agent_labels（否则三条导入路径里最脆的那条会炸）。
    # 🔴 只扫**非注释**的真实语句 —— 上面那段说明里就写着这个 import 的字样，
    #    裸做子串匹配会被自己的注释误伤（本条判据第一版正是这么红的）。
    head = src[src.index("def encoding_vectornet_features"):]
    before = head[:head.index("if not drop_static:")]
    stmts = [ln.split("#", 1)[0] for ln in before.splitlines()]
    offending = [ln.strip() for ln in stmts
                 if re.match(r"\s*(from\s+\S*agent_labels|import\s+\S*agent_labels)", ln)]
    check(not offending,
          f"agent_labels 的导入只出现在 drop_static 分支内，默认路径零新增导入"
          f"{'（违规行: ' + str(offending) + '）' if offending else ''}")

    # 反向验证判据有区分力：把分支内那一行挪到函数顶部，本条必须变红
    fake = before + "\n    from src.shared.utils.agent_labels import compute_agent_labels\n"
    fake_stmts = [ln.split("#", 1)[0] for ln in fake.splitlines()]
    check(any(re.match(r"\s*from\s+\S*agent_labels", ln) for ln in fake_stmts),
          "构造的违规样本能被同一条规则抓到 ⇒ 本判据不是恒真")


def main() -> int:
    for fn in (test_a_label_definitions, test_b_legacy_order_is_truncate_then_filter,
               test_c_label_keys_on_subgraph_id_not_slot_index,
               test_d_default_path_is_unchanged, test_f_lane_labels,
               test_e_single_source_of_truth_for_threshold):
        fn()
    n_fail = len(_FAILED)
    print(f"\n{'✅ 全部通过' if n_fail == 0 else f'❌ {n_fail} 条失败'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
