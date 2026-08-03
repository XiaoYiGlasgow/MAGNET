"""Q32 —— 量化「建图前的 agent 过滤漏斗」到底削掉多少交互输入。

背景
----
G19 干预审计查出 **29.5% 的场景图里一个他车都没有**（≤1 个占 49.5%），
而 `agent_past_process` 明明按距离取了**最近 20 个**对象。中间差掉的量必须查清 ——
若「交互建模」的输入端本身就被削掉大半，那 C1 测到的「模型不依赖 agent」就有一部分
不是模型的问题，而是**数据管线的问题**，两者在写作上必须分开归因。

漏斗的三道闸（`utils/data_utils.py`，已用 `inspect.getfile` 确认为生效副本）
------------------------------------------------------------------------
1. `agent_past_process:181-188` —— `agents = np.zeros((num_agents=20, T, D))`，
   按 `distance_to_ego` 升序只填最近 20 个；不足 20 个时余下槽位保持**全零**。
   ⚠️ 实测 val 3000 场景：留下与删掉的对象 **100% 都是 vehicle**，一个 pedestrian 都没有
   ⇒ 上游 `filter_agents_tensor` 已经把非车辆滤干净了，「行人抢槽位」这个担心不成立。
2. `encoding_vectornet_features:854` —— `np.all(agent_hist == 0)` 丢掉上面的全零占位。
3. `encoding_vectornet_features:850,867` —— `avg_speed < 0.01 m/s` 丢掉「静止」对象。
   ⚠️ 0.01 m/s = 1 cm/s，等红灯的车、路边停放的车 **全部**落在这道闸里。

本脚本对 val 全量逐场景还原这三道闸，输出各级留存量与类型构成，
并按 ego 当前速度分层 —— 用于检验一个具体假设：
**ego 慢/停时图里恰好没有他车**（因为周围的车也停着），
即数据管线在「最需要交互证据的时刻」把交互证据删干净了。

口径
----
* 只读已落盘的 `features_*.pkl`，**不重跑预处理**，因此与训练所见完全一致；
* 速度阈值与列索引直接从 `data_utils` 的生效副本对齐，不写死第二份；
* 类型 one-hot 在 `AGENT_HISTORIES` 的最后 3 维（`agent_past_process:189-195`）。
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 与 utils/data_utils.py:850 保持一致；改动必须两边同步
SPEED_THRESHOLD_MPS = 0.01
# AGENT_HISTORIES 的列语义（agent_past_process 打包顺序）
COL_VX, COL_VY = 3, 4
N_TYPE_COLS = 3          # 末 3 维 one-hot: [vehicle, pedestrian, other]
TYPE_NAMES = ("vehicle", "pedestrian", "other")


def _avg_speed(agent_hist: np.ndarray) -> float:
    """复刻 `encoding_vectornet_features:862-864` 的平均速度定义（逐段中点平均）。"""
    speed = np.sqrt(agent_hist[:, COL_VX] ** 2 + agent_hist[:, COL_VY] ** 2)
    if speed.shape[0] < 2:
        return 0.0
    return float(np.mean((speed[:-1] + speed[1:]) / 2))


def _agent_type(agent_hist: np.ndarray) -> str:
    """读末 3 维 one-hot。全零（未命中任何类型）记为 `unknown`。"""
    onehot = agent_hist[-1, -N_TYPE_COLS:]
    if not np.any(onehot):
        return "unknown"
    return TYPE_NAMES[int(np.argmax(onehot))]


def _ego_speed(row) -> float:
    """ego 当前速度 = ego 子图最后一行的 `v_end` 列。

    ⚠️ 不要用 `EGO_HISTORY` 的位移去估 —— 那个字段只有 x/y 且原点约定不统一，
    实测会把常速场景误判成静止。ego 子图的第 8 列是 `encoding_vectornet_features:834`
    显式写入的 `speed[1:]`，是唯一权威口径。
    """
    poly = np.asarray(row["POLYLINE_FEATURES"])
    start, end = row["TRAJ_ID_TO_INDICES"][0]          # ego 恒为 subgraph_ID 0
    return float(poly[end, 8])


def audit_one(pkl_path: Path) -> Optional[Dict]:
    """还原单个场景的三道闸。返回 None 表示该文件不可用（不静默计入分母）。"""
    try:
        df = pickle.load(open(pkl_path, "rb"))
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        logger.warning("读不了 %s: %s", pkl_path.name, exc)
        return None

    row = df.iloc[0]
    hist = np.asarray(row["AGENT_HISTORIES"])          # [20, T, 11]
    n_slots = hist.shape[0]

    kept_types: List[str] = []
    dropped_types: List[str] = []
    n_padding = 0
    for i in range(n_slots):
        a = hist[i]
        # 闸 2：全零占位（= 附近真实对象不足 20 个）
        if a.shape[0] < 2 or not np.any(a):
            n_padding += 1
            continue
        # 闸 3：静止过滤
        if _avg_speed(a) < SPEED_THRESHOLD_MPS:
            dropped_types.append(_agent_type(a))
        else:
            kept_types.append(_agent_type(a))

    # 交叉校验：图里实际的 agent 子图数应等于 kept 数
    type_map = row["SUBGRAPH_ID_TO_TYPE"]
    n_graph_agents = sum(1 for v in type_map.values() if v == "agent")

    return {
        "token": pkl_path.stem.replace("features_", ""),
        "n_slots": n_slots,
        "n_padding": n_padding,
        "n_real": n_slots - n_padding,
        "n_kept": len(kept_types),
        "n_dropped_static": len(dropped_types),
        "n_graph_agents": n_graph_agents,
        "kept_types": Counter(kept_types),
        "dropped_types": Counter(dropped_types),
        "ego_speed": _ego_speed(row),
    }


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def summarize(rows: List[Dict]) -> Dict:
    """汇总三道闸的留存率、类型构成，并按 ego 速度分层。"""
    n = len(rows)
    real = np.array([r["n_real"] for r in rows])
    kept = np.array([r["n_kept"] for r in rows])
    drop = np.array([r["n_dropped_static"] for r in rows])
    graph = np.array([r["n_graph_agents"] for r in rows])
    ego_v = np.array([r["ego_speed"] for r in rows])

    kept_t: Counter = Counter()
    drop_t: Counter = Counter()
    for r in rows:
        kept_t.update(r["kept_types"])
        drop_t.update(r["dropped_types"])

    # 一致性检查：kept 应与图里的 agent 子图数逐场景相等
    mismatch = int(np.sum(kept != graph))

    # 按 ego 速度分层（m/s）：停 / 极慢 / 慢 / 常速
    bins: List[Tuple[str, float, float]] = [
        ("ego 停 (<0.1)", -np.inf, 0.1),
        ("ego 极慢 [0.1,1)", 0.1, 1.0),
        ("ego 慢 [1,3)", 1.0, 3.0),
        ("ego 常速 >=3", 3.0, np.inf),
    ]
    by_speed = []
    for name, lo, hi in bins:
        m = (ego_v >= lo) & (ego_v < hi)
        if not m.any():
            continue
        by_speed.append({
            "bin": name,
            "n_scenes": int(m.sum()),
            "share_of_val_pct": round(_pct(int(m.sum()), n), 2),
            "mean_real": round(float(real[m].mean()), 2),
            "mean_kept": round(float(kept[m].mean()), 2),
            "mean_dropped_static": round(float(drop[m].mean()), 2),
            "empty_graph_pct": round(_pct(int((kept[m] == 0).sum()), int(m.sum())), 2),
            "le1_agent_pct": round(_pct(int((kept[m] <= 1).sum()), int(m.sum())), 2),
        })

    return {
        "n_scenes": n,
        "speed_threshold_mps": SPEED_THRESHOLD_MPS,
        "slots_per_scene": int(rows[0]["n_slots"]) if rows else 0,
        "funnel": {
            "闸1_取最近N个对象": int(rows[0]["n_slots"]) if rows else 0,
            "闸2后_真实对象_均值": round(float(real.mean()), 2),
            "闸2后_真实对象_中位数": float(np.median(real)),
            "闸3后_入图agent_均值": round(float(kept.mean()), 2),
            "闸3后_入图agent_中位数": float(np.median(kept)),
            "被静止过滤删除_均值": round(float(drop.mean()), 2),
            "静止过滤删除占真实对象比_pct": round(_pct(int(drop.sum()), int(real.sum())), 2),
        },
        "empty_graph_pct": round(_pct(int((kept == 0).sum()), n), 2),
        "le1_agent_pct": round(_pct(int((kept <= 1).sum()), n), 2),
        "kept_type_mix": {k: [v, round(_pct(v, sum(kept_t.values())), 2)] for k, v in kept_t.most_common()},
        "dropped_type_mix": {k: [v, round(_pct(v, sum(drop_t.values())), 2)] for k, v in drop_t.most_common()},
        "consistency_kept_vs_graph_mismatch": mismatch,
        "by_ego_speed": by_speed,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Q32：量化建图前 agent 过滤漏斗")
    p.add_argument("--val_root", required=True, help="含 features_*.pkl 的目录（绝对路径）")
    p.add_argument("--out", required=True, help="汇总 JSON 输出路径（绝对路径）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个，0=全量")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted(Path(a.val_root).glob("features_*.pkl"))
    if a.limit:
        files = files[: a.limit]
    logger.info("待审计场景 %d 个 <- %s", len(files), a.val_root)

    rows = []
    for i, f in enumerate(files):
        r = audit_one(f)
        if r is not None:
            rows.append(r)
        if (i + 1) % 500 == 0:
            logger.info("  已处理 %d/%d", i + 1, len(files))

    if not rows:
        raise SystemExit("没有任何可用场景，检查 --val_root")

    summary = summarize(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("写出 %s", a.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
