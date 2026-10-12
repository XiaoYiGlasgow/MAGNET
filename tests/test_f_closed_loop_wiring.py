#!/usr/bin/env python
"""F 留车切边 —— 闭环接线等价单测（门⑥；预注册判据 §5「等价单测 + smoke 先行」）。

被测对象：受管文件 `nuplan/planning/simulation/planner/gnn_trans_planner.py` 新增的
`_resolve_feature_contract` / `_apply_edge_mask` / `_check_feature_contract_consistency`。

[A] 契约表锁死：planner 的 full_retention 四项 == `create_full_retention_config()` 的四项
    （planner 不能 import 那个模块 —— 它裸 `import config` —— 所以靠单测保证单一真相）。
[B] 关闭态逐位等价：`edge_mask="none"` 返回**同一对象**（不 clone、边不动）。
[C] a2a mask 与开环审计同构：planner 切出的 `gnn_edge_index` 与 `f_edge_family_audit.cut_edges`
    对同一 fullret val 样本逐位相同；`gnn_x`/`x` 逐位不变；a2a 半边数按族守恒；EA ⊂ 边数关系成立。
[D] sidecar 校验：新档 ckpt sidecar（train_root 含 fullret）配 default 契约 ⇒ strict 下 raise；
    配 full_retention ⇒ 通过；显式 `feature_contract` 键优先于路径推断。

跑法::

    cd /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
    /home/team/anaconda3/envs/nuplan/bin/python tests/test_f_closed_loop_wiring.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
_DEVKIT = _GNN.parent
for _p in (str(_GNN), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VAL = _GNN / "data/processed_features_fullret_L1_logsplit/val"
_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def _planner_module():
    from nuplan.planning.simulation.planner import gnn_trans_planner as m  # noqa: E402
    return m


def _one_fullret_sample():
    """取 fullret val 第一个可用样本（df + 建图），走与 planner 同一 helper。"""
    from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
    for p in sorted(VAL.glob("features_*.pkl"))[:40]:
        df = pd.read_pickle(p)
        data, reason = build_graph_data_from_feature_df(df, "default")
        if data is not None and getattr(data, "gnn_x", None) is not None and data.gnn_x.numel():
            return df, data
    raise RuntimeError(f"{VAL} 下取不到可用样本")


def test_a_contract_table_matches_preprocessing() -> None:
    print("[A 契约表与预处理 twin 逐字相同]")
    m = _planner_module()
    cfg = m._load_gnn_config()
    default = m._resolve_feature_contract(cfg, "default")
    full = m._resolve_feature_contract(cfg, "full_retention")
    check(default["num_agents"] == cfg.NUM_AGENTS and default["lane_num"] == cfg.LANE_NUM
          and default["map_radius"] == float(cfg.MAP_RADIUS) and default["drop_static"] is True,
          f"default 契约 = config.py（agents {default['num_agents']}/LANE {default['lane_num']}/r {default['map_radius']}/drop_static True）")
    # 在**同一进程**里 import 预处理模块（它裸 import config —— 这里 sys.path[0]=GNN_Trans，解析正确）
    from scripts.preprocessing.process_nuplan_to_12d import create_full_retention_config  # noqa: E402
    twin = create_full_retention_config()
    check(full["num_agents"] == twin["num_agents"] == 64, f"num_agents {full['num_agents']} == twin {twin['num_agents']}")
    check(full["lane_num"] == twin["max_elements"]["LANE"] == 68, f"LANE {full['lane_num']} == twin {twin['max_elements']['LANE']}")
    check(full["map_radius"] == float(twin["map_radius"]) == 50.0, f"map_radius {full['map_radius']} == twin {twin['map_radius']}")
    check(full["drop_static"] is False and twin["drop_static"] is False, "drop_static False == twin False")
    raised = False
    try:
        m._resolve_feature_contract(cfg, "bogus")
    except ValueError:
        raised = True
    check(raised, "非法契约名 ⇒ ValueError")


def test_b_none_is_identity() -> None:
    print("[B edge_mask=none 逐位等价（同一对象）]")
    m = _planner_module()
    df, data = _one_fullret_sample()
    out = m._apply_edge_mask(data, df, "none")
    check(out is data, "返回同一对象（未 clone、未重建边）")


def test_c_a2a_mask_matches_audit() -> None:
    print("[C a2a/E/EA mask 与 f_edge_family_audit 同构]")
    m = _planner_module()
    from scripts.analysis.f_edge_family_audit import classify_edges, cut_edges  # noqa: E402
    from scripts.analysis.run_interaction_audit import _node_types_from_df  # noqa: E402
    df, data = _one_fullret_sample()
    node_types, is_ego, _ = _node_types_from_df(df)
    fam = classify_edges(data.gnn_edge_index, node_types, is_ego)
    n_a2a, n_e = int(fam.idx["a2a"].size), int(fam.idx["E"].size)
    m_half = int(fam.directed.shape[1])
    check(n_a2a > 0, f"样本含 a2a 半边 {n_a2a}（E {n_e}，总半边 {m_half}）")
    for name, drop in (("a2a", fam.idx["a2a"]), ("E", fam.idx["E"]),
                       ("EA", np.concatenate([fam.idx["a2a"], fam.idx["E"]]))):
        ref = cut_edges(data, fam, drop)
        out = m._apply_edge_mask(data, df, name)
        check(torch.equal(out.gnn_edge_index, ref.gnn_edge_index),
              f"{name}: gnn_edge_index 与审计 cut_edges 逐位相同（{ref.gnn_edge_index.shape[1]} 条）")
        check(int(out.gnn_edge_index.shape[1]) == 2 * (m_half - int(np.unique(drop).size)),
              f"{name}: 边数守恒 2×({m_half}−{int(np.unique(drop).size)})")
        check(torch.equal(out.gnn_x, data.gnn_x) and torch.equal(out.x, data.x),
              f"{name}: gnn_x / x 逐位不动")
        check(out is not data and torch.equal(data.gnn_edge_index, data.gnn_edge_index),
              f"{name}: 原对象未被就地修改")
    raised = False
    try:
        m._apply_edge_mask(data, df, "bogus")
    except ValueError:
        raised = True
    check(raised, "非法 edge_mask ⇒ ValueError")


def test_d_sidecar_consistency() -> None:
    print("[D 契约 sidecar 校验]")
    m = _planner_module()
    Planner = m.GNNTransPlanner

    def _mk(contract: str, meta: dict):
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model_default.pth")
            open(ckpt, "wb").close()
            with open(os.path.join(d, "train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            p = Planner(checkpoint_path=ckpt, device="cpu", feature_contract=contract)
            try:
                p._check_feature_contract_consistency(os.path.join(d, "train_meta.json"))
                return None
            except ValueError as exc:
                return str(exc)

    fullret_meta = {"route_mode": "none", "dataset_config": "default",
                    "train_root": str(VAL.parent / "train")}
    err = _mk("default", fullret_meta)
    check(err is not None and "full_retention" in err, "fullret sidecar + default 契约 ⇒ raise 且点名 full_retention")
    check(_mk("full_retention", fullret_meta) is None, "fullret sidecar + full_retention 契约 ⇒ 通过")
    old_meta = {"route_mode": "none", "dataset_config": "default",
                "train_root": "/x/processed_features_scaling_L1_logsplit/train"}
    check(_mk("default", old_meta) is None, "旧档 sidecar + default ⇒ 通过（既有 run 不受影响）")
    check(_mk("full_retention", old_meta) is not None, "旧档 sidecar + full_retention ⇒ raise")
    explicit = {**old_meta, "feature_contract": "full_retention"}
    check(_mk("full_retention", explicit) is None, "显式 feature_contract 键优先于 train_root 推断")
    # 默认构造（不传参、无 env）⇒ default/none
    for k in ("MAGNET_FEATURE_CONTRACT", "MAGNET_EDGE_MASK"):
        os.environ.pop(k, None)
    p = Planner(checkpoint_path="/nonexistent.pth", device="cpu")
    check(p._contract["name"] == "default" and p._edge_mask == "none",
          "默认构造 ⇒ feature_contract=default / edge_mask=none（既有 yaml 行为不变）")
    os.environ["MAGNET_FEATURE_CONTRACT"] = "full_retention"
    os.environ["MAGNET_EDGE_MASK"] = "a2a"
    try:
        p2 = Planner(checkpoint_path="/nonexistent.pth", device="cpu")
        check(p2._contract["num_agents"] == 64 and p2._edge_mask == "a2a", "环境变量通道生效（64 / a2a）")
    finally:
        os.environ.pop("MAGNET_FEATURE_CONTRACT", None)
        os.environ.pop("MAGNET_EDGE_MASK", None)


if __name__ == "__main__":
    test_a_contract_table_matches_preprocessing()
    test_b_none_is_identity()
    test_c_a2a_mask_matches_audit()
    test_d_sidecar_consistency()
    if _FAILED:
        print(f"\n🔴 {len(_FAILED)} 项失败:")
        for f in _FAILED:
            print("   -", f)
        sys.exit(1)
    print("\n✅ test_f_closed_loop_wiring 4/4 通过")
