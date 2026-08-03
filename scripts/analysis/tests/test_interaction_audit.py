"""干预审计的协议一致性单测 —— 守的是「读数会被静默毁掉」的那几处。

判据对应协议 `干预协议_INTERVENTION_PROTOCOL_v1_20260731.md`：

* 判据 A（§5）：ego **绝不**进入节点级干预池 —— 2026-07-31 查出旧实现把 `'ego'` 归一成
  `'agent'`，「删掉全部 agent」连自车节点一起删了，20 样本读数因此作废；
* 判据 B（§3）：`count_1` 是「1 个节点」而不是「100%」—— 旧的 `_resolve_budget` 把
  `budget=1` 当比例，这个歧义正好落在协议的 `count_1` 剂量上；
* 判据 C（§3）：`count` 口径两侧个数相同、`fraction` 口径两侧比例相同（个数可以不同）；
* 判据 D：`zero_nodes` 只动指定行且不改原张量；`intervene_remove_nodes` 保持边成对对称
  且不残留被删节点；
* 判据 E（§7）：no-op 干预的 response 恰好为 0。

跑法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest scripts/analysis/tests/test_interaction_audit.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_GNN_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.analysis.d2_semantic_intervention import (  # noqa: E402
    ego_mask_for_graph,
    intervene_remove_nodes,
    node_types_for_graph,
    response,
    split_directed_half,
    zero_nodes,
)
from scripts.analysis.run_interaction_audit import (  # noqa: E402
    DOSES,
    _node_interventions,
    _resolve_dose,
)

#: 一个典型场景的类型表：0 号是 ego，1–3 是 agent，4–7 是 lane。
ID_TO_TYPE = {0: "ego", 1: "agent", 2: "agent", 3: "agent",
              4: "lane", 5: "lane", 6: "lane", 7: "lane"}
ORDER = list(range(8))


def _complete_graph(n: int) -> torch.Tensor:
    """建一个 n 节点的完全图，按「正向半边 + 反向半边」的存储约定拼好。"""
    fwd = torch.tensor([[i, j] for i in range(n) for j in range(i + 1, n)],
                       dtype=torch.long).T
    return torch.cat([fwd, fwd.flip(0)], dim=1)


class _SumModel(torch.nn.Module):
    """把 `gnn_x` 的和广播成 120 维输出 —— 干预改了哪个节点就会体现在输出上。"""

    def forward(self, batch):                       # noqa: D102
        return batch.gnn_x.sum() * torch.ones(1, 120)


def _make_data(n_nodes: int = 8):
    from torch_geometric.data import Data

    edge_index = _complete_graph(n_nodes)
    data = Data(
        gnn_x=torch.arange(1, n_nodes * 9 + 1, dtype=torch.float).reshape(n_nodes, 9),
        gnn_edge_index=edge_index,
        gnn_edge_attr=torch.zeros(edge_index.shape[1]),
    )
    # PyG 的 collate 对含 "index" 的键要按 num_nodes 做偏移；真实 GraphData 有 x/edge_index
    # 可供推断，这里的最小构造必须显式给出，否则 `Batch.from_data_list` 抛 NoneType。
    data.num_nodes = n_nodes
    return data


def test_ego_mask_and_type_normalisation():
    """判据 A 的前半：类型表里 ego 被单独识别，而归一类型仍把它并进 agent。"""
    types = node_types_for_graph(ID_TO_TYPE, ORDER)
    is_ego = ego_mask_for_graph(ID_TO_TYPE, ORDER)
    assert types[0] == "agent", "边级分类仍需把 ego 并进 agent（SEMANTIC_CATEGORIES 只有三类）"
    assert is_ego.tolist() == [True] + [False] * 7
    assert int(is_ego.sum()) == 1


def test_ego_never_intervened():
    """判据 A：任何剂量下，ego 节点（下标 0）都不能被 zero 或 remove 掉。"""
    data = _make_data()
    types = node_types_for_graph(ID_TO_TYPE, ORDER)
    is_ego = ego_mask_for_graph(ID_TO_TYPE, ORDER)
    ego_row = data.gnn_x[0].clone()

    out = _node_interventions(_SumModel(), data, types, is_ego, DOSES,
                              np.random.default_rng(1000), torch.device("cpu"), None)
    # agent 池只剩 3 个（ego 被剔除），饱和剂量必须是 3 而不是 4
    assert out["count_sat"]["k_target"] == 3
    assert out["count_sat"]["frac_target"] == 1.0
    # zero 算子若误删 ego，其特征会变成 0；这里直接核对原张量未被动过
    assert torch.equal(data.gnn_x[0], ego_row)


def test_dose_semantics():
    """判据 B/C：count_1 是 1 个；count 两侧同数；fraction 两侧同比例。"""
    by_name = {d.name: d for d in DOSES}
    n_agent, n_lane = 8, 40

    k_t, k_c, skip = _resolve_dose(by_name["count_1"], n_agent, n_lane)
    assert (k_t, k_c, skip) == (1, 1, None), "count_1 必须是 1 个节点，不是 100%"

    k_t, k_c, skip = _resolve_dose(by_name["count_sat"], n_agent, n_lane)
    assert (k_t, k_c, skip) == (8, 8, None), "饱和剂量取两侧较小值，两侧个数必须相同"

    k_t, k_c, skip = _resolve_dose(by_name["frac_25"], n_agent, n_lane)
    assert (k_t, k_c, skip) == (2, 10, None), "fraction 口径两侧比例相同、个数可以不同"

    # 两类节点缺一不可；比例解析成 0 必须显式 skip 而不是静默当成「干预了没效果」
    assert _resolve_dose(by_name["count_sat"], 0, 40)[2] is not None
    assert _resolve_dose(by_name["frac_25"], 1, 40)[2] is not None


def test_zero_nodes_is_pure():
    """判据 D：只置零指定行，且不修改输入张量。"""
    x = torch.arange(12, dtype=torch.float).reshape(4, 3)
    original = x.clone()
    out = zero_nodes(x, np.array([1, 3]))
    assert torch.equal(x, original), "原张量必须不被修改"
    assert torch.equal(out[1], torch.zeros(3)) and torch.equal(out[3], torch.zeros(3))
    assert torch.equal(out[0], original[0]) and torch.equal(out[2], original[2])


def test_remove_nodes_keeps_symmetry():
    """判据 D：删点后边仍成对对称，且不残留指向被删节点的边。"""
    data = _make_data(6)
    gnn_x, edge_index = intervene_remove_nodes(data.gnn_x, data.gnn_edge_index,
                                               np.array([1, 4]))
    assert gnn_x.shape[0] == 4
    directed, e = split_directed_half(edge_index)      # 不对称会在这里直接抛错
    assert e == 4 * 3 // 2, "剩 4 个节点的完全图应有 6 条无向边"
    assert int(directed.max()) <= 3, "编号必须重排到 0..3"


def test_noop_response_is_exactly_zero():
    """判据 E：同一张图进出两次，位移必须恰好为 0（CPU 上逐位）。"""
    data = _make_data()
    out = response(_SumModel(), data, data.clone(), torch.device("cpu"))
    assert out["mean_disp"] == 0.0 and out["max_disp"] == 0.0


def test_horizon_truncation():
    """协议 §4.2：跨模型比较要能截到公共 horizon。"""
    data = _make_data()
    full = response(_SumModel(), data, data.clone(), torch.device("cpu"))
    cut = response(_SumModel(), data, data.clone(), torch.device("cpu"), horizon_steps=30)
    assert full["mean_disp"] == cut["mean_disp"] == 0.0   # no-op 下两者都为 0
    # 用一个真的有差异的干预来验证截断确实生效
    interv = data.clone()
    interv.gnn_x = zero_nodes(data.gnn_x, np.array([7]))
    r_full = response(_SumModel(), data, interv, torch.device("cpu"))
    r_cut = response(_SumModel(), data, interv, torch.device("cpu"), horizon_steps=10)
    assert r_cut["mean_disp"] < r_full["mean_disp"], "截短后累积位移必须更小"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
