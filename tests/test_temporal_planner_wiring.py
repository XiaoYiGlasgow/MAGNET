#!/usr/bin/env python
"""R1 逐帧图的**部署契约**单测（A4：`tg_x` 前置）。

被测的失效模式（此前无任何防线）：
    `TemporalPredictor._encode_temporal_gnn` 在 `tg_x` 缺失时返回零向量，
    而 planner 的 `_build_graph_data` 只产出 `gnn_x/...` ⇒ 用 R1 checkpoint 跑闭环会
    **静默**变成「GNN 分支恒零」：rc=0、有产物、分数低，与真实结果长得一模一样（第十条）。

判据设计原则（第八十九条：扰动必须能穿透被测组件的不变性）——
本组每条判据都构造出「修好与没修好会给出不同答案」的输入：

* [A] `strict_temporal` 的两态：默认 False 保持训练侧逐位旧行为；True 时必须 **raise**。
  🔴 若把默认写成 True，训练/审计路径会当场崩 ⇒ [A] 的第一条就是防这个。
* [B] 挂上真实逐帧图后前向能跑通，且输出与「零向量兜底」**不同** ——
  否则「挂了 tg_x」这件事没有可观测后果，测了等于没测。
* [C] `AttachTemporalGraph` 的失败模式：它自己会把重建失败降级成空张量
  （训练侧可接受），而空张量在 strict 下必须同样 raise，不能漏网。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

GNN_ROOT = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
DEVKIT = GNN_ROOT.parent
for _p in (str(GNN_ROOT), str(DEVKIT), str(DEVKIT / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
from src.shared.utils.temporal_graph import AttachTemporalGraph  # noqa: E402
from temporal_predictor import TemporalPredictor  # noqa: E402

VAL = GNN_ROOT / "data/processed_features_scaling_L1_logsplit/val"
_FAILED = []


def check(cond: bool, msg: str) -> None:
    print(f"   {'✅' if cond else '🔴'} {msg}")
    if not cond:
        _FAILED.append(msg)


def _one_sample():
    """取 L1 val 第一个可用样本，走与训练/planner 同一条建图 helper。"""
    for p in sorted(VAL.glob("features_*.pkl"))[:20]:
        built = build_graph_data_from_feature_df(pd.read_pickle(p), "default")
        data = built[0] if isinstance(built, tuple) else built
        if data is not None and getattr(data, "gnn_x", None) is not None and data.gnn_x.numel():
            return data
    raise RuntimeError(f"{VAL} 下取不到可用样本")


def test_a_strict_flag_two_states() -> None:
    """[A] `strict_temporal` 两态：默认沉默兜底，开启后硬失败。"""
    print("[A strict_temporal 两态]")
    model = TemporalPredictor().eval()
    data = _one_sample()   # 只有 gnn_x，没有 tg_x —— 正是 planner 此前的状态

    check(not getattr(model, "strict_temporal", False),
          "默认 strict_temporal=False（训练/审计路径逐位不变）")

    with torch.no_grad():
        out_silent = model(data)
    check(tuple(out_silent.shape) == (1, 120), "默认态下仍能前向（返回零向量兜底）")

    model.strict_temporal = True
    raised = False
    try:
        with torch.no_grad():
            model(data)
    except RuntimeError as exc:
        raised = "tg_x" in str(exc)
    check(raised, "strict_temporal=True 且缺 tg_x ⇒ raise 且报文点名 tg_x")


def test_b_attached_graph_changes_output() -> None:
    """[B] 挂上逐帧图后：strict 下不再 raise，且输出与零向量兜底**不同**。"""
    print("[B 挂上 tg_x 后有可观测后果]")
    data = _one_sample()
    model = TemporalPredictor().eval()
    with torch.no_grad():
        out_zero = model(data)          # 兜底路径（GNN 分支恒零）

    attached = AttachTemporalGraph(topology="full")(_one_sample())
    check(getattr(attached, "tg_x", None) is not None and attached.tg_x.numel() > 0,
          f"AttachTemporalGraph 挂上非空 tg_x（{tuple(attached.tg_x.shape)}）")
    check(attached.tg_edge_index.shape[0] == 2 and attached.tg_frame.numel() == attached.tg_x.shape[0],
          "tg_edge_index 是 [2,E]、tg_frame 与 tg_x 行数一一对应")

    model.strict_temporal = True
    with torch.no_grad():
        out_real = model(attached)      # strict 下不该 raise
    check(tuple(out_real.shape) == (1, 120), "挂上后 strict 态可正常前向")
    check(not torch.allclose(out_real, out_zero),
          "输出与零向量兜底不同 ⇒『挂没挂 tg_x』是可观测的（否则闭环差异测不出来）")


def test_c_empty_rebuild_is_not_silently_accepted() -> None:
    """[C] 🔴 命门：重建失败被降级成空张量时，strict 下必须同样 raise。"""
    print("[C 空张量不得漏网]")
    data = _one_sample()
    # 模拟 AttachTemporalGraph 的降级分支（它对 TemporalGraphError 只 warning + 空张量）
    data.tg_x = torch.zeros(0, 9, dtype=torch.float32)
    data.tg_edge_index = torch.zeros(2, 0, dtype=torch.long)
    data.tg_frame = torch.zeros(0, dtype=torch.long)

    model = TemporalPredictor().eval()
    model.strict_temporal = True
    raised = False
    try:
        with torch.no_grad():
            model(data)
    except RuntimeError as exc:
        raised = "numel=0" in str(exc) or "tg_x" in str(exc)
    check(raised, "tg_x 存在但为空 ⇒ 同样 raise（不能只查 None）")

    model.strict_temporal = False
    with torch.no_grad():
        out = model(data)
    check(tuple(out.shape) == (1, 120), "非 strict 态下空张量仍走兜底（训练侧行为不变）")


def test_d_actual_planner_path_via_batch() -> None:
    """[D] 🔴 走 planner **真正的**那条路：`Batch.from_data_list([data])`。

    [B] 喂的是裸 `Data`，而 `compute_planner_trajectory` 喂的是 `Batch`。
    两者不等价：`Batch` 会建出 `data.batch`、改变 `num_graphs`，
    且 PyG 默认对键名含 `index` 的属性按 `num_nodes` 做偏移
    —— 而本 `Data` 的 `num_nodes` 由 `x`(2017 行) 决定、不是 `tg_x`(817 行)。
    batch=1 时偏移量恒为 0 所以安全，但这件事**必须有测试钉住**，
    否则哪天有人把 planner 改成批量推理就会静默错图。
    """
    print("[D planner 真实路径（Batch 包装）]")
    from torch_geometric.data import Batch  # noqa: WPS433

    data = AttachTemporalGraph(topology="full")(_one_sample())
    n_tg = int(data.tg_x.shape[0])
    max_before = int(data.tg_edge_index.max())

    batch = Batch.from_data_list([data])          # 与 compute_planner_trajectory 逐字同构
    check(int(batch.tg_edge_index.max()) == max_before,
          f"batch=1 下 tg_edge_index 未被偏移（max={max_before}，仍 < tg_x 行数 {n_tg}）")
    check(not hasattr(batch, "tg_x_batch"),
          "未传 follow_batch ⇒ 无 tg_x_batch ⇒ 模型走单样本分支（planner 就是这条）")

    model = TemporalPredictor().eval()
    model.strict_temporal = True
    with torch.no_grad():
        out_batched = model(batch)
    check(tuple(out_batched.shape) == (1, 120), "strict 态下 Batch 路径可正常前向")

    bare = TemporalPredictor().eval()             # 同一份随机初始化不可复用，故只比「是否恒零兜底」
    bare.load_state_dict(model.state_dict())
    with torch.no_grad():
        out_zero = bare(Batch.from_data_list([_one_sample()]))   # 没挂 tg_x ⇒ 兜底
    check(not torch.allclose(out_batched, out_zero),
          "Batch 路径下『挂没挂 tg_x』同样可观测（这才是闭环真正跑的路径）")


def main() -> int:
    for fn in (test_a_strict_flag_two_states, test_b_attached_graph_changes_output,
               test_c_empty_rebuild_is_not_silently_accepted,
               test_d_actual_planner_path_via_batch):
        fn()
    n_fail = len(_FAILED)
    print(f"\n{'✅ 全部通过' if n_fail == 0 else f'❌ {n_fail} 条失败'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
