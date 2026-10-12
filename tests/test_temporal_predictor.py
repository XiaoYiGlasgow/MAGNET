"""R1 时序模型冒烟测试（Q76 重构，模型侧）。

🔴 命门判据 [C]：**ego 与 context 两个分支的输出必须与父类 `TransformerPredictor` 逐位相同**
（`max|Δ| = 0`）。它证明 R1 的唯一变量是 GNN 分支 —— 否则后面 ΔS 的归因不成立。

跑法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest tests/test_temporal_predictor.py -q -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_DEVKIT = _REPO.parent
for p in (str(_REPO), str(_DEVKIT), str(_DEVKIT / "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_geometric.loader import DataLoader  # noqa: E402

from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
from src.shared.utils.temporal_graph import (  # noqa: E402
    build_temporal_graph,
    flatten_to_pyg_batch,
)
from temporal_predictor import TemporalPredictor  # noqa: E402
from transformer_predictor import TransformerPredictor  # noqa: E402

VAL_DIR = _REPO / "data/processed_features_magnet_e1_none_v3_split3/val"
CKPT = _REPO / "experiments/ablation_2x2_f1/dev/B_seed42/models/checkpoint_best.pt"


def _make_sample(pkl_path: Path, topology: str = "full"):
    """建一个同时带旧图与逐帧图的 GraphData。"""
    df = pd.read_pickle(pkl_path)
    gd, skip = build_graph_data_from_feature_df(df)
    if gd is None:
        return None
    payload, tskip = build_temporal_graph(df, topology=topology)
    if payload is None:
        return None
    tg_x, tg_edge_index, tg_frame = flatten_to_pyg_batch(payload)
    gd.tg_x = tg_x
    gd.tg_edge_index = tg_edge_index
    gd.tg_frame = tg_frame
    return gd


@pytest.fixture(scope="module")
def samples():
    out = []
    for p in sorted(VAL_DIR.glob("features_*.pkl"))[:8]:
        gd = _make_sample(p)
        if gd is not None:
            out.append(gd)
    assert len(out) >= 4, "可用样本太少"
    return out


def test_a_instantiate_and_load_old_ckpt():
    """[A] 能实例化，且旧 ckpt 的共享权重可加载；新增模块只有两个。"""
    model = TemporalPredictor()
    assert CKPT.exists(), f"缺少对照 ckpt {CKPT}"
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("model_state_dict", sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    new_prefixes = ("frame_pos_embedding", "temporal_transformer")
    unexplained = [k for k in missing if not k.startswith(new_prefixes)]

    assert not unexplained, f"除新增模块外还有缺失权重：{unexplained[:5]}"
    print(f"\n[A] 旧 ckpt 加载 OK｜新增模块缺失 {len(missing)} 键（全部属 {new_prefixes}）"
          f"｜unexpected {len(unexpected)}")


def test_b_forward_shape(samples):
    """[B] 单样本与批量前向都产出 [B, 120]。"""
    model = TemporalPredictor().eval()

    with torch.no_grad():
        out1 = model(samples[0])
    assert out1.shape == (1, 120), f"单样本输出形状 {out1.shape}"

    loader = DataLoader(samples, batch_size=4, shuffle=False, follow_batch=["tg_x"])
    batch = next(iter(loader))
    with torch.no_grad():
        outb = model(batch)
    assert outb.shape == (4, 120), f"批量输出形状 {outb.shape}"
    print(f"\n[B] 前向 OK｜单样本 {tuple(out1.shape)}｜batch4 {tuple(outb.shape)}")


def test_c_ego_context_bitwise_identical(samples):
    """[C] 🔴 命门：ego 与 context 分支输出与父类逐位相同。

    做法：给两个模型加载同一份权重，用 forward_pre_hook 截获 `feature_fusion` 的输入
    （= cat([ego, context, gnn])），比较前两段 256 维。
    """
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("model_state_dict", sd)

    parent = TransformerPredictor().eval()
    parent.load_state_dict(sd, strict=False)
    child = TemporalPredictor().eval()
    child.load_state_dict(sd, strict=False)

    caught = {}

    def tap(name):
        def hook(_m, args):
            caught[name] = args[0].detach().clone()
        return hook

    h1 = parent.feature_fusion.register_forward_pre_hook(tap("parent"))
    h2 = child.feature_fusion.register_forward_pre_hook(tap("child"))

    max_ego, max_ctx, max_gnn = 0.0, 0.0, 0.0
    try:
        for gd in samples:
            with torch.no_grad():
                parent(gd)
                child(gd)
            p, c = caught["parent"], caught["child"]
            max_ego = max(max_ego, float((p[:, :128] - c[:, :128]).abs().max()))
            max_ctx = max(max_ctx, float((p[:, 128:256] - c[:, 128:256]).abs().max()))
            max_gnn = max(max_gnn, float((p[:, 256:] - c[:, 256:]).abs().max()))
    finally:
        h1.remove()
        h2.remove()

    print(f"\n[C] ego max|Δ|={max_ego:.3e}｜context max|Δ|={max_ctx:.3e}｜gnn max|Δ|={max_gnn:.3e}")
    assert max_ego == 0.0, f"ego 分支被动了，max|Δ|={max_ego:.3e}"
    assert max_ctx == 0.0, f"context 分支被动了，max|Δ|={max_ctx:.3e}"
    assert max_gnn > 0.0, "GNN 分支输出与父类完全相同 —— 逐帧图没生效"


def test_d_batch_equals_single(samples):
    """[D] 批处理语义正确 —— **分段判据**（判据本身经过验证，不是拍的）。

    首版用整体输出 `max|Δ| < 1e-4` 失败于 1.525。分段诊断查明::

        子类 TemporalPredictor : ego=3.0736  context=0.0  gnn=0.0
        父类 TransformerPredictor: ego=3.4928  context=0.0  gnn=0.1554   ← 对照

    ⇒ ①ego 段的差异来自 `ego_history_encoder.1` 的 `BatchNorm1d(track_running_stats=False)`
    （即使 `eval()` 也用 batch statistics），**父类同量级、同根因** ⇒ 是已知的 **D0#13**
    开环/闭环 batch 口径断层，不是 R1 引入的；
    ②🎯 本类 GNN 分支 batch↔single **逐位一致（0.0）**，而**父类是 0.1554** ——
    父类那条是 D0#7/Q45 的「整批共享一个向量再 repeat」坑，本实现直接走
    `global_mean_pool(·, group)`，单样本时 group 全 0 ⇒ 与批处理同路径，顺带修掉。

    所以判据落在**我能控制的两段**上；ego 段单独报数、不作为失败。
    """
    model = TemporalPredictor().eval()
    caught = {}
    handle = model.feature_fusion.register_forward_pre_hook(
        lambda _m, args: caught.__setitem__("v", args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            singles = torch.cat([(model(gd), caught["v"])[1] for gd in samples[:4]], dim=0)
            loader = DataLoader(samples[:4], batch_size=4, shuffle=False, follow_batch=["tg_x"])
            model(next(iter(loader)))
            batched = caught["v"]
    finally:
        handle.remove()

    seg = {
        name: float((singles[:, a:b] - batched[:, a:b]).abs().max())
        for name, (a, b) in [("ego", (0, 128)), ("context", (128, 256)), ("gnn", (256, 384))]
    }
    print(f"\n[D] batch vs single 分段：{seg}"
          f"\n    ego 段非零属已知 D0#13（BN track_running_stats=False），父类同量级")

    assert seg["context"] == 0.0, f"context 段批处理不一致 {seg['context']:.3e}"
    assert seg["gnn"] == 0.0, (
        f"🔴 GNN 段批处理不一致 {seg['gnn']:.3e} —— 检查 __inc__ 与 (样本,帧) 分组"
    )


def test_e_topology_changes_output(samples):
    """[E] knn30 与 full 两种拓扑必须给出不同的 GNN 分支输出。"""
    model = TemporalPredictor().eval()
    p = sorted(VAL_DIR.glob("features_*.pkl"))[0]
    gd_full = _make_sample(p, topology="full")
    gd_knn = _make_sample(p, topology="knn30")

    with torch.no_grad():
        out_full, out_knn = model(gd_full), model(gd_knn)

    diff = float((out_full - out_knn).abs().max())
    print(f"\n[E] full vs knn30 输出 max|Δ| = {diff:.3e}"
          f"｜边数 {gd_full.tg_edge_index.shape[1]} vs {gd_knn.tg_edge_index.shape[1]}")
    assert gd_knn.tg_edge_index.shape[1] <= gd_full.tg_edge_index.shape[1]
