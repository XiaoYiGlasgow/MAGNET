"""D0#7 验收单测：GNN 分支逐样本池化修复（阶段15 Z2）。

两条判据（阶段15 §3.1 第 7 项原文）：
  A. 修复后**同 batch 内不同样本的 `gnn_feat` 不相同**；
  B. **batch=1 时与修复前逐字节一致**（旧行为零改变，在线 planner 路径不受影响）。

再补三条本次实现引入的判据：
  C. 没有 `gnn_x_batch` 时（旧 DataLoader / 单样本）修复版与父类逐字节一致；
  D. `gnn_x_batch` 覆盖不全时**抛错而非静默退化**（父类会退回 mean(0).repeat，正是要消除的东西）；
  E. 未修复的父类在 batch>1 时确实**所有样本共享同一个 gnn_feat** —— 即缺陷复现，
     否则这个测试通过也没有意义。

运行（不占 GPU，CPU 即可）：
  PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python \
    /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/training/tests/test_gnn_pooling.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parents[1]
_GNN_ROOT = _THIS.parents[3]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_geometric.loader import DataLoader  # noqa: E402

from magnet_imitation_pooled import (  # noqa: E402
    GnnBatchCoverageError, MagnetImitationPooled, attach_gnn_batch,
)
from src.shared.utils.dataset import GraphData  # noqa: E402
from transformer_predictor import TransformerPredictor  # noqa: E402

SEED = 20260728
N_FUTURE = 60


def make_sample(n_ctx: int, n_gnn: int, gen: torch.Generator) -> GraphData:
    """一个形状合法的最小 GraphData。特征随机 => 不同样本的 gnn 嵌入本应不同。"""
    x = torch.randn(n_ctx, 12, generator=gen)
    edge_index = torch.stack([
        torch.arange(n_ctx).repeat_interleave(n_ctx),
        torch.arange(n_ctx).repeat(n_ctx),
    ])
    gnn_x = torch.randn(n_gnn, 12, generator=gen)
    gnn_edge_index = torch.stack([
        torch.arange(n_gnn).repeat_interleave(n_gnn),
        torch.arange(n_gnn).repeat(n_gnn),
    ])
    return GraphData(
        x=x, edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 2, generator=gen),
        y=torch.randn(N_FUTURE * 2, generator=gen),
        gnn_x=gnn_x, gnn_edge_index=gnn_edge_index,
        gnn_edge_attr=torch.zeros(gnn_edge_index.size(1), dtype=torch.long),
        ego_history=torch.randn(20, 2, generator=gen),
        ego_last_pos=torch.zeros(1, 2),
    )


def gnn_feat_of(model: TransformerPredictor, data) -> torch.Tensor:
    """抓 forward 里第 3 支的 `gnn_feat`：在 feature_fusion 上挂 hook 取输入的后 128 维。

    `combined_feat = cat([ego, context, gnn], dim=1)` -> fusion 的输入就是 [B,384]，
    末 128 列即 gnn_feat。比复制 forward 更可靠：它测的是**真实执行路径**。
    """
    captured: List[torch.Tensor] = []
    handle = model.feature_fusion.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        model(data)
    finally:
        handle.remove()
    assert captured, "feature_fusion 未被调用，forward 路径与预期不符"
    return captured[0][:, 256:]


def build_models() -> tuple:
    """父类与修复子类共享同一份权重，否则任何差异都无法归因到池化。"""
    torch.manual_seed(SEED)
    parent = TransformerPredictor().eval()
    torch.manual_seed(SEED)
    fixed = MagnetImitationPooled().eval()
    fixed.load_state_dict(parent.state_dict())
    return parent, fixed


def _batch(samples, follow: bool):
    loader = DataLoader(samples, batch_size=len(samples), shuffle=False,
                        follow_batch=["gnn_x"] if follow else None)
    return next(iter(loader))


def test_e_defect_reproduces() -> None:
    """E：先证明缺陷真实存在 —— 未修复的父类在 batch>1 时所有样本 gnn_feat 相同。"""
    gen = torch.Generator().manual_seed(SEED)
    samples = [make_sample(6 + i, 4 + i, gen) for i in range(4)]
    parent, _ = build_models()
    feats = gnn_feat_of(parent, _batch(samples, follow=False))
    assert feats.shape == (4, 128), feats.shape
    spread = (feats - feats[0:1]).abs().max().item()
    assert spread == 0.0, f"缺陷未复现：父类各样本 gnn_feat 已不同（max diff={spread:.3e}）"
    print(f"  E ✅ 缺陷复现：父类 batch=4 的 gnn_feat 完全相同（max diff={spread:.1e}）")


def test_a_per_sample_after_fix() -> None:
    """A：修复后同 batch 内不同样本的 gnn_feat 必须互不相同。"""
    gen = torch.Generator().manual_seed(SEED)
    samples = [make_sample(6 + i, 4 + i, gen) for i in range(4)]
    _, fixed = build_models()
    feats = gnn_feat_of(fixed, _batch(samples, follow=True))
    assert feats.shape == (4, 128), feats.shape
    for i in range(feats.size(0)):
        for j in range(i + 1, feats.size(0)):
            diff = (feats[i] - feats[j]).abs().max().item()
            assert diff > 1e-6, f"样本 {i} 与 {j} 的 gnn_feat 仍相同（max diff={diff:.3e}）"
    pair_min = min((feats[i] - feats[j]).abs().max().item()
                   for i in range(4) for j in range(i + 1, 4))
    print(f"  A ✅ 修复后逐样本：最接近的一对 gnn_feat 也相差 {pair_min:.4f}")


def test_b_batch_one_bitwise_identical() -> None:
    """B：batch=1 时修复版与父类**逐字节**一致（在线 planner 路径零改变）。"""
    gen = torch.Generator().manual_seed(SEED)
    sample = make_sample(7, 5, gen)
    parent, fixed = build_models()
    before = gnn_feat_of(parent, _batch([sample], follow=False))
    after = gnn_feat_of(fixed, _batch([sample], follow=True))
    assert torch.equal(before, after), \
        f"batch=1 不再逐字节一致，max diff={(before - after).abs().max().item():.3e}"
    out_before, out_after = parent(_batch([sample], follow=False)), fixed(_batch([sample], follow=True))
    assert torch.equal(out_before, out_after), "batch=1 的最终输出也必须逐字节一致"
    print("  B ✅ batch=1 逐字节一致（gnn_feat 与最终 120 维输出都相等）")


def test_c_no_follow_batch_is_noop() -> None:
    """C：没有 gnn_x_batch 时修复版 == 父类（旧 DataLoader 调用方零影响）。"""
    gen = torch.Generator().manual_seed(SEED)
    samples = [make_sample(6 + i, 4 + i, gen) for i in range(3)]
    parent, fixed = build_models()
    before = gnn_feat_of(parent, _batch(samples, follow=False))
    after = gnn_feat_of(fixed, _batch(samples, follow=False))
    assert torch.equal(before, after), "无 follow_batch 时修复版必须完全退化为父类行为"
    print("  C ✅ 无 gnn_x_batch 时为 no-op，旧调用方逐字节不变")


def test_d_incomplete_coverage_raises() -> None:
    """D：gnn_x_batch 覆盖不全时抛错，而不是让父类静默退回 mean(0).repeat。"""
    gen = torch.Generator().manual_seed(SEED)
    batch = _batch([make_sample(6, 4, gen) for _ in range(3)], follow=True)
    batch.gnn_x_batch = torch.clamp(batch.gnn_x_batch, max=1)   # 抹掉第 3 个图
    try:
        attach_gnn_batch(batch, batch_size=3, strict=True)
    except GnnBatchCoverageError as exc:
        assert "3" in str(exc)
        print("  D ✅ 覆盖不全时抛 GnnBatchCoverageError，未静默退化")
        assert attach_gnn_batch(batch, batch_size=3, strict=False) is batch
        assert getattr(batch, "gnn_batch", None) is None, "strict=False 应保持旧行为"
        return
    raise AssertionError("覆盖不全却没有抛错 —— 修复会静默失效")


def main() -> int:
    print("=" * 66)
    print(" D0#7 GNN 逐样本池化 —— 单测")
    print("=" * 66)
    tests = [test_e_defect_reproduces, test_a_per_sample_after_fix,
             test_b_batch_one_bitwise_identical, test_c_no_follow_batch_is_noop,
             test_d_incomplete_coverage_raises]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  ❌ {fn.__name__}: {exc}")
    print("-" * 66)
    print(f" {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
