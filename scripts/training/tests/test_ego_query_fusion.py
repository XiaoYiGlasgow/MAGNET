"""C2 ego-query 融合验收单测（对齐 `test_context_pooling.py` / `test_gnn_pooling.py` 的判据风格）。

判据：
  A. `ego_query_fusion=False` 时与 `MagnetContextPooled` **逐字节一致**（默认零改变）。
  B. 开启后 batch>1 时不同样本输出**互不相同**（没有跨样本串台）。
  C. 开启后 batch=1 与 batch>1 对**同一个样本**给出一致结果（容差 1e-5，float32 归约噪声）。
  D. **机制判据（本修复的全部意义所在）**：注意力权重**不是均匀分布** ——
     mean pooling 恒为 1/N，本修复必须让权重由数据决定。不满足这条，修复就没发生。
  E. **靶点判据**：ego query 真的参与了计算 —— 换掉 ego 历史（其余不变）必须改变输出。
     mean pooling 下 context/gnn 分支对 ego 完全免疫，这条会失败。
  F. 对齐失败时**抛错而非静默退化**（ego query 缺失场景）。
  G. 两条分支的池化器**参数不共享**（context 与 gnn 的节点语义不同）。

运行（CPU，不占 GPU）：
  PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python \
    /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/training/tests/test_ego_query_fusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parents[1]
_GNN_ROOT = _THIS.parents[3]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_geometric.loader import DataLoader  # noqa: E402

from magnet_context_pooled import MagnetContextPooled  # noqa: E402
from magnet_ego_query_fusion import (  # noqa: E402
    EgoQueryAlignmentError, MagnetEgoQueryFusion,
)
from src.shared.utils.dataset import GraphData  # noqa: E402

SEED = 20260731
N_FUTURE = 60


def _make_sample(n_poly: int, pts_per_poly: int, n_gnn: int,
                 gen: torch.Generator, ego: torch.Tensor | None = None) -> GraphData:
    m = n_poly * pts_per_poly
    src, dst = [], []
    for i in range(n_gnn):
        for j in range(i + 1, n_gnn):
            src += [i, j]
            dst += [j, i]
    return GraphData(
        x=torch.rand(m, 12, generator=gen),
        y=torch.rand(1, N_FUTURE * 2, generator=gen),
        cluster=torch.arange(n_poly).repeat_interleave(pts_per_poly),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        gnn_x=torch.rand(n_gnn, 12, generator=gen),
        gnn_edge_index=torch.tensor([src, dst], dtype=torch.long),
        gnn_edge_attr=torch.zeros(len(src), dtype=torch.long),
        ego_history=torch.rand(20, 2, generator=gen) if ego is None else ego,
        ego_last_pos=torch.zeros(2),
    )


def _batch(samples, follow=("gnn_x",)):
    return next(iter(DataLoader(samples, batch_size=len(samples),
                                shuffle=False, follow_batch=list(follow))))


def _model(cls, **kw):
    torch.manual_seed(SEED)
    return cls(**kw).eval()


def _samples(n=3):
    gen = torch.Generator().manual_seed(SEED)
    return [_make_sample(6 + i, 5, 8 + i, gen) for i in range(n)]


# ---------------------------------------------------------------- 判据 A
def test_a_off_is_byte_identical() -> None:
    s = _samples(2)
    ref = _model(MagnetContextPooled, context_pooling=True)
    new = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=False)

    # 本类在父类之上又包了一层 ⇒ 参数名多一级 `inner.`；按前缀重映射后必须逐参数相等。
    # （两者构造时同 seed，且新增池化器在 `super().__init__()` **之后**才消耗 RNG，
    #   故父类参数天然一致；这里显式核对，避免日后改动构造顺序后静默分叉。）
    rs, ns = ref.state_dict(), new.state_dict()
    remap = {}
    for k in rs:
        for pre in ("context_transformer.", "gnn_encoder."):
            if k.startswith(pre):
                remap[k] = pre + "inner." + k[len(pre):]
                break
        else:
            remap[k] = k
    missing = [k for k, v in remap.items() if v not in ns]
    assert not missing, f"重映射后仍找不到的父类参数: {missing[:5]}"
    diff = [k for k, v in remap.items() if not torch.equal(rs[k], ns[v])]
    assert not diff, f"父类参数值不一致: {diff[:5]}"

    with torch.no_grad():
        d = _batch(s)
        delta = (ref(d) - new(d)).abs().max().item()
    assert delta == 0.0, f"关闭时应逐字节一致，实测 max|Δ|={delta:.3e}"
    print(f"  [A] 关闭时逐字节一致           PASS  "
          f"父类参数 {len(remap)} 项全等，输出 max|Δ|={delta:.3e}")


# ---------------------------------------------------------------- 判据 B
def test_b_no_cross_sample_leak() -> None:
    m = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)
    with torch.no_grad():
        out = m(_batch(_samples(3)))
    assert out.size(0) == 3, f"输出样本数应为 3，实得 {out.size(0)}"
    pairs = [(out[i] - out[j]).abs().max().item()
             for i in range(3) for j in range(i + 1, 3)]
    assert min(pairs) > 1e-4, f"样本间输出过于接近，疑似串台: {pairs}"
    print(f"  [B] batch>1 无跨样本串台        PASS  最小两两 max|Δ|={min(pairs):.3e}")


# ---------------------------------------------------------------- 判据 C
def test_c_batch_invariant() -> None:
    """本修复**不得引入新的** batch 依赖。

    ⚠️ 模型本身已有 **D0#13** 的 batch 依赖 —— `ego_history_encoder[1]` 是
    `BatchNorm1d(track_running_stats=False)`，eval 下仍用当前 batch 统计量，
    量级 ~1e-1，会把本修复自身 ~1e-7 的归约噪声完全淹没。
    故与 `test_context_pooling.py` 判据 C 用**同一手法**：中性化 BN 后再测，
    并同时报含 BN 的数字作为 matched control。
    """
    s = _samples(3)

    def _dep(model) -> float:
        with torch.no_grad():
            multi = model(_batch(s))[1]
            single = model(_batch([s[1]]))[0]
        return (multi - single).abs().max().item()

    def _kill_bn(model):
        model.ego_history_encoder[1] = torch.nn.Identity()
        return model

    d_bn_off = _dep(_model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=False))
    d_bn_on = _dep(_model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True))
    d_off = _dep(_kill_bn(_model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=False)))
    d_on = _dep(_kill_bn(_model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)))

    assert d_on < 1e-5, f"去 BN 后本修复仍有 batch 依赖: {d_on:.3e}"
    print(f"  [C] 去 BN 后零 batch 依赖       PASS  "
          f"含BN: 关={d_bn_off:.3e} 开={d_bn_on:.3e}（D0#13 主导）｜"
          f"去BN: 关={d_off:.3e} 开={d_on:.3e}")


# ---------------------------------------------------------------- 判据 D（机制）
def test_d_attention_is_not_uniform() -> None:
    """mean pooling 恒为 1/N；修复的意义就是让权重由数据决定。

    🔴 **但 GNN 分支在完全图上必然是均匀的，这不是缺陷而是被证实的性质**：
    GCN 的对称归一化邻接 `Â = D^{-1/2}(A+I)D^{-1/2}` 在完全图（每个节点邻域相同）上
    **每一行都相同** ⇒ `ÂX` 的每一行都相同 ⇒ **无论权重是什么，一层之后所有节点嵌入恒等**。
    真实 `B_seed42_calibrated` + 真实 val 样本实测：8/8 场景是严格完全图
    （e = n(n−1)），**conv1 之后**节点间最大差已只有 ~1e-7（float32 精度内相同）。
    ⇒ **GNN 分支的信息在池化之前就已经被销毁**，attention 池化对它无能为力。
    ⇒ **C2 必须与非完全图（`nearby_only`）一起用**；G16 已证明这不付出精度代价。
    """
    m = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)
    with torch.no_grad():
        m(_batch(_samples(2)))          # `_make_sample` 造的正是完全图

    # D-1 context 分支：无图、不过平滑 ⇒ 权重必须非均匀（修复在这里真的生效）
    w = m.context_pool.last_attn_weights
    dev_ctx = (w - 1.0 / w.size(-1)).abs().max().item()
    assert dev_ctx > 1e-4, f"context 分支权重与均匀无异（dev={dev_ctx:.2e}）⇒ 修复未生效"

    # D-2 GNN 分支 @ 完全图：权重**应当**均匀（过平滑的直接后果）
    w = m.gnn_pool.last_attn_weights
    dev_full = (w - 1.0 / w.size(-1)).abs().max().item()
    assert dev_full < 1e-5, (
        f"完全图上 GNN 节点嵌入应恒等 ⇒ 权重应均匀，实测偏离 {dev_full:.2e}；"
        "若本条失败说明过平滑判断有误，需重新核实")

    # D-3 同一模型换成链式图（非完全）⇒ 权重必须变得非均匀
    gen = torch.Generator().manual_seed(SEED)
    s = _make_sample(6, 5, 8, gen)
    n = 8
    src = list(range(n - 1)) + list(range(1, n))
    dst = list(range(1, n)) + list(range(n - 1))
    s.gnn_edge_index = torch.tensor([src, dst], dtype=torch.long)
    s.gnn_edge_attr = torch.zeros(len(src), dtype=torch.long)
    with torch.no_grad():
        m(_batch([s]))
    w = m.gnn_pool.last_attn_weights
    dev_chain = (w - 1.0 / w.size(-1)).abs().max().item()
    assert dev_chain > 1e-4, (
        f"非完全图上 GNN 权重仍均匀（dev={dev_chain:.2e}）⇒ ego query 没接进 GNN 分支")

    print(f"  [D] 权重由数据决定              PASS  "
          f"context={dev_ctx:.4f}；GNN@完全图={dev_full:.2e}（过平滑，应均匀）；"
          f"GNN@链式图={dev_chain:.4f}")


# ---------------------------------------------------------------- 判据 E（靶点）
def test_e_ego_actually_conditions_the_pool() -> None:
    """父类 mean pooling 下 context/gnn 分支对 ego **完全免疫**；修复后必须不再免疫。"""
    gen = torch.Generator().manual_seed(SEED)
    base = _make_sample(6, 5, 8, gen)
    alt = _make_sample(6, 5, 8, torch.Generator().manual_seed(SEED))
    # ⚠️ 不能用「加常数」扰动 ego —— `ego_history_encoder[1]` 是 BatchNorm1d，
    #    常数平移会被它完全消掉（首次写测试时踩过：Δ 只有 4.8e-07）。
    #    改用**独立随机**的 ego 历史，其余字段逐字节相同。
    alt.ego_history = torch.rand(20, 2, generator=torch.Generator().manual_seed(SEED + 977))
    assert not torch.equal(alt.ego_history, base.ego_history)
    assert torch.equal(alt.x, base.x) and torch.equal(alt.gnn_x, base.gnn_x)

    m = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)

    def _feats(sample):
        with torch.no_grad():
            return m.get_encoder_features(_batch([sample]))

    # encoder 特征 = [ego(0:128) | context(128:256) | gnn(256:384)]
    f0, f1 = _feats(base), _feats(alt)
    d_ctx = (f0[:, 128:256] - f1[:, 128:256]).abs().max().item()
    d_gnn_full = (f0[:, 256:384] - f1[:, 256:384]).abs().max().item()

    # E-1 context 分支：必须不再对 ego 免疫（父类 mean pooling 下它是完全免疫的）
    assert d_ctx > 1e-5, f"context 分支仍对 ego 免疫（Δ={d_ctx:.2e}）⇒ ego query 没接上"

    # E-2 GNN 分支 @ 完全图：**必然**仍对 ego 免疫 —— 见 test_d 的过平滑说明。
    #     所有 key 恒等 ⇒ 无论 query 是什么，attention 输出都是那个常量。
    #     ⇒ **在完全图上，任何融合层面的修复都救不了 GNN 分支。**
    assert d_gnn_full < 1e-5, (
        f"完全图上 GNN 分支应对 ego 免疫（过平滑），实测 Δ={d_gnn_full:.2e}")

    # E-3 换成链式图（非完全）⇒ GNN 分支必须变得对 ego 敏感
    for s in (base, alt):
        n = s.gnn_x.size(0)
        src = list(range(n - 1)) + list(range(1, n))
        dst = list(range(1, n)) + list(range(n - 1))
        s.gnn_edge_index = torch.tensor([src, dst], dtype=torch.long)
        s.gnn_edge_attr = torch.zeros(len(src), dtype=torch.long)
    g0, g1 = _feats(base), _feats(alt)
    d_gnn_chain = (g0[:, 256:384] - g1[:, 256:384]).abs().max().item()
    assert d_gnn_chain > 1e-5, (
        f"非完全图上 GNN 分支仍对 ego 免疫（Δ={d_gnn_chain:.2e}）⇒ ego query 没接进 GNN 分支")

    print(f"  [E] ego 真的在条件化分支        PASS  Δcontext={d_ctx:.3e}；"
          f"ΔGNN@完全图={d_gnn_full:.2e}（必然免疫）；ΔGNN@链式图={d_gnn_chain:.3e}")


# ---------------------------------------------------------------- 判据 F
def test_f_raises_instead_of_degrading() -> None:
    """ego 分支未触发时必须抛错 —— 静默回退到 mean pooling 会让实验读数失去意义。"""
    m = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)
    d = _batch(_samples(1))
    m._prepare_ego_query(d)
    m._ego_feat = None                                # 模拟 ego 分支未跑
    try:
        m.context_transformer.pool  # noqa: B018
        m.context_transformer(torch.rand(1, 5, 128))
    except EgoQueryAlignmentError as exc:
        assert "ego_transformer" in str(exc)
        print("  [F] ego query 缺失时抛错        PASS")
        return
    raise AssertionError("ego query 缺失时应抛 EgoQueryAlignmentError，实际静默通过")


# ---------------------------------------------------------------- 判据 G
def test_g_pools_are_not_shared() -> None:
    m = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)
    assert m.context_pool is not m.gnn_pool, "两条分支不应共享池化器实例"
    same = torch.equal(m.context_pool.q_proj.weight, m.gnn_pool.q_proj.weight)
    assert not same, "两条分支的 q_proj 权重相同 ⇒ 疑似共享参数"
    print("  [G] context / gnn 池化器不共享  PASS")


if __name__ == "__main__":
    tests = [test_a_off_is_byte_identical, test_b_no_cross_sample_leak,
             test_c_batch_invariant, test_d_attention_is_not_uniform,
             test_e_ego_actually_conditions_the_pool,
             test_f_raises_instead_of_degrading, test_g_pools_are_not_shared]
    print(f"C2 ego-query 融合验收（{len(tests)} 项判据）")
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} PASS")
