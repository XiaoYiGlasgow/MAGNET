"""闸④（Q50）切断 ego 捷径的验收单测（对齐 `test_ego_query_fusion.py` 的判据风格）。

判据：
  A. `use_ego_history=True` 时与 `MagnetEgoQueryFusion` **逐字节一致**（默认零改变）。
  B. **靶点判据（本修复的全部意义）**：打开后改掉**前 18 帧**历史（最后两帧保持不变）
     ⇒ 输出**逐字节不变**；同一扰动下**父类必须变化**（否则本判据是空的）。
  C. **反向判据**：只改**最后一帧** ⇒ 输出必须变化 —— 当前状态真的被用上了，
     不是把 ego 分支整条接地。
  D. `state_dropout` **只在 train 模式生效**：eval 下多次前向逐字节一致；
     train 下多次前向必须不同；`state_dropout=0` 时 train 下也一致。
  E. 极端 dropout（0.99）下**不出 NaN** —— 恒可见的 3 个 token 是功能必需，这条守住它。
  F. 契约违例**抛错而非静默退化**（`ego_history` 形状不符）。
  G. **与 C2（闸②③）叠加**：`ego_query_fusion=True` 时 C2 的 ego query 也必须来自当前状态
     —— 否则捷径会从 query 侧漏回来。判据 = 改前 18 帧后 **context 分支特征也不变**。
  H. 当前状态的通道语义可核对：给定合成增量，`vel_x/acc_x/yaw_rate` 与手算一致；
     静止（增量≈0）时 yaw_rate 置 0 而不是把跟踪抖动当信号。

运行（CPU，不占 GPU）：
  PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python \
    /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/training/tests/test_ego_state_dropout.py
"""
from __future__ import annotations

import math
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

from magnet_ego_query_fusion import MagnetEgoQueryFusion  # noqa: E402
from magnet_ego_shortcut import (  # noqa: E402
    EGO_DT, EgoStateAlignmentError, MagnetEgoShortcutSuppressed,
    ego_current_state_from_history,
)
from src.shared.utils.dataset import GraphData  # noqa: E402

SEED = 20260801
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


def _perturbed(sample: GraphData, rows: slice, seed: int) -> GraphData:
    """复制一个样本，只替换 `ego_history[rows]`，其余字段逐字节相同。

    ⚠️ 不能用「加常数」扰动 —— `ego_history_encoder[1]` 是 BatchNorm1d，常数平移会被消掉
    （`test_ego_query_fusion.py` 判据 E 已踩过）。这里换**独立随机**值。
    """
    eh = sample.ego_history.clone()
    eh[rows] = torch.rand(eh[rows].shape, generator=torch.Generator().manual_seed(seed))
    out = _make_sample(1, 1, 2, torch.Generator().manual_seed(0))   # 占位，字段随后整体覆盖
    for k in ("x", "y", "cluster", "edge_index", "gnn_x", "gnn_edge_index",
              "gnn_edge_attr", "ego_last_pos"):
        setattr(out, k, getattr(sample, k).clone())
    out.ego_history = eh
    return out


# ---------------------------------------------------------------- 判据 A
def test_a_off_is_byte_identical() -> None:
    s = _samples(2)
    ref = _model(MagnetEgoQueryFusion, context_pooling=True, ego_query_fusion=True)
    new = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
                 use_ego_history=True)

    # 本类在父类之上又包了一层 ⇒ `ego_transformer.*` 参数名多一级 `inner.`。
    rs, ns = ref.state_dict(), new.state_dict()
    remap = {}
    for k in rs:
        pre = "ego_transformer."
        remap[k] = (pre + "inner." + k[len(pre):]) if k.startswith(pre) else k
    missing = [k for k, v in remap.items() if v not in ns]
    assert not missing, f"重映射后仍找不到的父类参数: {missing[:5]}"
    diff = [k for k, v in remap.items() if not torch.equal(rs[k], ns[v])]
    assert not diff, f"父类参数值不一致: {diff[:5]}"

    with torch.no_grad():
        d = _batch(s)
        delta = (ref(d) - new(d)).abs().max().item()
    assert delta == 0.0, f"关闭时应逐字节一致，实测 max|Δ|={delta:.3e}"
    print(f"  [A] 关闭时逐字节一致            PASS  "
          f"父类参数 {len(remap)} 项全等，输出 max|Δ|={delta:.3e}")


# ---------------------------------------------------------------- 判据 B（靶点）
def test_b_early_history_has_no_effect() -> None:
    """捷径切断的**可验证含义**：前 18 帧历史对输出零影响。"""
    base = _samples(1)[0]
    alt = _perturbed(base, slice(0, 18), SEED + 101)
    assert not torch.equal(alt.ego_history, base.ego_history)
    assert torch.equal(alt.ego_history[-2:], base.ego_history[-2:]), "最后两帧必须保持不变"
    assert torch.equal(alt.x, base.x) and torch.equal(alt.gnn_x, base.gnn_x)

    on = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
                use_ego_history=False)
    off = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
                 use_ego_history=True)
    with torch.no_grad():
        d_on = (on(_batch([base])) - on(_batch([alt]))).abs().max().item()
        d_off = (off(_batch([base])) - off(_batch([alt]))).abs().max().item()

    assert d_on == 0.0, f"打开后前 18 帧仍影响输出（Δ={d_on:.3e}）⇒ 捷径没切干净"
    assert d_off > 1e-5, (
        f"父类对同一扰动也不敏感（Δ={d_off:.3e}）⇒ 判据 B 是空的，需换扰动")
    print(f"  [B] 前 18 帧对输出零影响        PASS  "
          f"开={d_on:.3e}（逐字节不变）｜关={d_off:.3e}（父类对照，必须非零）")


# ---------------------------------------------------------------- 判据 C（反向）
def test_c_current_state_is_used() -> None:
    """只改最后一帧 ⇒ 必须变 —— 否则等于把 ego 分支整条接地，那不是本修复的语义。"""
    base = _samples(1)[0]
    alt = _perturbed(base, slice(19, 20), SEED + 202)
    assert torch.equal(alt.ego_history[:19], base.ego_history[:19])

    on = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
                use_ego_history=False)
    with torch.no_grad():
        delta = (on(_batch([base])) - on(_batch([alt]))).abs().max().item()
    assert delta > 1e-5, f"最后一帧不影响输出（Δ={delta:.3e}）⇒ 当前状态没接上"
    print(f"  [C] 当前状态真的被使用          PASS  Δ={delta:.3e}")


# ---------------------------------------------------------------- 判据 D（dropout 门控）
def test_d_dropout_only_in_train_mode() -> None:
    s = _samples(2)
    m = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
               use_ego_history=False, ego_state_dropout=0.75)

    with torch.no_grad():
        d = _batch(s)
        m.eval()
        e = (m(d) - m(d)).abs().max().item()
        m.train()
        torch.manual_seed(1)
        a = m(d)
        torch.manual_seed(2)
        b = m(d)
        t = (a - b).abs().max().item()

    m0 = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
                use_ego_history=False, ego_state_dropout=0.0)
    # ⚠️ 父类 output_head 里有 nn.Dropout(0.1) ⇒ train 模式下即使 state_dropout=0 也会变。
    #    故这一格直接查 state 编码器本身，隔离出 state_dropout 的效应。
    m0.train()
    st = ego_current_state_from_history(s[0].ego_history.unsqueeze(0), EGO_DT)
    with torch.no_grad():
        torch.manual_seed(1)
        z0 = m0.ego_state_emb(st)
        torch.manual_seed(2)
        z1 = m0.ego_state_emb(st)
    t0 = (z0 - z1).abs().max().item()

    assert e == 0.0, f"eval 下两次前向应逐字节一致，实测 {e:.3e}"
    assert t > 1e-5, f"train 下 state_dropout 未生效（Δ={t:.3e}）"
    assert t0 == 0.0, f"state_dropout=0 时 train 下也应确定，实测 {t0:.3e}"
    print(f"  [D] dropout 仅在 train 生效     PASS  "
          f"eval Δ={e:.3e}｜train Δ={t:.3e}｜p=0 时 train Δ={t0:.3e}")


# ---------------------------------------------------------------- 判据 E（NaN 防线）
def test_e_no_nan_under_extreme_dropout() -> None:
    """恒可见的 3 个 token 是功能必需：全 mask 会让 attention 出 NaN。"""
    m = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
               use_ego_history=False, ego_state_dropout=0.99)
    m.train()
    st = torch.randn(256, 6)
    with torch.no_grad():
        torch.manual_seed(7)
        out = m.ego_state_emb(st)
    assert torch.isfinite(out).all(), "极端 dropout 下出现 NaN/Inf ⇒ 恒可见 token 防线失效"
    print(f"  [E] 极端 dropout 无 NaN         PASS  n=256, p=0.99, "
          f"finite={int(torch.isfinite(out).all())}")


# ---------------------------------------------------------------- 判据 F
def test_f_raises_instead_of_degrading() -> None:
    m = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
               use_ego_history=False)
    bad = _samples(1)[0]
    bad.ego_history = torch.rand(7, 2)          # 契约要求 [20,2]
    try:
        m(_batch([bad]))
    except EgoStateAlignmentError as exc:
        assert "ego_history" in str(exc)
        print("  [F] 契约违例时抛错              PASS")
        return
    raise AssertionError("ego_history 形状不符时应抛 EgoStateAlignmentError，实际静默通过")


# ---------------------------------------------------------------- 判据 G（与 C2 叠加）
def test_g_c2_ego_query_also_cut() -> None:
    """C2 的 ego query 取自 `_ego_feat`；闸④ 打开后它必须也来自当前状态。

    若父类那个挂在**内层** `ego_transformer` 的 hook 没被补挂到包装器上，
    这里会以「context 分支仍对前 18 帧敏感」或「ego query 取不到值抛错」的形式暴露。
    """
    base = _samples(1)[0]
    alt = _perturbed(base, slice(0, 18), SEED + 303)

    m = _model(MagnetEgoShortcutSuppressed, context_pooling=True, ego_query_fusion=True,
               use_ego_history=False)
    with torch.no_grad():
        f0 = m.get_encoder_features(_batch([base]))
        f1 = m.get_encoder_features(_batch([alt]))
    # encoder 特征 = [ego(0:128) | context(128:256) | gnn(256:384)]
    d_ego = (f0[:, :128] - f1[:, :128]).abs().max().item()
    d_ctx = (f0[:, 128:256] - f1[:, 128:256]).abs().max().item()
    assert d_ego == 0.0, f"ego 分支仍受前 18 帧影响（Δ={d_ego:.3e}）"
    assert d_ctx == 0.0, (
        f"context 分支仍受前 18 帧影响（Δ={d_ctx:.3e}）⇒ 捷径从 C2 的 ego query 侧漏回来了")

    # 同时确认 ego query 确实非零（不是「取不到值退化成 0」）
    assert m._ego_feat is not None and m._ego_feat.abs().max().item() > 0
    print(f"  [G] C2 ego query 也已切断       PASS  "
          f"Δego={d_ego:.3e}｜Δcontext={d_ctx:.3e}｜ego_query 非零")


# ---------------------------------------------------------------- 判据 H（通道语义）
def test_h_state_channels_match_hand_computation() -> None:
    h = torch.zeros(2, 20, 2)
    # 样本 0：纵向 1.0 m/帧 → 1.2 m/帧，方向从 +x 转到 45°
    h[0, -2] = torch.tensor([1.0, 0.0])
    h[0, -1] = torch.tensor([1.2, 1.2])
    # 样本 1：静止（增量远小于 _STATIONARY_EPS=1e-3 m/帧）
    h[1, -2] = torch.tensor([1e-5, 0.0])
    h[1, -1] = torch.tensor([1e-5, 1e-5])

    st = ego_current_state_from_history(h, EGO_DT)
    assert st.shape == (2, 6), f"状态形状应为 [2,6]，实得 {tuple(st.shape)}"
    assert st[:, :3].abs().max().item() == 0.0, "前 3 通道应恒为 0（ego-centric 位姿）"

    exp_v = 1.2 / EGO_DT
    exp_a = (1.2 - 1.0) / EGO_DT ** 2
    exp_w = (math.pi / 4) / EGO_DT
    for name, got, exp in (("vel_x", st[0, 3].item(), exp_v),
                           ("acc_x", st[0, 4].item(), exp_a),
                           ("yaw_rate", st[0, 5].item(), exp_w)):
        assert abs(got - exp) < 1e-4, f"{name} 手算 {exp:.4f}，实得 {got:.4f}"
    assert st[1, 5].item() == 0.0, (
        f"静止样本的 yaw_rate 应置 0（跟踪抖动不是信号），实得 {st[1, 5].item():.4e}")

    print(f"  [H] 通道语义与手算一致          PASS  "
          f"vel_x={st[0,3]:.2f} m/s｜acc_x={st[0,4]:.2f} m/s²｜"
          f"yaw_rate={st[0,5]:.3f} rad/s｜静止样本 yaw_rate=0")


if __name__ == "__main__":
    tests = [test_a_off_is_byte_identical, test_b_early_history_has_no_effect,
             test_c_current_state_is_used, test_d_dropout_only_in_train_mode,
             test_e_no_nan_under_extreme_dropout, test_f_raises_instead_of_degrading,
             test_g_c2_ego_query_also_cut, test_h_state_channels_match_hand_computation]
    print(f"闸④（Q50）切断 ego 捷径验收（{len(tests)} 项判据）")
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} PASS")
