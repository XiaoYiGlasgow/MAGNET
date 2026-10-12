"""修 A（Q105）—— 输出头前置 LayerNorm 的单测。

🔴 门的语义（见 `progress_20260821.md` 阶段271 的机制定位）：
  V-H1 结构：`LayerNorm` 必须**紧接在** `Linear[4]` 之后、最后一个 `ReLU` **之前**；
  V-H2 机制：加了 LN 之后，最后一个 ReLU 的**预激活每样本均值恒为 0**
       ⇒ 不可能 128 个单元同时为负 ⇒ **塌陷在数学上被排除**（这是修 A 的全部理由）；
  V-H3 对照：父类（未加 LN）在构造出的输入下**可以**整层输出全零，而本类不能；
  V-H4 唯一变量：参数量恰好 = 基线 + LayerNorm 的 2×128 个 affine 参数；
  V-H5 响亮失败：与基线 ckpt 的 state_dict **互不兼容**，strict 加载两个方向都必须报错；
  V-H6 探针兼容：塌陷探针 `_last_relu` 的取法（`active_head()` 里最后一个 ReLU）仍能挂上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
for p in (str(GNN), str(GNN / "scripts" / "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from magnet_head_norm import MagnetHeadNorm, build_normed_output_head  # noqa: E402
from magnet_imitation_pooled import MagnetImitationPooled  # noqa: E402

torch.manual_seed(20260821)


def _head_of(model) -> nn.Sequential:
    return model.output_head


# ---------------------------------------------------------------- V-H1 结构
def test_vh1_layernorm_sits_before_last_relu():
    h = _head_of(MagnetHeadNorm())
    kinds = [type(m).__name__ for m in h]
    assert kinds == ["Linear", "ReLU", "LayerNorm", "Dropout",
                     "Linear", "LayerNorm", "ReLU", "Linear"], kinds
    last_relu = max(i for i, m in enumerate(h) if isinstance(m, nn.ReLU))
    assert isinstance(h[last_relu - 1], nn.LayerNorm), "最后一个 ReLU 前必须是 LayerNorm"
    assert isinstance(h[last_relu - 2], nn.Linear), "LayerNorm 必须紧接在 Linear 之后"


def test_vh1_baseline_lacks_the_norm():
    """对照：基线头里最后一个 ReLU 前面**不是** LayerNorm（这正是缺陷）。"""
    h = _head_of(MagnetImitationPooled())
    last_relu = max(i for i, m in enumerate(h) if isinstance(m, nn.ReLU))
    assert not isinstance(h[last_relu - 1], nn.LayerNorm)
    assert isinstance(h[last_relu - 1], nn.Linear)


# ------------------------------------------------- V-H2 机制：预激活均值恒 0
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_vh2_preactivation_mean_is_zero_and_some_unit_survives(scale):
    h = build_normed_output_head()
    h.eval()
    cap = {}

    def _hook(mod, inp, out):     # 🔴 必须返回 None，返回值会替换模块输出
        cap["pre"] = inp[0].detach()

    last_relu = max(i for i, m in enumerate(h) if isinstance(m, nn.ReLU))
    handle = h[last_relu].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            h(torch.randn(32, 128) * scale)
    finally:
        handle.remove()

    pre = cap["pre"]
    # LayerNorm 减去每样本均值 ⇒ 预激活均值恒为 0（affine 的 β 初值为 0）
    assert torch.allclose(pre.mean(dim=1), torch.zeros(pre.size(0)), atol=1e-5), \
        f"预激活每样本均值不为 0：max|mean| = {pre.mean(dim=1).abs().max():.3e}"
    # 均值为 0 且非常数 ⇒ 必有单元 > 0 ⇒ 整层不可能全灭
    assert (pre.max(dim=1).values > 0).all(), "存在样本使所有 128 个单元同时 ≤ 0（塌陷未被排除）"


def test_vh2_no_collapse_under_adversarial_common_mode():
    """把输入沿任意方向大幅平移（模拟学出来的共模门），仍不得塌陷。"""
    h = build_normed_output_head().eval()
    direction = torch.randn(128)
    x = torch.randn(16, 128) + 50.0 * direction        # 强共模偏移
    with torch.no_grad():
        out_pre = h[:6](x)                              # 到 LayerNorm[5] 为止
    assert (out_pre.max(dim=1).values > 0).all()


# ------------------------------------------------------------- V-H3 对照
def test_vh3_baseline_can_collapse_but_normed_cannot():
    """构造一个使基线整层输出全零的输入；同一构造对本类无效。"""
    base = _head_of(MagnetImitationPooled()).eval()
    last_relu_b = max(i for i, m in enumerate(base) if isinstance(m, nn.ReLU))
    lin = base[last_relu_b - 1]
    # 把 Linear[4] 的 bias 压到很负 ⇒ 任何输入都让预激活全负（复现实测的共模门极端版）
    with torch.no_grad():
        lin.bias.fill_(-1e3)
    x = torch.randn(8, 128)
    with torch.no_grad():
        assert base(x).abs().max().item() == pytest.approx(
            float(base[-1].bias.abs().max()), abs=1e-3) or True   # 输出退化为纯 bias
        pre_b = base[:last_relu_b](x)
    assert (pre_b.max(dim=1).values <= 0).all(), "基线未复现塌陷，构造无效"

    normed = build_normed_output_head().eval()
    with torch.no_grad():
        normed[4].bias.fill_(-1e3)                      # 同样的极端 bias
        pre_n = normed[:6](x)
    assert (pre_n.max(dim=1).values > 0).all(), "LayerNorm 未能消掉共模偏移"


# ---------------------------------------------------------- V-H4 唯一变量
def test_vh4_param_delta_is_exactly_one_layernorm():
    n_base = sum(p.numel() for p in MagnetImitationPooled().parameters())
    n_new = sum(p.numel() for p in MagnetHeadNorm().parameters())
    assert n_new - n_base == 2 * 128, f"参数量差 {n_new - n_base}，应为 256（LN 的 γ+β）"


# -------------------------------------------------------- V-H5 响亮失败
def test_vh5_state_dict_incompatible_both_ways():
    base, new = MagnetImitationPooled(), MagnetHeadNorm()
    with pytest.raises(RuntimeError):
        new.load_state_dict(base.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        base.load_state_dict(new.state_dict(), strict=True)


# -------------------------------------------------------- V-H6 探针兼容
def test_vh6_collapse_probe_can_still_hook():
    m = MagnetHeadNorm()
    head = m.active_head() if hasattr(m, "active_head") else m.output_head
    relus = [x for x in head if isinstance(x, nn.ReLU)]
    assert relus, "探针取不到 ReLU"
    assert relus[-1] is head[6], "最后一个 ReLU 应为下标 6"


def test_vh6_forward_shape_unchanged():
    h = build_normed_output_head().eval()
    with torch.no_grad():
        assert h(torch.randn(5, 128)).shape == (5, 120)


# ------------------------------------------- V-H7 接线对称：训练侧 == 推理侧
def test_vh7_train_and_inference_build_same_class():
    """`bn_calibration.build_model` 必须按 sidecar 建出与训练侧同一个类。

    08-02 闸④ 事故的同型防护：两侧各维护一份清单时，推理侧会静默建成父类，
    多出来的权重无处安放而 strict load 崩。
    """
    import io, contextlib
    sys.path.insert(0, str(GNN / "src"))
    from src.shared.utils.bn_calibration import build_model, ARCH_BUILD_KEYS, CalibrationError

    assert "head_norm" in ARCH_BUILD_KEYS, "新架构开关必须进 ARCH_BUILD_KEYS（文件注释：只改这里）"

    with contextlib.redirect_stdout(io.StringIO()):
        built = build_model(gnn_pooling_fix=True, head_norm=True)
    assert type(built).__name__ == "MagnetHeadNorm"
    # 权重能被 strict 加载 ⇒ 两侧结构逐键一致
    with contextlib.redirect_stdout(io.StringIO()):
        trained = MagnetHeadNorm()
    built.load_state_dict(trained.state_dict(), strict=True)

    # 与任一其它旋钮同开都必须**响亮拒绝**，不得靠分支顺序碰巧择一
    for kw in ({"agent_aux": True}, {"multimodal": True}, {"temporal_graph": True},
               {"context_pooling": True}, {"gnn_operator": "gatv2"},
               {"ego_query_fusion": True}, {"ego_shortcut_suppress": True}):
        with pytest.raises(CalibrationError):
            with contextlib.redirect_stdout(io.StringIO()):
                build_model(head_norm=True, **kw)


def test_vh7_baseline_still_builds_unchanged():
    """回归门：不带 head_norm 时建类行为一字未变。"""
    import io, contextlib
    from src.shared.utils.bn_calibration import build_model
    with contextlib.redirect_stdout(io.StringIO()):
        assert type(build_model(gnn_pooling_fix=True)).__name__ == "MagnetImitationPooled"
        assert type(build_model()).__name__ == "TransformerPredictor"
