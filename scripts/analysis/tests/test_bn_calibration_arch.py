"""C3 前置 —— 推理侧能不能正确重建 **C2 架构**的 checkpoint。

为什么必须先测
--------------
C3 的 planner 走 `bn_calibration.load_model_for_inference` 加载 ckpt。
在 C2 之前只有两种架构（`TransformerPredictor` / `MagnetImitationPooled`），
`build_model` 也只认这两种 ⇒ **C2 的 `MagnetEgoQueryFusion` 权重根本加载不进去**，
C3 会在第一个场景就崩。这是 2026-08-01 在 C2 训练起跑前查出的。

两个隐蔽点
----------
1. **calibrated 产物是 bare state_dict**，架构信息不在权重里 ⇒ 只能靠同目录 `train_meta.json` sidecar；
2. **`ego_query_heads` 无法从权重形状反推**（`MultiheadAttention.in_proj_weight` 是 `[3D, D]`，
   与 heads 无关）⇒ heads 给错时 `load_state_dict` **依然会成功**，但模型算的是另一回事。
   判据 D 专门盯这条。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/analysis/tests/test_bn_calibration_arch.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[3]
for _p in (str(_GNN_ROOT), str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.bn_calibration import (  # noqa: E402
    build_model, load_model_for_inference, read_architecture,
)

C2_ARCH = {
    "class": "MagnetEgoQueryFusion",
    "gnn_pooling_fix": False,          # C2 命令行不带该 flag
    "gnn_pooling_fix_effective": True,  # 但融合隐含开了它（Q45）
    "context_pooling": False,
    "context_pool_mode": "max",
    "ego_query_fusion": True,
    "ego_query_heads": 4,
}


def _write_ckpt(d: Path, arch: dict | None, heads: int = 4) -> Path:
    """造一个 C2 架构的 bare state_dict + 可选 sidecar（模拟 calibrated 产物形态）。"""
    from magnet_ego_query_fusion import MagnetEgoQueryFusion
    torch.manual_seed(0)
    m = MagnetEgoQueryFusion(ego_query_fusion=True, ego_query_heads=heads)
    ckpt = d / "best_model_default.pth"
    torch.save(m.state_dict(), ckpt)
    if arch is not None:
        (d / "train_meta.json").write_text(
            json.dumps({"route_mode": "none", "architecture": arch}, ensure_ascii=False),
            encoding="utf-8")
    return ckpt


def test_build_model_supports_c2() -> None:
    """判据 A：`build_model` 能建出 C2 的类。"""
    m = build_model(ego_query_fusion=True, ego_query_heads=4)
    assert type(m).__name__ == "MagnetEgoQueryFusion", f"实得 {type(m).__name__}"
    # 老两态不受影响
    assert type(build_model()).__name__ == "TransformerPredictor"
    assert type(build_model(gnn_pooling_fix=True)).__name__ == "MagnetImitationPooled"
    print("[A] build_model 支持 C2 且老两态不变   PASS")


def test_load_c2_ckpt_via_sidecar() -> None:
    """判据 B（保护 C3）：bare state_dict + sidecar ⇒ 能加载，且不需要调用方知道架构。"""
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_ckpt(Path(d), C2_ARCH)
        m = load_model_for_inference(ckpt)          # 只给路径，架构全靠 sidecar
        assert type(m).__name__ == "MagnetEgoQueryFusion"
        assert not m.training, "必须是 eval 模式"
    print("[B] C2 ckpt 经 sidecar 可加载          PASS  (调用方无需知道架构)")


def test_missing_sidecar_raises_not_silent() -> None:
    """判据 C：没有 sidecar 时必须**抛错**，不许静默建成老架构后部分加载。"""
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_ckpt(Path(d), None)
        try:
            load_model_for_inference(ckpt)
        except RuntimeError:
            print("[C] 缺 sidecar 时抛错不静默            PASS")
            return
    raise AssertionError("C2 权重喂给老架构应当抛 RuntimeError（strict load）")


def test_wrong_heads_would_be_silent_hence_sidecar_is_mandatory() -> None:
    """判据 D：证明 `ego_query_heads` 给错时 `load_state_dict` **不会**报错。

    这条不是在测代码行为对不对，而是把「为什么 heads 必须来自 sidecar 而不能靠反推」
    固化成可执行的证据 —— 以后有人想把 sidecar 依赖去掉时，这条会拦住他。
    """
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_ckpt(Path(d), None, heads=8)      # 权重用 8 头训练
        m = build_model(ego_query_fusion=True, ego_query_heads=4)   # 却按 4 头重建
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))     # 不报错！
        assert m.context_pool.attn.num_heads == 4, "重建的确实是 4 头"
    print("[D] heads 给错不报错 ⇒ sidecar 必需     PASS  (8头权重装进4头模型不抛错)")


def test_sidecar_prefers_effective_flag() -> None:
    """判据 E：sidecar 里要用 `gnn_pooling_fix_effective` 而不是原始 flag（Q45）。

    C2 的命令行不带 `--gnn_pooling_fix`，原始 flag 记的是 False，
    但 GNN 分支实际是逐样本的 —— 按原始 flag 重建会建错分支语义。
    """
    arch = read_architecture(Path("/nonexistent"))
    assert arch is None, "路径不存在时应返回 None 而不是抛错"
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_ckpt(Path(d), C2_ARCH)
        got = read_architecture(ckpt)
        assert got["gnn_pooling_fix"] is False and got["gnn_pooling_fix_effective"] is True, \
            "测试数据本身要体现两者不同，否则这条判据没有意义"
        m = load_model_for_inference(ckpt)
        assert type(m).__name__ == "MagnetEgoQueryFusion"
    print("[E] 用 effective 而非原始 flag         PASS  (flag=False / effective=True)")


def test_explicit_override_wins() -> None:
    """判据 F：调用方显式传参优先于 sidecar（留给排障与消融）。"""
    with tempfile.TemporaryDirectory() as d:
        ckpt = _write_ckpt(Path(d), {**C2_ARCH, "ego_query_heads": 4})
        m = load_model_for_inference(ckpt, ego_query_heads=4)
        assert m.context_pool.attn.num_heads == 4
        # 显式关掉融合 ⇒ 架构变老类 ⇒ strict load 必然失败，证明 override 真的生效了
        try:
            load_model_for_inference(ckpt, ego_query_fusion=False)
        except RuntimeError:
            print("[F] 显式参数优先于 sidecar            PASS")
            return
    raise AssertionError("显式 ego_query_fusion=False 应当改变建模并触发 strict load 失败")


def test_arch_key_list_is_single_sourced() -> None:
    """判据 G（08-02 新增，锁住已发生过的事故）：架构键清单只能有一份。

    闸④ 的 calibration 在 08-02 崩掉，原因是 `bn_calibrate_checkpoint.py` 自己
    维护了一份键清单，漏了 `ego_shortcut_suppress` / `ego_state_dropout` ⇒
    `MagnetEgoShortcutSuppressed` 的 ckpt 被建成父类 `MagnetEgoQueryFusion`，
    `ego_state_emb.*` 无处安放，strict load 报 Unexpected key。

    本判据两件事：
      ① `ARCH_BUILD_KEYS` 必须覆盖 `build_model` 除 `gnn_pooling_fix` 外的全部形参
         —— 加了新开关却忘了登记，这里立刻红；
      ② 两个消费点都不许再出现硬编码的键元组。
    """
    import inspect
    import re

    from src.shared.utils.bn_calibration import ARCH_BUILD_KEYS

    params = set(inspect.signature(build_model).parameters) - {"gnn_pooling_fix"}
    missing = params - set(ARCH_BUILD_KEYS)
    assert not missing, f"build_model 有形参没登记进 ARCH_BUILD_KEYS: {sorted(missing)}"
    extra = set(ARCH_BUILD_KEYS) - params
    assert not extra, f"ARCH_BUILD_KEYS 有 build_model 不认识的键: {sorted(extra)}"

    consumer = _GNN_ROOT / "scripts/analysis/bn_calibrate_checkpoint.py"
    source = consumer.read_text(encoding="utf-8")
    hardcoded = re.search(r'for\s+k\s+in\s+\(\s*"context_pooling"', source)
    assert hardcoded is None, \
        f"{consumer.name} 又出现硬编码架构键清单，必须用 ARCH_BUILD_KEYS"
    print(f"[G] 架构键单一来源（{len(ARCH_BUILD_KEYS)} 个）    PASS  (闸④ 事故已锁)")


def test_gate4_ckpt_via_sidecar() -> None:
    """判据 H（08-02 新增）：闸④ 架构走同一条 sidecar 路径能重建。

    闸④ = `MagnetEgoShortcutSuppressed`，是 `MagnetEgoQueryFusion` 的**子类**，
    多出 `ego_state_emb.*` / `ego_transformer.*`。父子类都能建成、且不会互相串。
    """
    from magnet_ego_shortcut import MagnetEgoShortcutSuppressed

    gate4_arch = {
        "class": "MagnetEgoShortcutSuppressed",
        "gnn_pooling_fix": False,
        "gnn_pooling_fix_effective": True,
        "context_pooling": False,
        "context_pool_mode": "max",
        "ego_query_fusion": True,
        "ego_query_heads": 4,
        "ego_shortcut_suppress": True,
        "ego_state_dropout": 0.75,
    }
    with tempfile.TemporaryDirectory() as d:
        torch.manual_seed(0)
        model = MagnetEgoShortcutSuppressed(
            ego_query_fusion=True, ego_query_heads=4,
            use_ego_history=False, ego_state_dropout=0.75,
        )
        ckpt = Path(d) / "best_model_default.pth"
        torch.save(model.state_dict(), ckpt)
        (Path(d) / "train_meta.json").write_text(
            json.dumps({"route_mode": "none", "architecture": gate4_arch}, ensure_ascii=False),
            encoding="utf-8")

        loaded = load_model_for_inference(ckpt)
        assert type(loaded).__name__ == "MagnetEgoShortcutSuppressed", \
            f"实得 {type(loaded).__name__}（建成父类就是 08-02 那个 bug）"
    print("[H] 闸④ ckpt 经 sidecar 可加载        PASS  (子类不会被建成父类)")


if __name__ == "__main__":
    tests = [
        test_build_model_supports_c2,
        test_load_c2_ckpt_via_sidecar,
        test_missing_sidecar_raises_not_silent,
        test_wrong_heads_would_be_silent_hence_sidecar_is_mandatory,
        test_sidecar_prefers_effective_flag,
        test_explicit_override_wins,
        test_arch_key_list_is_single_sourced,
        test_gate4_ckpt_via_sidecar,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failed += 1
            print(f"[{t.__name__}] FAIL  {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} PASS")
    sys.exit(1 if failed else 0)
