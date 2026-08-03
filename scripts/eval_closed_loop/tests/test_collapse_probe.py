"""Q46 塌陷探针单测 —— 核心是证明「开探针不改变任何输出」。

为什么这条判据最重要
--------------------
探针要挂在 **C3 这次决定性闭环**的旁边跑。若它哪怕改动一个 bit 的模型输出，
带探针那组就不再是同一个实验，整条 C1→C2→C3 的证据链断掉。
所以判据 A 直接对**同一批输入**逐位比对开/关探针的模型输出，要求 `max|Δ| == 0`。

其余判据覆盖：挂点正确、塌陷判定语义正确、落盘格式含 join key、挂不上必须抛错。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python scripts/eval_closed_loop/tests/test_collapse_probe.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/local/Xiao_Yi/nuplan-devkit")

from nuplan.planning.simulation.planner.gnn_trans_planner import (  # noqa: E402
    GNNTransPlanner, _CollapseProbe,
)


class _FakeModel(nn.Module):
    """最小复刻 `TransformerPredictor` 里探针关心的那一段：`output_head` 末尾是 ReLU→Linear。"""

    def __init__(self, force_collapse: bool = False) -> None:
        super().__init__()
        self.output_head = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 4),
        )
        if force_collapse:
            # 让末个 ReLU 前的 Linear 恒输出负值 ⇒ ReLU 全零 ⇒ 末层退化为纯 bias
            with torch.no_grad():
                self.output_head[2].weight.fill_(0.0)
                self.output_head[2].bias.fill_(-10.0)

    def forward(self, x):
        return self.output_head(x)


def test_probe_does_not_change_output() -> None:
    """判据 A（最重要）：挂上探针后模型输出**逐位不变**。"""
    torch.manual_seed(0)
    model = _FakeModel()
    x = torch.randn(3, 8)

    with torch.no_grad():
        before = model(x).clone()

    with tempfile.TemporaryDirectory() as d:
        probe = _CollapseProbe(Path(d) / "p.jsonl")
        probe.attach(model)
        with torch.no_grad():
            after = model(x).clone()

    delta = float((before - after).abs().max())
    assert delta == 0.0, f"探针改变了输出 max|Δ|={delta:.3e}"
    print(f"[A] 开探针输出逐位不变            PASS  max|Δ|={delta:.1e}")


def test_probe_detects_collapse() -> None:
    """判据 B：末个 ReLU 恒零时判为塌陷，正常时判为未塌陷。"""
    with tempfile.TemporaryDirectory() as d:
        for force, expect in ((True, True), (False, False)):
            torch.manual_seed(1)
            model = _FakeModel(force_collapse=force)
            probe = _CollapseProbe(Path(d) / f"p{force}.jsonl")
            probe.attach(model)
            with torch.no_grad():
                model(torch.randn(2, 8))
            got = probe._last["collapsed"]
            assert got is expect, f"force_collapse={force} 期望 collapsed={expect}，实得 {got}"
    print("[B] 塌陷判定语义正确              PASS  (恒零->True / 正常->False)")


def test_record_writes_join_key() -> None:
    """判据 C：每行 JSONL 必须带 token/log_name/scenario_type + iteration，否则无法与闭环分数 join。"""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.jsonl"
        torch.manual_seed(2)
        model = _FakeModel()
        probe = _CollapseProbe(out)
        probe.attach(model)
        sid = {"token": "abc123", "log_name": "some_log", "scenario_type": "following_lane"}
        for it in range(3):
            with torch.no_grad():
                model(torch.randn(1, 8))
            probe.record(sid, it)

        rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3, f"应写 3 行，实得 {len(rows)}"
    for i, r in enumerate(rows):
        assert r["iteration"] == i
        for k in ("token", "log_name", "scenario_type", "collapsed", "zero_frac"):
            assert k in r, f"第 {i} 行缺字段 {k}"
    print(f"[C] JSONL 含完整 join key         PASS  ({len(rows)} 行，字段齐全)")


def test_record_never_silently_drops_a_step() -> None:
    """判据 D：没捕获到激活也要写一行（`collapsed=None`），**不允许静默丢步**。

    丢步会让「塌陷率」的分母偷偷变小，是最难发现的一类偏差。
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "s.jsonl"
        probe = _CollapseProbe(out)          # 故意不 attach
        probe.record({"token": "t"}, 0)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["collapsed"] is None, f"实得 {rows}"
    print("[D] 未捕获也写一行不丢步          PASS  (collapsed=None)")


def test_attach_raises_when_no_relu() -> None:
    """判据 E：挂不上必须抛错。静默失效 = 事后拿到一份看着正常的空证据。"""
    class NoRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_head = nn.Sequential(nn.Linear(4, 4))

    with tempfile.TemporaryDirectory() as d:
        probe = _CollapseProbe(Path(d) / "p.jsonl")
        try:
            probe.attach(NoRelu())
        except RuntimeError:
            print("[E] 挂不上时抛错不静默            PASS")
            return
    raise AssertionError("output_head 无 ReLU 时应抛 RuntimeError")


def test_planner_defaults_probe_off() -> None:
    """判据 F：不传 `collapse_probe_dir` 且无环境变量时探针为 None ⇒ 与加入本参数之前行为一致。

    同时确认 `requires_scenario=True`（否则 `build_planners` 不会注入 scenario，
    落盘就没有 token，join 不上闭环分数）。
    """
    import os
    assert GNNTransPlanner.requires_scenario is True, "requires_scenario 必须为 True 才能拿到 token"
    old = os.environ.pop("MAGNET_COLLAPSE_PROBE_DIR", None)
    try:
        p = GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu")
        assert p._probe is None and p._collapse_probe_dir is None, "默认必须是关闭的"
        assert p._scenario_id == {"token": None, "log_name": None, "scenario_type": None}
    finally:
        if old is not None:
            os.environ["MAGNET_COLLAPSE_PROBE_DIR"] = old
    print("[F] 默认关闭 + requires_scenario  PASS")


def test_env_var_enables_probe() -> None:
    """判据 G：环境变量 `MAGNET_COLLAPSE_PROBE_DIR` 能开启探针，yaml 参数优先级更高。

    这条路径存在的理由：`run_closed_loop.sh` 没有通用 override 透传，
    而改它会动 harness sha 并触发规则基线四格重跑。
    """
    import os
    old = os.environ.get("MAGNET_COLLAPSE_PROBE_DIR")
    try:
        os.environ["MAGNET_COLLAPSE_PROBE_DIR"] = "/tmp/from_env"
        p = GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu")
        assert p._collapse_probe_dir == "/tmp/from_env", f"env var 未生效，实得 {p._collapse_probe_dir}"
        # yaml 参数优先
        p2 = GNNTransPlanner(checkpoint_path="/nonexistent.pth", device="cpu",
                             collapse_probe_dir="/tmp/from_yaml")
        assert p2._collapse_probe_dir == "/tmp/from_yaml", f"yaml 未优先，实得 {p2._collapse_probe_dir}"
    finally:
        os.environ.pop("MAGNET_COLLAPSE_PROBE_DIR", None)
        if old is not None:
            os.environ["MAGNET_COLLAPSE_PROBE_DIR"] = old
    print("[G] env var 可开启 + yaml 优先    PASS")


def test_official_build_planners_path() -> None:
    """判据 H（保护 C3）：走**官方 `build_planners`**（`simulation_builder.py:90` 那条路）能实例化。

    `requires_scenario=True` 会让 hydra 以 `instantiate(config, scenario=scenario)` 调用构造函数。
    若签名不接受它，整条闭环在第一个场景就崩 —— 这条测试是 C3 起跑前的唯一防线。
    """
    from omegaconf import OmegaConf
    from nuplan.planning.script.builders.planner_builder import build_planners

    cfg = OmegaConf.load(
        "/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/planning/script/config/simulation/planner/gnn_trans_planner.yaml"
    )

    class FakeScenario:
        token = "deadbeef12345678"
        log_name = "some_log"
        scenario_type = "following_lane_with_lead"

    p = build_planners(cfg, FakeScenario())[0]
    assert type(p).__name__ == "GNNTransPlanner"
    assert p._scenario_id["token"] == "deadbeef12345678", "scenario 未被注入，落盘会没有 join key"
    assert p._probe is None, "默认必须关闭"
    print("[H] 官方 build_planners 路径可用  PASS  (token 已注入)")


if __name__ == "__main__":
    tests = [
        test_probe_does_not_change_output,
        test_probe_detects_collapse,
        test_record_writes_join_key,
        test_record_never_silently_drops_a_step,
        test_attach_raises_when_no_relu,
        test_planner_defaults_probe_off,
        test_env_var_enables_probe,
        test_official_build_planners_path,
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
