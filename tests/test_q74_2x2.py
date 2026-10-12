#!/usr/bin/env python3
"""Q74 2×2 通路分解的两道起跑前置门（判据 §2-2，🔴 先写后跑，门⑥收编）。

冻结判据: docs/work_records/paper/MAGNET/VT-2026-04191_修改工作记录/
Q74_2x2通路分解_预注册判据_20260810_冻结.md

  [V1] 空操作等价：删除集为空时，四格（G1/G2/G3/G4）输出与 base **逐位相同**
       （`torch.equal` 级 —— 模型确定性 ⇒ 不存在浮点容差问题）；
  [V2] 通路隔离复检：非空删除集下，
       G2（仅删 GNN 侧）的 `context_feat` max|Δ| = 0（阶段85 已测，此处回归门化）；
       G3（仅删 context 侧）的 `gnn_feat` max|Δ| = 0（新验证，防 context 删行经共享
       `feature_norm` 泄漏 —— 该 LayerNorm 逐行、理论上不泄漏，但必须实测）。
       顺带断言两格都不动 `ego_feat`，并要求干预**打中了自己的通路**
       （G2 改变 gnn_feat / G3 改变 context_feat 至少各命中一次 —— 防「假隔离」：
       两边都是 0 也能通过隔离断言，但那说明干预根本没接上）。

输入：3-token 迷你清单（`head -3 shards_r1_L1/shard_00.txt` 落到临时文件），CPU。
ckpt：k0_seed42 best（判据 §2-6 的主套；路径查实依据见下方常量注释）。

三个分支特征经 `feature_fusion` 的 forward_pre_hook 截获：
输入是 cat([ego_feat, context_feat, gnn_feat]) [B, 384]，切片即得
（与 `scripts/analysis/verify_pathway_split.py` 的 BranchTap 同构）。

用法::

    /home/team/anaconda3/envs/nuplan/bin/python -m pytest tests/test_q74_2x2.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
if str(_GNN) not in sys.path:
    sys.path.insert(0, str(_GNN))

torch.set_num_threads(2)    # CPU 余量：守卫 C 正用 6 worker，本测试 ≤2 线程

from scripts.analysis import q74_2x2_audit as q74  # noqa: E402

#: k0_seed42 best —— 查实依据：
#: ① `scratchpad/k0_audit_worker_20260807.sh:26` 生产 K0 audit 用的正是
#:    `experiments/knobs/k0_poolfix_L1/seed${SEED}/models/best_model_default.pth`；
#: ② 同目录 sidecar `train_meta.json`：class=MagnetImitationPooled、
#:    gnn_pooling_fix_effective=true（判据 §2-6「fix=True，主」逐字吻合）。
CKPT = _GNN / "experiments/knobs/k0_poolfix_L1/seed42/models/best_model_default.pth"

#: 与 k0 audit worker 的 --raw_dir 一致（token 来自 shards_r1_L1，与该档逐一命中）。
RAW = _GNN / "data/processed_features_scaling_L1_logsplit/val"
SHARD00 = _GNN / "experiments/interaction_audit/shards_r1_L1/shard_00.txt"

DEVICE = torch.device("cpu")

#: feature_fusion 输入 [B,384] 的三段切片（与 verify_pathway_split.py 逐字一致）。
EGO_SL, CTX_SL, GNN_SL = slice(0, 128), slice(128, 256), slice(256, 384)


class _Tap:
    """挂在 `feature_fusion` 上截获 cat([ego, context, gnn])（只读，不改前向）。"""

    def __init__(self, model) -> None:
        self.captured = None
        self.handle = model.feature_fusion.register_forward_pre_hook(self._hook)

    def _hook(self, module, args) -> None:
        self.captured = args[0].detach().clone()

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def _capture(model, tap: _Tap, data) -> torch.Tensor:
    """一次 batch=1 前向，返回截获的 [1, 384] 三分支特征。"""
    from torch_geometric.data import Batch
    model(Batch.from_data_list([data]).to(DEVICE))
    assert tap.captured is not None and tap.captured.shape[1] == 384
    return tap.captured.clone()


@pytest.fixture(scope="module")
def mini_tokens(tmp_path_factory) -> Path:
    """迷你 3-token 清单：head -3 shard_00.txt 落到临时文件（判据允许的单测输入）。"""
    assert SHARD00.exists(), f"{SHARD00} 不在盘"
    toks = [t.strip() for t in SHARD00.read_text().splitlines() if t.strip()][:3]
    assert len(toks) == 3, f"shard_00 取不满 3 个 token：{toks}"
    f = tmp_path_factory.mktemp("q74_mini") / "mini3.txt"
    f.write_text("\n".join(toks) + "\n")
    return f


@pytest.fixture(scope="module")
def model():
    """k0_seed42 best，经与生产 audit 同一入口加载（sidecar 决定类与 fix）。"""
    assert CKPT.exists(), f"{CKPT} 不在盘 —— k0_seed42 best 缺失"
    agg = q74.configure_subgraph_agg(CKPT)
    assert agg == "mean", f"k0 sidecar 不该带非 mean 聚合，实得 {agg}"
    m = q74._load_model(CKPT, False, DEVICE, temporal_graph=False)
    assert type(m).__name__ == "MagnetImitationPooled", (
        f"k0 ckpt 应建成 MagnetImitationPooled（sidecar），实得 {type(m).__name__}")
    return m


@pytest.fixture(scope="module")
def samples(mini_tokens):
    """3 个 token 的审计输入（含 context 映射两道自检，任一 skip 即测试失败）。"""
    out = []
    for tok in mini_tokens.read_text().split():
        s, reason = q74.load_token_sample(RAW, tok, "default")
        assert s is not None, f"{tok}: 构建失败（{reason}）—— 迷你输入必须 3/3 可用"
        out.append(s)
    return out


def test_v1_noop_bitwise_equivalence(model, samples) -> None:
    """[V1] 空删除集下四格输出与 base 逐位相同（torch.equal 级）。"""
    empty = np.zeros(0, dtype=np.int64)
    for s in samples:
        base = q74._forward_once(model, s.data, DEVICE)
        for cell in q74.CELLS:
            cell_data = q74.make_cell(s.data, cell, empty, s.node_order, s.ent_rows)
            out = q74._forward_once(model, cell_data, DEVICE)
            assert torch.equal(out, base), (
                f"[V1] {s.token} {cell}: 空操作输出与 base 不逐位相同，"
                f"max|Δ|={float((out - base).abs().max()):.3e}")
        # 运行时门函数本身也必须静默通过（生产路径用的是它）
        q74.noop_gate(model, s.data, s.node_order, s.ent_rows, DEVICE)
    print(f"  ✅ [V1] {len(samples)} token × 四格：空操作逐位等价（torch.equal）")


def test_v2_pathway_isolation(model, samples) -> None:
    """[V2] 非空删除集下的通路隔离：G2 不动 context_feat，G3 不动 gnn_feat。

    删除集 = count_sat 剂量在 audit seed 1000 下的选点（与生产口径同一剂量合同、
    同一 rng 消耗顺序），target(agent) 与 control(lane) 两侧都测。
    若 G3 下 gnn_feat 出现非零 Δ ⇒ 真实泄漏：按任务纪律**只报数值不改口径**，
    断言消息里带 max|Δ| 供裁定。
    """
    count_sat = q74.DOSES[0]
    assert count_sat.name == "count_sat", "冻结剂量表首位应为 count_sat"

    n_checked = n_hit_gnn = n_hit_ctx = 0
    tap = _Tap(model)
    try:
        for s in samples:
            types = np.array(s.node_types)
            pool_t = np.flatnonzero((types == q74.TARGET_TYPE) & ~s.is_ego)
            pool_c = np.flatnonzero((types == q74.CONTROL_TYPE) & ~s.is_ego)
            k_t, k_c, skip = q74._resolve_dose(count_sat, pool_t.size, pool_c.size)
            if skip is not None:
                continue
            rng = np.random.default_rng(1000)   # 协议首个 audit seed，消耗顺序同生产
            pick_t = rng.choice(pool_t, size=k_t, replace=False)
            pick_c = rng.choice(pool_c, size=k_c, replace=False)

            f_base = _capture(model, tap, s.data)
            for side, pick in (("agent", pick_t), ("lane", pick_c)):
                assert pick.size > 0
                f_g2 = _capture(model, tap,
                                q74.make_cell(s.data, "G2", pick, s.node_order, s.ent_rows))
                f_g3 = _capture(model, tap,
                                q74.make_cell(s.data, "G3", pick, s.node_order, s.ent_rows))

                def _d(f: torch.Tensor, sl: slice) -> float:
                    return float((f[:, sl] - f_base[:, sl]).abs().max())

                assert _d(f_g2, CTX_SL) == 0.0, (
                    f"[V2] {s.token}/{side}: G2（仅删 GNN 侧）改变了 context_feat，"
                    f"max|Δ|={_d(f_g2, CTX_SL):.3e} —— 通路隔离被破坏")
                assert _d(f_g3, GNN_SL) == 0.0, (
                    f"[V2] {s.token}/{side}: G3（仅删 context 侧）改变了 gnn_feat，"
                    f"max|Δ|={_d(f_g3, GNN_SL):.3e} —— 疑经共享 feature_norm 泄漏，"
                    "🔴 不改口径、报此数值等裁定")
                assert _d(f_g2, EGO_SL) == 0.0, (
                    f"[V2] {s.token}/{side}: G2 动了 ego_feat，max|Δ|={_d(f_g2, EGO_SL):.3e}")
                assert _d(f_g3, EGO_SL) == 0.0, (
                    f"[V2] {s.token}/{side}: G3 动了 ego_feat，max|Δ|={_d(f_g3, EGO_SL):.3e}")

                n_hit_gnn += int(_d(f_g2, GNN_SL) > 0.0)
                n_hit_ctx += int(_d(f_g3, CTX_SL) > 0.0)
                n_checked += 1
    finally:
        tap.close()

    assert n_checked > 0, "count_sat 在 3 个迷你 token 上全部 skip —— V2 没有非空删除集可测"
    assert n_hit_gnn > 0, "G2 从未改变 gnn_feat —— 干预没打中 GNN 通路（假隔离，接线有误）"
    assert n_hit_ctx > 0, "G3 从未改变 context_feat —— 干预没打中 context 通路（假隔离，接线有误）"
    print(f"  ✅ [V2] {n_checked} 组（token×两侧）：G2 context_feat Δ=0、G3 gnn_feat Δ=0、"
          f"ego_feat 两格均 Δ=0；自通路命中 gnn {n_hit_gnn}/ctx {n_hit_ctx}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
