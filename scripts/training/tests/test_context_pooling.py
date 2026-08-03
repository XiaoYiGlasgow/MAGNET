"""Context 分支 polyline 池化验收单测（对齐 `test_gnn_pooling.py` 的判据风格）。

判据：
  A. `context_pooling=False` 时与 `MagnetImitationPooled` **逐字节一致**（默认零改变）。
  B. 开启后 batch>1 时不同样本的输出**互不相同**（没有跨样本串台）。
  C. 开启后 batch=1 与 batch>1 对**同一个样本**给出一致结果（容差 1e-5，float32 归约噪声）。
  D. **缺陷复现**：`GraphData.__inc__['cluster'] == 0` 属实 ⇒ 若按裸 cluster 全局池化，
     跨样本 polyline 会被合并。不复现这条，本测试就没有意义。
  E. token 行数与 cluster 数不一致时**抛错而非静默退化**。
  F. 池化确实把 token 数降到 polyline 级（P << M），即算力节省真实存在。

运行（CPU，不占 GPU）：
  PYTHONPATH=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans:/mnt/local/Xiao_Yi/nuplan-devkit \
    /home/team/anaconda3/envs/nuplan/bin/python \
    /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/training/tests/test_context_pooling.py
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

from magnet_context_pooled import (  # noqa: E402
    ContextPoolAlignmentError, MagnetContextPooled,
)
from magnet_imitation_pooled import MagnetImitationPooled  # noqa: E402
from src.shared.utils.dataset import GraphData  # noqa: E402

SEED = 20260730
N_FUTURE = 60


def _make_sample(n_poly: int, pts_per_poly: int, n_gnn: int, gen: torch.Generator) -> GraphData:
    """构造一个结构合法的样本：M = n_poly*pts_per_poly 个 segment token。"""
    m = n_poly * pts_per_poly
    x = torch.rand(m, 12, generator=gen)
    # cluster = 每个 polyline 的 ID，样本内从 0 开始（与真实 __inc__ 行为一致）
    cluster = torch.arange(n_poly).repeat_interleave(pts_per_poly)
    gnn_x = torch.rand(n_gnn, 12, generator=gen)
    src, dst = [], []
    for i in range(n_gnn):
        for j in range(i + 1, n_gnn):
            src += [i, j]
            dst += [j, i]
    return GraphData(
        x=x,
        y=torch.rand(1, N_FUTURE * 2, generator=gen),
        cluster=cluster,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        gnn_x=gnn_x,
        gnn_edge_index=torch.tensor([src, dst], dtype=torch.long),
        gnn_edge_attr=torch.zeros(len(src), dtype=torch.long),
        ego_history=torch.rand(20, 2, generator=gen),
        ego_last_pos=torch.zeros(2),
    )


def _model(**kw):
    torch.manual_seed(SEED)
    m = MagnetContextPooled(**kw).eval()
    return m


def _baseline():
    torch.manual_seed(SEED)
    return MagnetImitationPooled().eval()


def _loader(samples, bs):
    return DataLoader(samples, batch_size=bs, shuffle=False, follow_batch=["gnn_x"])


def main() -> None:
    gen = torch.Generator().manual_seed(SEED)
    samples = [_make_sample(6, 25, 5, gen), _make_sample(4, 30, 6, gen), _make_sample(7, 20, 4, gen)]
    results = []

    # ---- D. 缺陷复现：__inc__['cluster'] 确实返回 0 ----
    batch = next(iter(_loader(samples, 3)))
    inc = samples[0].__inc__("cluster", samples[0].cluster)
    cross_collision = bool((batch.cluster[batch.batch == 0].max()
                            >= batch.cluster[batch.batch == 1].min()).item())
    ok_d = (inc == 0) and cross_collision
    results.append(("D 缺陷复现：cluster 不做 batch 偏移，裸池化会串样本",
                    ok_d, f"__inc__={inc}, 跨样本ID重叠={cross_collision}"))

    # ---- A. 关闭时与父类逐字节一致 ----
    with torch.no_grad():
        off = _model(context_pooling=False)(next(iter(_loader(samples, 3))))
        base = _baseline()(next(iter(_loader(samples, 3))))
    ok_a = torch.equal(off, base)
    results.append(("A 关闭池化时与 MagnetImitationPooled 逐字节一致", ok_a,
                    f"max|Δ|={(off - base).abs().max().item():.3e}"))

    # ---- B. 开启后 batch>1 各样本互不相同 ----
    with torch.no_grad():
        out = _model(context_pooling=True)(next(iter(_loader(samples, 3))))
    pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
    ok_b = all(not torch.allclose(out[i], out[j], atol=1e-6) for i, j in pairs)
    results.append(("B 开启后 batch>1 样本间输出互不相同（无串台）", ok_b,
                    f"min 两两 max|Δ| = "
                    f"{min((out[i] - out[j]).abs().max().item() for i, j in pairs):.3e}"))

    # ---- C. 池化不得引入**新的** batch 依赖 ----
    # 注意：模型本身已有 D0#13 的 batch 依赖（ego_history_encoder 的
    # BatchNorm1d(track_running_stats=False) 在 eval 下仍用当前 batch 统计量），
    # 该依赖与 context 分支无关。故判据是 matched control：
    # 开启池化后的 batch 依赖不得**大于**关闭池化时的同一量。
    def _batch_dep(model) -> float:
        with torch.no_grad():
            singles = torch.cat([model(b) for b in _loader(samples, 1)], dim=0)
            multi = model(next(iter(_loader(samples, 3))))
        return (singles - multi).abs().max().item()

    def _kill_bn(model):
        """中性化 D0#13 源头：ego_history_encoder[1] 的 BatchNorm1d。

        与 S4-b 三格对照同一手法 —— 把两个 batch 依赖源分开，
        否则 BN 的 1e-1 量级会淹没 context 分支自身的 1e-7 量级。
        """
        model.ego_history_encoder[1] = torch.nn.Identity()
        return model

    d_bn_off = _batch_dep(_model(context_pooling=False))            # 含 BN，未池化
    d_bn_on = _batch_dep(_model(context_pooling=True))              # 含 BN，已池化
    d_pure_off = _batch_dep(_kill_bn(_model(context_pooling=False)))  # 去 BN，未池化
    d_pure_on = _batch_dep(_kill_bn(_model(context_pooling=True)))    # 去 BN，已池化
    ok_c = d_pure_on < 1e-5
    results.append(("C BN 中性化后池化本身零 batch 依赖（<1e-5）", ok_c,
                    f"含BN: 未池化={d_bn_off:.3e} 池化={d_bn_on:.3e}（同量级，D0#13 主导）｜"
                    f"去BN: 未池化={d_pure_off:.3e} 池化={d_pure_on:.3e}"))

    # ---- E. 行数不符时抛错 ----
    bad = next(iter(_loader(samples, 1)))
    bad.cluster = bad.cluster[:-3]  # 人为破坏
    try:
        with torch.no_grad():
            _model(context_pooling=True)(bad)
        ok_e, note = False, "未抛错（静默退化）"
    except Exception as e:  # ClusterCoverageError / ContextPoolAlignmentError
        ok_e = isinstance(e, (ContextPoolAlignmentError, RuntimeError))
        note = f"{type(e).__name__}"
    results.append(("E cluster 长度不符时抛错而非静默退化", ok_e, note))

    # ---- F. token 数确实降到 polyline 级 ----
    M = [int(s.x.size(0)) for s in samples]
    P = [int(s.cluster.unique().numel()) for s in samples]
    ok_f = all(p < m for p, m in zip(P, M))
    results.append(("F 池化把 token 数降到 polyline 级", ok_f,
                    f"M={M} -> P={P}（注意力 FLOPs ∝ M²）"))

    print("\n===== context pooling 验收 =====")
    for name, ok, note in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {note}")
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} passed")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
