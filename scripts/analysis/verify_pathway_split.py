"""核实：C1 干预（只改 gnn_*）是否影响 context 通路。

做三件事，全部是断言，不下结论：
  [A] 施加与 run_interaction_audit.py:202/209 完全相同的干预后，data.x 是否逐位不变
  [B] 干预前后 ego_feat / context_feat / gnn_feat 三段各自的 max|Δ|
  [C] 通路置零的 2x2 demo：整条通路关掉后输出轨迹的位移

三个分支特征通过 feature_fusion 的 forward_pre_hook 取：
输入是 cat([ego_feat, context_feat, gnn_feat]) [B,384]，切片即得。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
sys.path.insert(0, str(_REPO))

from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402
from src.shared.utils.bn_calibration import load_model_for_inference  # noqa: E402

CKPT = _REPO / "experiments/ablation_2x2_f1/dev/B_seed42/models/checkpoint_best.pt"
RAW = _REPO / "data/processed_features_magnet_e1_none_v3_split3/val"
N_TOKENS = 20
DEVICE = torch.device("cpu")


class BranchTap:
    """挂在 feature_fusion 上，截获 cat([ego, context, gnn]) 并可选地把某几段置零。"""

    def __init__(self, model):
        self.captured = None
        self.zero_slices = ()
        self.handle = model.feature_fusion.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = args[0]
        self.captured = x.detach().clone()
        if not self.zero_slices:
            return None
        x = x.clone()
        for lo, hi in self.zero_slices:
            x[:, lo:hi] = 0.0
        return (x,)

    def close(self):
        self.handle.remove()


EGO, CTX, GNN = (0, 128), (128, 256), (256, 384)


def clone_with(data, **overrides):
    """与 run_interaction_audit.py:138-143 逐字同构。"""
    out = data.clone()
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


def main():
    model = load_model_for_inference(CKPT, gnn_pooling_fix=False, device=DEVICE)
    model.eval()
    tap = BranchTap(model)

    pkls = sorted(RAW.glob("features_*.pkl"))[:N_TOKENS]
    print(f"ckpt = {CKPT.name}   tokens = {len(pkls)}   device = cpu\n")

    rows = []
    for p in pkls:
        df = pd.read_pickle(p)
        built = build_graph_data_from_feature_df(df)
        data = built[0] if isinstance(built, tuple) else built
        if data is None or not hasattr(data, "gnn_x") or data.gnn_x.numel() == 0:
            continue
        n_nodes = data.gnn_x.shape[0]
        if n_nodes < 3:
            continue

        # ---- 干预：删掉除节点 0(ego) 外的全部 GNN 节点，等价于 C1 的 frac=1.0 remove ----
        keep = torch.zeros(n_nodes, dtype=torch.bool)
        keep[0] = True
        idx_map = -torch.ones(n_nodes, dtype=torch.long)
        idx_map[keep] = torch.arange(int(keep.sum()))
        ei = data.gnn_edge_index
        emask = keep[ei[0]] & keep[ei[1]]
        new_ei = idx_map[ei[:, emask]]
        interv = clone_with(
            data,
            gnn_x=data.gnn_x[keep],
            gnn_edge_index=new_ei,
            gnn_edge_attr=data.gnn_edge_attr.new_zeros(new_ei.shape[1]),
        )

        # [A] data.x 是否被动过
        dx = float((interv.x - data.x).abs().max())

        # [B] 三分支特征的变化
        tap.zero_slices = ()
        with torch.no_grad():
            y_base = model(data)
        f_base = tap.captured.clone()
        with torch.no_grad():
            y_int = model(interv)
        f_int = tap.captured.clone()

        d_ego = float((f_int[:, EGO[0]:EGO[1]] - f_base[:, EGO[0]:EGO[1]]).abs().max())
        d_ctx = float((f_int[:, CTX[0]:CTX[1]] - f_base[:, CTX[0]:CTX[1]]).abs().max())
        d_gnn = float((f_int[:, GNN[0]:GNN[1]] - f_base[:, GNN[0]:GNN[1]]).abs().max())

        def disp(a, b):
            a = a.view(-1, 60, 2).cumsum(1)
            b = b.view(-1, 60, 2).cumsum(1)
            return float((a - b).norm(dim=-1).mean())

        d_traj_c1 = disp(y_int, y_base)

        # [C] 通路置零 2x2（整条通路关掉，非按实体删）
        outs = {}
        for name, sl in [("gnn_off", (GNN,)), ("ctx_off", (CTX,)), ("both_off", (CTX, GNN))]:
            tap.zero_slices = sl
            with torch.no_grad():
                outs[name] = model(data)
        tap.zero_slices = ()

        base_norm = float(y_base.view(-1, 60, 2).cumsum(1).norm(dim=-1).mean()) + 1e-9
        rows.append(dict(
            token=p.stem[9:17], n=n_nodes,
            dx=dx, d_ego=d_ego, d_ctx=d_ctx, d_gnn=d_gnn,
            c1_disp=d_traj_c1, c1_rel=d_traj_c1 / base_norm,
            gnn_off=disp(outs["gnn_off"], y_base) / base_norm,
            ctx_off=disp(outs["ctx_off"], y_base) / base_norm,
            both_off=disp(outs["both_off"], y_base) / base_norm,
        ))

    tap.close()
    R = pd.DataFrame(rows)
    if R.empty:
        print("没有可用样本")
        return

    print("=" * 78)
    print("[A] 干预后 data.x 的 max|Δ|")
    print(f"    全部 {len(R)} 个样本： max = {R.dx.max():.3e}   "
          f"{'⇒ data.x 逐位未变' if R.dx.max() == 0 else '⇒ data.x 被改动了'}")
    print()
    print("[B] 干预前后三分支特征 max|Δ| （中位数 / 最大值）")
    for k, lab in [("d_ego", "ego_feat    "), ("d_ctx", "context_feat"), ("d_gnn", "gnn_feat    ")]:
        print(f"    {lab}  median={R[k].median():.3e}   max={R[k].max():.3e}")
    print()
    print("[C] 相对位移（占基线轨迹长度的比例，中位数）")
    print(f"    C1 式干预（只删 gnn 节点）      : {R.c1_rel.median():.4%}")
    print(f"    整条 GNN 通路置零              : {R.gnn_off.median():.4%}")
    print(f"    整条 context 通路置零          : {R.ctx_off.median():.4%}")
    print(f"    两条通路同时置零               : {R.both_off.median():.4%}")
    print("=" * 78)
    R.to_csv("/tmp/claude-1001/-mnt-local-Xiao-Yi/84bc7812-49dc-42f9-b0ac-8c485e171fcf/scratchpad/pathway_verify.csv", index=False)


if __name__ == "__main__":
    main()
