"""J1b —— **逐时域误差曲线**：旧 MAGNET vs 常速基线（ego 侧，冻结 test，位置域）。

## 它回答什么

J1 实测「旧 MAGNET 的 ego **ADE** 与常速基线打平（2.8318 vs 2.8515）而 **FDE** 好 10.5%
（7.0245 vs 7.8130）」。两者第 0 帧都恰在原点（MAGNET 走 `cumsum`、常速走 `p0 + v0·k·dt`）
⇒ 锚点相同 ⇒ FDE 好而 ADE 平**只能**意味着 MAGNET 在**中段帧**上劣于常速。
本脚本把这句推断变成逐帧读数：**学出来的模型从第几秒开始才挣钱**。

## 口径（与 J1 / `eval_test_set.py` 逐字一致，不得自行放宽）

* 冻结 test、**batch=8**（ego `BatchNorm(track_running_stats=False)` 使读数随 batch 变）；
* 位置域：模型输出是逐帧增量 ⇒ `cumsum`；GT 同样 `cumsum`；
* 架构由 ckpt sidecar 决定（`load_model_for_inference`），**不裸建**；
* `follow_batch=['gnn_x']` 按 sidecar 的 `gnn_pooling_fix_effective` 决定（D0#7 同族缺陷）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_THIS = Path(__file__).resolve()
GNN_ROOT = _THIS.parents[2]
for _p in (str(GNN_ROOT), str(GNN_ROOT / "scripts" / "training"),
           str(GNN_ROOT.parent), str(GNN_ROOT.parent / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from src.shared.utils.bn_calibration import load_model_for_inference, read_architecture  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config, set_subgraph_agg  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

logger = logging.getLogger("j1b")
DT = 0.1
EGO_HISTORY_FRAMES = 20


def _cv_positions(data, n_graphs: int, n_future: int, window: int = 1) -> torch.Tensor:
    """常速外推的**位置域**轨迹 `[B, 60, 2]`，锚点 = 原点（ego 自身坐标系）。

    ego target 的第 0 帧实测逐位在原点 ⇒ 时间基取 `t_k = k·dt`（J1 已核）。
    """
    eh = data.ego_history
    if eh.dim() == 3:
        eh = eh.reshape(-1, eh.shape[-1])
    seq = eh[:, :2].view(n_graphs, EGO_HISTORY_FRAMES, 2)
    v0 = seq[:, -window:].mean(dim=1) / DT                                   # [B, 2]
    t = torch.arange(n_future, device=v0.device, dtype=v0.dtype) * DT        # [60]
    return v0.unsqueeze(1) * t.view(1, -1, 1)


def run_one(ckpt: Path, test_root: Path, batch_size: int, device: str) -> Dict[str, object]:
    sidecar = read_architecture(ckpt) or {}
    set_subgraph_agg(sidecar.get("subgraph_agg") or "mean")
    meta_path = ckpt.parent / "train_meta.json"
    train_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dataset_config = train_meta.get("dataset_config") or "default"
    builder = get_adjacency_config(dataset_config)
    if dataset_config != "default" and type(builder).__name__ == "DefaultBuilder":
        raise SystemExit(f"邻接配置 {dataset_config!r} 未注册（会静默回退 DefaultBuilder）⇒ 拒绝评测")

    ds = GraphDataset(str(test_root), transform=None)
    follow = ["gnn_x"] if sidecar.get("gnn_pooling_fix_effective",
                                      sidecar.get("gnn_pooling_fix", False)) else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, follow_batch=follow)
    dev = torch.device(device)
    model = load_model_for_inference(ckpt, gnn_pooling_fix=False, device=dev).to(dev).eval()

    n_f = config.NUM_FUTURE_POSES
    s_model = torch.zeros(n_f, dtype=torch.float64)
    s_cv = torch.zeros(n_f, dtype=torch.float64)
    s_still = torch.zeros(n_f, dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(dev)
            gt = torch.cumsum(batch.y.reshape(-1, n_f, 2), dim=1)
            b = gt.shape[0]
            pred = torch.cumsum(model(batch).reshape(-1, n_f, 2), dim=1)
            cv = _cv_positions(batch, b, n_f)
            s_model += torch.norm(pred - gt, dim=-1).sum(0).double().cpu()
            s_cv += torch.norm(cv - gt, dim=-1).sum(0).double().cpu()
            s_still += torch.norm(gt, dim=-1).sum(0).double().cpu()
            n += b
    out = {"checkpoint": str(ckpt), "n_samples": n, "dataset_config": dataset_config,
           "batch_size": batch_size,
           "model_curve": (s_model / n).tolist(), "cv_curve": (s_cv / n).tolist(),
           "stay_still_curve": (s_still / n).tolist()}
    for tag, cur in (("model", s_model), ("cv", s_cv), ("stay_still", s_still)):
        c = (cur / n).numpy()
        out[f"{tag}_ADE"] = float(c.mean())
        out[f"{tag}_FDE"] = float(c[-1])
        # 🔴 索引 i ↔ t0 + i·0.1 s（`cumsum` 首帧是锚点）⇒ 索引 9 = 0.9 s、索引 59 = 5.9 s。
        #    项目惯例把索引 59 报作「6 s」（FDE），此处键名用**真实时刻**。
        out[f"{tag}_at_s"] = {f"{i * 0.1:.1f}": float(c[i]) for i in (9, 19, 29, 39, 49, 59)}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description="J1b 逐时域误差曲线（旧 MAGNET vs 常速）")
    p.add_argument("--checkpoint", nargs="+", required=True, help="一个或多个 ckpt（多 seed）")
    p.add_argument("--test_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()

    runs: List[Dict[str, object]] = []
    for c in a.checkpoint:
        logger.info("评 %s", c)
        runs.append(run_one(Path(c).resolve(), Path(a.test_root).resolve(), a.batch_size, a.device))

    mc = np.mean([r["model_curve"] for r in runs], axis=0)
    cv = np.asarray(runs[0]["cv_curve"])                 # 基线与 ckpt 无关，取第一个即可
    st = np.asarray(runs[0]["stay_still_curve"])
    agg = {"n_ckpt": len(runs), "model_curve_mean": mc.tolist(), "cv_curve": cv.tolist(),
           "stay_still_curve": st.tolist(),
           "model_ADE_mean": float(mc.mean()), "cv_ADE": float(cv.mean()),
           "model_FDE_mean": float(mc[-1]), "cv_FDE": float(cv[-1]),
           "crossover_frame": int(np.argmax(mc < cv)) if bool((mc < cv).any()) else None,
           "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    print(f"\n=== 逐时域误差（冻结 test，{runs[0]['n_samples']} 样本，batch={a.batch_size}，"
          f"{len(runs)} 个 ckpt 平均）===")
    print(f"{'t (s)':>7s} {'MAGNET':>9s} {'常速':>9s} {'原地不动':>10s} {'MAGNET/常速':>12s}"
          f"   ← 🔴 t = 索引×0.1 s；末点 5.9 s 即惯例的「6 s / FDE」")
    for i in (9, 19, 29, 39, 49, 59):
        print(f"{i * 0.1:7.1f} {mc[i]:9.4f} {cv[i]:9.4f} {st[i]:10.4f} {mc[i] / cv[i]:12.3f}")
    print(f"{'ADE':>7s} {mc.mean():9.4f} {cv.mean():9.4f} {st.mean():10.4f} "
          f"{mc.mean() / cv.mean():12.3f}")
    xo = agg["crossover_frame"]
    print(f"\n🔑 MAGNET 首次优于常速的帧 = {xo}（t = {xo * 0.1:.1f} s）"
          if xo is not None else "\n🔑 MAGNET 在任何帧上都未优于常速")
    print(f"落盘 {a.out}")


if __name__ == "__main__":
    main()
