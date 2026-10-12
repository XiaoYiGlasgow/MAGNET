"""独立 test set 评测器（VT-2026-04191 阶段⑬，回应 R4-7）。

R4-7 原文批评：loss 权重按 val 选、early stopping 按 val ADE、拓扑按 val ADE 选，
最终 Table II 又在**同一个 val** 上报结果。本脚本把 headline 报数与模型选择彻底隔离。

**使用纪律（写死在流程里，不靠自觉）**：
  - 输入必须是**已冻结**的 checkpoint；本脚本不训练、不做 early stopping、不选超参、不选拓扑；
  - test 只在方法冻结后正式运行一次；
  - 报数一律用 **saved-checkpoint** 口径，**不得**使用训练曲线的独立 minima
    （阶段9 §4.3 已订正：曲线 minima 不保证来自保存的 checkpoint，仅作训练诊断）。

口径（2026-08-18 补，见 `progress_20260818.md` 阶段206）：
  - **batch_size 冻结为 8** —— ego BatchNorm `track_running_stats=False`，eval 模式仍用当前 batch 统计量，
    读数随 batch 组成变化；8 是新档训练时 val 所用值，选它才与训练侧 val 同口径（回归门：
    ego_only/seed42 在 val 上重放得 ADE 3.044 / FDE 7.379，与训练日志逐位一致）。
  - **必须传 `follow_batch=["gnn_x"]`**（脚本已自动按 sidecar 决定）：否则逐样本 GNN 池化退化为整批池化。
  - **邻接配置默认从 ckpt sidecar 的 `dataset_config` 取**（2026-08-18 19:5x 补的第三处缺陷）：
    `get_adjacency_config()` 对未注册的名字**静默回退 DefaultBuilder**，所以传格子名 `f_nearby`
    而不是 builder 名 `nearby_only` 时不报错，只是把 nearby_only 训出来的 ckpt 放到完全图上评。
    现在改成：缺省读 sidecar；显式传入与 sidecar 不符 ⇒ 报错；非 default 的名字拿回 DefaultBuilder ⇒ 报错。

产出：
  test_metrics.json  汇总 ADE/FDE + 完整复现元数据（ckpt/seed/route_mode/perturbation/split/代码版本）
  test_per_scene.csv  逐样本 ADE/FDE

运行：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/eval_test_set.py \
      --checkpoint /abs/.../models/best_model_default.pth \
      --test_root  /abs/.../<data>_split3/test \
      --output_dir /abs/.../experiments/<run>/test_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
_MODELS = _DEVKIT / "models"
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_MODELS), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
from transformer_predictor import TransformerPredictor  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_test_set")


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_GNN_ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    """逐样本 ADE/FDE。模型输出为 60×2 局部逐帧增量，需 cumsum 成位置后与 GT 位置比。"""
    model.eval()
    per_scene = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).reshape(-1, config.NUM_FUTURE_POSES, 2)      # [B,60,2] 增量
        gt = batch.y.reshape(-1, config.NUM_FUTURE_POSES, 2)             # [B,60,2] 增量
        pred_pos = torch.cumsum(pred, dim=1)
        gt_pos = torch.cumsum(gt, dim=1)
        d = torch.norm(pred_pos - gt_pos, dim=-1)                        # [B,60]
        ade = d.mean(dim=1).cpu().numpy()
        fde = d[:, -1].cpu().numpy()
        for a, f in zip(ade, fde):
            per_scene.append((float(a), float(f)))
    arr = np.asarray(per_scene, dtype=np.float64)
    return {
        "n_samples": int(arr.shape[0]),
        "ade_mean": float(arr[:, 0].mean()), "ade_std": float(arr[:, 0].std(ddof=1)),
        "fde_mean": float(arr[:, 1].mean()), "fde_std": float(arr[:, 1].std(ddof=1)),
        "ade_median": float(np.median(arr[:, 0])), "fde_median": float(np.median(arr[:, 1])),
        "_per_scene": per_scene,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="独立 test set 评测（R4-7）；只读 checkpoint，不训练")
    p.add_argument("--checkpoint", required=True, type=str, help="冻结的 bare state_dict（绝对路径）")
    p.add_argument("--test_root", required=True, type=str, help="split3 的 test leaf root（绝对路径）")
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--dataset_config", default=None,
                   help="邻接构图配置名；缺省从 ckpt 的 train_meta.json 取（推荐），"
                        "显式传入且与 sidecar 不符时**直接报错**，不静默采用其一")
    p.add_argument("--gnn_pooling_fix", action="store_true",
                   help="无 train_meta.json sidecar 的老 ckpt 才需要显式传；有 sidecar 时以 sidecar 为准")
    p.add_argument("--batch_size", type=int, default=8,
                   help="🔴 默认 8 且不要改：ego BatchNorm 是 track_running_stats=False，eval 模式下仍用"
                        "**当前 batch 的统计量** ⇒ 读数随 batch 组成变化（实测同 ckpt 同 val：b=8 → ADE 3.044、"
                        "b=32 → 2.787，差 8.4%）。8 = 新档训练时 val 用的 batch，选它才能与训练侧 val 读数同口径")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split_manifest", default=None, help="split3_manifest.json；默认取 test_root 上级")
    p.add_argument("--tag", default="", help="实验标签，如 B/R/P/RP + seed")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    ckpt = Path(a.checkpoint).resolve()
    test_root = Path(a.test_root).resolve()
    out_dir = Path(a.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        logger.error("checkpoint 不存在: %s", ckpt)
        sys.exit(1)

    # 训练侧元数据（route_mode / perturbation），供报表与一致性追溯
    meta_path = ckpt.parent / "train_meta.json"
    train_meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            train_meta = json.load(f)
    else:
        logger.warning("checkpoint 无 train_meta.json sidecar：%s（route_mode 按 'none' 记录）", meta_path)

    if a.split_manifest:
        sm_path = Path(a.split_manifest)
    else:
        # 旧档三分 split 叫 split3_manifest.json；新档（fullret L1）叫 test_split_manifest.json
        cands = [test_root.parent / "test_split_manifest.json", test_root.parent / "split3_manifest.json"]
        sm_path = next((c for c in cands if c.exists()), cands[-1])
    split_meta = {}
    if sm_path.exists():
        with open(sm_path, encoding="utf-8") as f:
            sm = json.load(f)
        split_meta = {k: v for k, v in sm.items() if k != "logs"}

    device = torch.device(a.device)
    if a.batch_size != 8:
        logger.warning("🔴 batch_size=%d ≠ 8：ego BatchNorm(track_running_stats=False) 使读数随 batch 组成变化，"
                       "本次结果与训练侧 val 口径**不可比**（实测 b=8 → 3.044 vs b=32 → 2.787）。"
                       "只有确知自己在做敏感性分析时才这么跑。", a.batch_size)
    # 🔴🔴 邻接配置必须与被评 ckpt **训练时**所用的一致，且必须是注册过的 builder 名。
    #    2026-08-18 查出的坑：`get_adjacency_config()` 的 else 分支**静默回退 DefaultBuilder**，
    #    所以传格子名（`f_nearby`）而不是 builder 名（`nearby_only`）不会报错，只会把 nearby_only
    #    训出来的 ckpt 放到完全图上评 —— 读数被污染而通路一路 rc=0。此处三重把关：
    #      ① 缺省从 sidecar `dataset_config` 取（单一真相源）；
    #      ② 显式传入与 sidecar 不符 ⇒ 报错退出（不猜哪个对）；
    #      ③ 非 "default" 的名字却拿回 DefaultBuilder ⇒ 说明名字没注册，报错退出。
    cfg_meta = train_meta.get("dataset_config")
    dataset_config = a.dataset_config or cfg_meta
    if dataset_config is None:
        logger.error("ckpt 无 train_meta.dataset_config 且未显式传 --dataset_config，无法确定邻接配置")
        sys.exit(4)
    if a.dataset_config and cfg_meta and a.dataset_config != cfg_meta:
        logger.error("--dataset_config=%r 与 ckpt 训练时的 %r 不一致 ⇒ 拒绝评测（拓扑错配会污染读数）",
                     a.dataset_config, cfg_meta)
        sys.exit(5)
    builder = get_adjacency_config(dataset_config)
    if dataset_config != "default" and type(builder).__name__ == "DefaultBuilder":
        logger.error("邻接配置 %r 未在 get_adjacency_config 注册（静默回退到 DefaultBuilder）⇒ 拒绝评测",
                     dataset_config)
        sys.exit(6)
    logger.info("邻接配置 = %r（来源：%s），builder=%s", dataset_config,
                "CLI" if a.dataset_config else "ckpt sidecar", type(builder).__name__)

    # 🔴 建图聚合必须与被评 ckpt 训练时所见一致（旋钮 A 判据 20260810 同款）：
    #    按 sidecar 设全局，缺键（老 ckpt）回退 mean = 旧行为逐位不变。
    #    必须在 GraphDataset 之前设 —— 池化发生在建图期，设晚了会静默用错聚合。
    from src.shared.utils.bn_calibration import load_model_for_inference, read_architecture
    from src.shared.utils.dataset import set_subgraph_agg
    sidecar = read_architecture(ckpt) or {}
    subgraph_agg = sidecar.get("subgraph_agg") or "mean"
    set_subgraph_agg(subgraph_agg)

    ds = GraphDataset(str(test_root), transform=None)   # ★ test 绝不增广
    # 🔴🔴 D0#7 同族缺陷（2026-08-18 查出）：`gnn_pooling_fix` 的**逐样本池化**依赖
    #    DataLoader 的 `follow_batch=["gnn_x"]` 产出的 `gnn_x_batch`（MagnetImitationPooled 别名为
    #    `gnn_batch`）。本脚本原来不传 follow_batch ⇒ 该键缺失 ⇒ 池化退化成**整个 batch 一起池化**，
    #    预测被污染且随 batch_size 变化（实测 ego_only/seed42 val：b=32 → ADE 3.423、b=8 → 3.651，
    #    而训练侧同 ckpt 同 val 的真值是 3.044）。test 通路此前从未跑过（全仓 0 个 test_metrics.json），
    #    所以这个缺陷一直潜伏，会直接污染 headline test 数字。
    follow = ["gnn_x"] if sidecar.get("gnn_pooling_fix_effective",
                                      sidecar.get("gnn_pooling_fix", a.gnn_pooling_fix)) else None
    if sidecar.get("temporal_graph"):
        follow = (follow or []) + ["tg_x"]   # R1 需要 tg_x_batch 做 (样本, 帧) 双重分组
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, follow_batch=follow)
    logger.info("test 集: %d 样本 (%s)，subgraph_agg=%s", len(ds), test_root, subgraph_agg)

    # 🔴 不再裸建 TransformerPredictor —— 新档 ckpt 是 MagnetImitationPooled(gnn_pooling_fix=True)，
    #    裸建会 load_state_dict 失配或静默建错模型。统一走 load_model_for_inference（sidecar 决定架构）。
    model = load_model_for_inference(ckpt, gnn_pooling_fix=a.gnn_pooling_fix, device=device).to(device)
    logger.info("已加载冻结 checkpoint: %s（class=%s, gnn_pooling_fix=%s）",
                ckpt, type(model).__name__, sidecar.get("gnn_pooling_fix_effective", a.gnn_pooling_fix))

    t0 = time.time()
    res = evaluate(model, loader, device)
    per_scene = res.pop("_per_scene")

    summary = {
        "tag": a.tag,
        "protocol": "saved-checkpoint 口径；test 仅用于最终 headline，不参与任何选择（R4-7）",
        **res,
        "checkpoint": str(ckpt),
        "test_root": str(test_root),
        "dataset_config": dataset_config,
        "dataset_config_source": "CLI" if a.dataset_config else "ckpt_train_meta",
        "model_class": type(model).__name__,
        "architecture_resolved": sidecar or {"note": "无 sidecar，按 --gnn_pooling_fix 老两态建模"},
        "subgraph_agg": subgraph_agg,
        "follow_batch": follow,
        "batch_size": a.batch_size,
        "route_mode": train_meta.get("route_mode", "none"),
        "perturbation": train_meta.get("perturbation"),
        "train_seed": train_meta.get("seed"),
        "split_manifest": split_meta,
        "git_rev": _git_rev(),
        "eval_seconds": round(time.time() - t0, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "test_per_scene.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ade_m", "fde_m"])
        for i, (ade, fde) in enumerate(per_scene):
            w.writerow([i, f"{ade:.6f}", f"{fde:.6f}"])

    print()
    print("=" * 64)
    print(f"独立 TEST 评测{(' [' + a.tag + ']') if a.tag else ''}")
    print("=" * 64)
    print(f"样本数    : {res['n_samples']}")
    print(f"ADE       : {res['ade_mean']:.3f} ± {res['ade_std']:.3f} m  (median {res['ade_median']:.3f})")
    print(f"FDE       : {res['fde_mean']:.3f} ± {res['fde_std']:.3f} m  (median {res['fde_median']:.3f})")
    print(f"route_mode: {summary['route_mode']}   perturbation: "
          f"{(summary['perturbation'] or {}).get('enabled')}")
    print(f"产物      : {out_dir}")


if __name__ == "__main__":
    main()
