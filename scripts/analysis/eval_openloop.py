"""按指定 inference 口径评测一个 checkpoint 的开环 ADE/FDE，产出可被 `agg_2x2.py` 消费的 JSON。

为什么需要它（阶段⑳ §2.4）
--------------------------
`checkpoint_best.pt` 里存的 `val_ade`/`val_fde` 是**训练时 batch=32、BN 未校准**的 diagnostic 数字
（D0#13）。CP1 的方向、paired contrasts 与噪声判断只能用 **calibrated** 口径重算的 dev 汇总，
所以每个 run 都要用同一个 inference function 重新评测一遍，结果写成 JSON 供汇总脚本读取。

口径说明
--------
* `--batch_size 1` = planner 真实部署口径，且此时 GNN 池化两种语义**逐字节相同**
  （`GNNEncoder` 单图下 `mean(0)` ≡ `global_mean_pool`），故 fix=False 的 G1′ checkpoint
  也能给出与 Q13 无关的 calibrated 数字。
* calibrated checkpoint 与原始 checkpoint 都能读；是否 calibrated 由权重里有没有
  `ego_history_encoder.1.running_mean` 自动判定并写进 JSON。

用法::

    python3 scripts/analysis/eval_openloop.py \
        --ckpt experiments/calibration/calibrated/B_seed42/B_seed42_calibrated.pth \
        --eval_root data/processed_features_magnet_e1_none_v3_split3/val \
        --batch_size 1 --run_name B_seed42 \
        --out experiments/calibration/openloop/B_seed42_calibrated_b1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

_GNN_ROOT = Path(__file__).resolve().parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_GNN_ROOT), str(_DEVKIT / "models"), str(_DEVKIT),
           str(_GNN_ROOT / "scripts" / "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.shared.utils.bn_calibration import BN_MODULE_PATH, file_sha256, load_model_for_inference  # noqa: E402
from src.shared.utils.dataset import GraphDataset, get_adjacency_config  # noqa: E402
from train_magnet_imitation import _ade_fde, move_batch_to_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="开环 ADE/FDE 评测（指定 inference 口径）")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--eval_root", required=True, type=Path)
    p.add_argument("--run_name", required=True, help="写进 JSON 的 run 标识，如 B_seed42")
    p.add_argument("--batch_size", type=int, default=1, help="1 = planner 部署口径（默认）")
    p.add_argument("--n", type=int, default=0, help="0 = 全部样本")
    p.add_argument("--gnn_pooling_fix", action="store_true")
    p.add_argument("--dataset_config", default="default")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, data_list: List[Any], batch_size: int, device: torch.device,
             follow_batch: List[str] | None) -> Dict[str, float]:
    from torch_geometric.loader import DataLoader  # noqa: WPS433

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, follow_batch=follow_batch)
    preds, ys = [], []
    for data in loader:
        data = move_batch_to_device(data, device)
        bs = int(data.num_graphs)
        preds.append(model(data)[:bs].cpu())
        ys.append(data.y.reshape(bs, -1).cpu())
    ade, fde = _ade_fde(torch.cat(preds), torch.cat(ys))
    return {"ade": float(ade.mean()), "fde": float(fde.mean()),
            "ade_sd_per_sample": float(ade.std(unbiased=True)),
            "fde_sd_per_sample": float(fde.std(unbiased=True))}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    get_adjacency_config(args.dataset_config)
    ds = GraphDataset(str(args.eval_root), transform=None)
    n = len(ds) if args.n <= 0 else min(args.n, len(ds))
    data_list = [ds[i] for i in range(n)]
    logger.info("eval set: %s  n=%d/%d", args.eval_root, n, len(ds))

    blob = torch.load(args.ckpt, map_location="cpu")
    state = blob["model_state_dict"] if isinstance(blob, dict) and "model_state_dict" in blob else blob
    is_calibrated = f"{BN_MODULE_PATH}.running_mean" in state

    model = load_model_for_inference(args.ckpt, args.gnn_pooling_fix, device)
    follow = ["gnn_x"] if args.gnn_pooling_fix else None
    t0 = time.time()
    metrics = evaluate(model, data_list, args.batch_size, device, follow)
    elapsed = time.time() - t0

    doc: Dict[str, Any] = {
        "run_name": args.run_name,
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256_file": file_sha256(args.ckpt),
        "is_calibrated": is_calibrated,
        "gnn_pooling_fix": bool(args.gnn_pooling_fix),
        "inference_contract": ("calibrated" if is_calibrated else "uncalibrated")
                              + f"@batch{args.batch_size}",
        "eval_root": str(args.eval_root.resolve()),
        "n_samples": n,
        "batch_size": args.batch_size,
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    logger.info("%s  %s  ADE=%.4f FDE=%.4f  (%.0fs)  -> %s",
                args.run_name, doc["inference_contract"], metrics["ade"], metrics["fde"],
                elapsed, args.out)


if __name__ == "__main__":
    main()
