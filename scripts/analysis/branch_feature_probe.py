"""分支特征探针 —— 一次前向同时产出实验 A（判别力）与实验 B（分层 ADE）的原料。

背景（阶段74）：闸① 把入图 agent 从 2.75 提到 10.17，但新增的 7.4 辆**全是停放车**。
提出四条候选机制解释「加宽交互通路为何不改善甚至变差」，其中两条现在就能测：

* **机制 1（判别力稀释）**：`global_mean_pool` 的输出方差随节点数下降（`Var[pool] ~ 1/N`）
  ⇒ 节点 43→51 后，`gnn_feat` 在**场景之间**的差异应当变小。
  判别设计 = 同一份权重喂两种图（`base_on_base` vs `base_on_gate1`），
  把**纯池化效应**与**重训后的适应**分开。
* **机制 3（容量竞争）**：`Linear(384,128)` 总容量固定，加宽交互 = 削弱 ego 通路；
  若成立，改动的收益应当只出现在**交互密度高**的场景 ⇒ 需要逐场景 ADE + 逐场景 agent 数。

本脚本只**采集**，不下结论；统计与判读在 `branch_feature_readout.py`。

⚠️ 模型加载必须走 `load_model_for_inference`（与 `run_interaction_audit.py:310` 同一入口）——
直接 `TransformerPredictor()` + `load_state_dict` 会把 calibrated ckpt 的 BN running stats
当成 unexpected keys 丢掉，实际加载的是 raw 权重。

用法::

    python3 scripts/analysis/branch_feature_probe.py \
        --ckpt <abs.pth> --raw_dir <abs val dir> --out <abs.npz> --threads 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.shared.utils.dataset import build_graph_data_from_feature_df  # noqa: E402

logger = logging.getLogger("branch_probe")

NUM_FUTURE_POSES = 60


def _load_feature_df(pkl_path: Path) -> Optional[pd.DataFrame]:
    """读单个 ``features_<token>.pkl``；读不出来返回 None。"""
    try:
        df = pd.read_pickle(pkl_path)
    except (OSError, EOFError, ValueError) as exc:
        logger.warning("读取失败 %s: %s", pkl_path.name, exc)
        return None
    return df if isinstance(df, pd.DataFrame) and len(df) > 0 else None


def _node_counts(df: pd.DataFrame) -> tuple:
    """从 SUBGRAPH_ID_TO_TYPE 数各类节点，口径与干预审计一致。"""
    if "SUBGRAPH_ID_TO_TYPE" not in df.columns:
        return (-1, -1)
    id_to_type = df["SUBGRAPH_ID_TO_TYPE"].values[0]
    types = list(id_to_type.values()) if isinstance(id_to_type, dict) else list(id_to_type)
    n_traj = sum(1 for t in types if str(t).lower().startswith(("traj", "agent")))
    n_lane = sum(1 for t in types if str(t).lower().startswith("lane"))
    return (n_traj, n_lane)


def _ade_fde(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    """与 ``train_magnet_imitation.py`` 同口径：增量先 cumsum 再算位移。"""
    p = torch.cumsum(pred.view(NUM_FUTURE_POSES, 2), dim=0)
    t = torch.cumsum(target.view(NUM_FUTURE_POSES, 2), dim=0)
    disp = torch.norm(p - t, dim=1)
    return float(disp.mean()), float(disp[-1])


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--raw_dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--adjacency_config", default="default")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 个 token（排序后取，可复现）")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")

    from src.shared.utils.bn_calibration import load_model_for_inference
    model = load_model_for_inference(a.ckpt, gnn_pooling_fix=False, device=device)

    pkls = sorted(a.raw_dir.glob("features_*.pkl"))
    if a.limit:
        pkls = pkls[: a.limit]
    logger.info("待处理 token=%d  ckpt=%s", len(pkls), a.ckpt.name)

    tokens, ego_f, ctx_f, gnn_f = [], [], [], []
    n_traj_l, n_lane_l, n_node_l, ade_l, fde_l = [], [], [], [], []
    skipped = 0
    t0 = time.time()

    from torch_geometric.data import Batch

    for i, pkl in enumerate(pkls):
        token = pkl.stem.replace("features_", "")
        df = _load_feature_df(pkl)
        if df is None:
            skipped += 1
            continue
        data, reason = build_graph_data_from_feature_df(df, a.adjacency_config)
        if data is None:
            skipped += 1
            continue
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            feats = model.get_encoder_features(batch)      # [1, 384]
            pred = model(batch)                            # [1, 120]
        f = feats[0].cpu().numpy()
        ego_f.append(f[:128]); ctx_f.append(f[128:256]); gnn_f.append(f[256:])

        ade, fde = (np.nan, np.nan)
        if hasattr(batch, "y") and batch.y is not None and batch.y.numel() >= NUM_FUTURE_POSES * 2:
            ade, fde = _ade_fde(pred[0].cpu(), batch.y.reshape(-1).cpu()[: NUM_FUTURE_POSES * 2])

        n_traj, n_lane = _node_counts(df)
        tokens.append(token); n_traj_l.append(n_traj); n_lane_l.append(n_lane)
        n_node_l.append(int(data.gnn_x.shape[0])); ade_l.append(ade); fde_l.append(fde)

        if (i + 1) % 500 == 0:
            logger.info("  %d/%d  用时 %.1f min", i + 1, len(pkls), (time.time() - t0) / 60)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        tokens=np.array(tokens), ego=np.array(ego_f, dtype=np.float32),
        ctx=np.array(ctx_f, dtype=np.float32), gnn=np.array(gnn_f, dtype=np.float32),
        n_traj=np.array(n_traj_l), n_lane=np.array(n_lane_l), n_node=np.array(n_node_l),
        ade=np.array(ade_l, dtype=np.float32), fde=np.array(fde_l, dtype=np.float32),
        ckpt=str(a.ckpt), raw_dir=str(a.raw_dir),
    )
    logger.info("完成 %d 条（跳过 %d），用时 %.1f min -> %s",
                len(tokens), skipped, (time.time() - t0) / 60, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
