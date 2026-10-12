"""E1 —— 把 `build_oracle_gate_cache.py` 的逐边门挂回 `AgentAuxGraphDataset`（不改上游）。

与 `magnet_agent_aux_dataset` 同一条纪律：**对齐不靠约定，靠断言**。
每条缓存带 `n_nodes` / `n_edges` / `gnn_x_sum` 三个指纹，`get(idx)` 时逐样本核对；
不一致立刻抛错。理由：错位不会报错，只会训出一份**没人能解释的产物**。

⚠️ PyG 批处理行为：字段名 `oracle_gate` 里不含 `index`
⇒ 走 `super().__inc__` 返回 0（不加偏移）、`cat_dim=0`（沿 0 维拼接）。
`gnn_edge_index` 是 `[2, E]` 沿 dim=1 拼接，两者的**逐样本顺序相同**
⇒ 批处理后 `oracle_gate[k]` 仍对应 `gnn_edge_index[:, k]`（与 `gnn_edge_attr` 同型，后者已在用）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT), str(_DEVKIT / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_oracle_gate_cache import cache_path  # noqa: E402
from magnet_agent_aux_dataset import AgentAuxGraphDataset  # noqa: E402

logger = logging.getLogger(__name__)

#: 与 aux 缓存同一容差 —— `gnn_x_sum` 是 float32 求和，跨进程可能有末位差异。
FINGERPRINT_RTOL = 1e-4


class OracleGateAlignmentError(RuntimeError):
    """门缓存与 dataset 样本对不上 —— 必须当场炸。"""


class OracleGateGraphDataset(AgentAuxGraphDataset):
    """`AgentAuxGraphDataset` + 一个逐边字段 `oracle_gate`（`[E]`，未打标边恒 1.0）。"""

    def __init__(self, root, transform=None, pre_transform=None,
                 adjacency_config: str = "default", variant: str = "oracle") -> None:
        super().__init__(root, transform, pre_transform, adjacency_config=adjacency_config)
        cp = cache_path(root, variant, adjacency_config)
        if not cp.exists():
            raise FileNotFoundError(
                f"缺 oracle 门缓存 {cp} —— 先跑 "
                f"`build_oracle_gate_cache.py --root {root} --labels <S3 目录> "
                f"--variant {variant} --adjacency_config {adjacency_config}`")
        payload = torch.load(cp)
        self._gate: List[Dict] = payload["entries"]
        self._gate_stats = payload.get("stats", {})
        self._gate_meta = payload.get("meta", {})
        n_self = super().len()
        if len(self._gate) != n_self:
            raise OracleGateAlignmentError(
                f"门缓存条数 {len(self._gate)} ≠ dataset 样本数 {n_self} —— 不是同一批数据")
        s = self._gate_stats
        logger.info("OracleGateGraphDataset: %d 样本｜variant=%s｜打标边占比 %.2f%%｜"
                    "被打标边 w mean %.4f｜全边 w mean %.4f",
                    n_self, variant, 100 * float(s.get("frac_edges_labelled", 0.0)),
                    float(s.get("w_mean_labelled", 1.0)), float(s.get("w_mean_all", 1.0)))

    def get(self, idx):                                   # noqa: D102  见类 docstring
        data = super().get(idx)
        e = self._gate[idx]
        n_nodes = int(data.gnn_x.shape[0]) if getattr(data, "gnn_x", None) is not None else 0
        n_edges = int(data.gnn_edge_index.shape[1])
        if n_nodes != int(e["n_nodes"]) or n_edges != int(e["n_edges"]):
            raise OracleGateAlignmentError(
                f"样本 {idx} 规模对不上：dataset ({n_nodes} 节点, {n_edges} 边) vs "
                f"缓存 ({e['n_nodes']}, {e['n_edges']}) ⇒ 缓存错位")
        if n_nodes:
            fp, ref = float(data.gnn_x.sum()), float(e["gnn_x_sum"])
            if abs(fp - ref) > FINGERPRINT_RTOL * max(abs(ref), 1.0):
                raise OracleGateAlignmentError(
                    f"样本 {idx} 指纹对不上：{fp:.6f} vs {ref:.6f} ⇒ 缓存错位")
        data.oracle_gate = e["gate"]
        return data


__all__ = ["OracleGateAlignmentError", "OracleGateGraphDataset"]
