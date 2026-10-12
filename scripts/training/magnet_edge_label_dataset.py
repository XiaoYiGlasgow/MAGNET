"""S4 —— 逐边 `U` / `U_sd_teacher` / `stratum` 缓存 + dataset 挂载。

🔴 **为什么不扩展 `build_oracle_gate_cache.py` / `magnet_oracle_gate_dataset.py`**：
E1 链的阶段3（`eval_joint_test.py`）还要 import 那两个文件。在链跑完之前动它们，
改坏了就是整条链白跑。本文件只**复用**它们的 `_match_edges`（向量化 join），不改它们一个字。

## 与 oracle 门缓存的区别

`oracle_gate` 存的是**逐目标 min-max 归一后的 `w`**（E1 的注入量）；本缓存存**原始 `U`**
与**跨 teacher 离散度 `U_sd`**。理由：S4 的排序损失需要 `|U_a − U_b|` 的**绝对尺度**
才能跟 teacher 噪声比大小，而 min-max 把每个目标的尺度都抹平了、反推不回来。

## 存储：**稠密逐边数组，未打标填 `NaN`**

三个字段都是 `[E]`、与 `gnn_edge_index` 逐列并列。未打标的边填 `NaN`（`stratum` 填 -1）。

🔴 **为什么用 NaN 不用 0**：S3 的 meta 逐字写着「未出现在产物里的边**不是零影响边** ——
S4 必须靠 join 失败来 mask 掉，**不得填 0**」。填 0 会被损失函数当成「这条边没用」的
真实标签；`NaN` 则在任何算术里都自曝，mask 写成 `~isnan` 也读得懂。

🔴 **为什么不存稀疏列号**：列号在 PyG 批处理时需要加该样本的边偏移，那要自定义
`Data.__inc__`（不是 `Dataset.__inc__` —— 批处理调的是前者）。稠密数组与 `gnn_edge_attr`
同型，走的是 `oracle_gate` 已经验证过的同一条路径，不需要任何自定义批处理行为。
代价是体积：train 约 1.4 GB、val 约 160 MB。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_DEVKIT), str(_DEVKIT / "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_oracle_gate_cache import _match_edges  # noqa: E402  向量化 join，逐字复用
from magnet_agent_aux_dataset import AgentAuxGraphDataset  # noqa: E402

sys.path.insert(0, str(_GNN_ROOT / "scripts" / "analysis"))
from joint_usage_probe import resolve_tokens  # noqa: E402  下标↔token，自带文件数守卫

logger = logging.getLogger(__name__)

CACHE_NAME = "edge_label_s4_{cfg}.pt"
FINGERPRINT_RTOL = 1e-4

#: `dst_stratum` 的取值（S3 产物逐字），G3 必须**分层**报 —— ego 层信号最弱
#: （目标内 `U` 极差 ÷ teacher 分歧 = 3.62，13.9% 的目标不及噪声）。
STRATA = ("ego", "moving_agent", "static_or_truncated")


class EdgeLabelAlignmentError(RuntimeError):
    """标签缓存与 dataset 样本对不上 —— 必须当场炸，错位不会报错只会训出无法解释的产物。"""


def label_cache_path(root, adjacency_config: str = "default") -> Path:
    return (Path(root) / "processed" / f"config_{adjacency_config}"
            / CACHE_NAME.format(cfg=adjacency_config))


def load_labels_full(label_root: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict]:
    """读 S3 合并产物，返回 `{token: {src, dst, U, U_sd, dst_stratum}}` 与 meta。"""
    meta = json.loads((Path(label_root) / "meta.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(Path(label_root).glob("labels_*.npz")):
        with np.load(f, allow_pickle=True) as z:
            ptr = np.asarray(z["scene_ptr"])
            toks = [str(t) for t in z["token"]]
            cols = {k: np.asarray(z[k]) for k in ("src", "dst", "U", "U_sd_teacher", "dst_stratum")}
            for i, tk in enumerate(toks):
                sl = slice(int(ptr[i]), int(ptr[i + 1]))
                out[tk] = {"src": cols["src"][sl].copy(), "dst": cols["dst"][sl].copy(),
                           "U": cols["U"][sl].astype(np.float32).copy(),
                           "U_sd": cols["U_sd_teacher"][sl].astype(np.float32).copy(),
                           "stratum": cols["dst_stratum"][sl].copy()}
    if not out:
        raise EdgeLabelAlignmentError(f"{label_root} 下没有可读的 labels_*.npz")
    return out, meta


def _stratum_code(arr: np.ndarray) -> np.ndarray:
    """S3 的 `dst_stratum` **已经是 uint8 编码**（0=ego / 1=moving_agent / 2=static_or_truncated）。

    实测 val 占比 4.2% / 8.6% / 87.2%，与阶段312 的 train 全量分层（3.7% / 8.7% / 87.6%）
    同序同量级 ⇒ 编码顺序与 :data:`STRATA` 一致。

    🔴 第一版按字符串查表写，把 `0/1/2` 全映成 -1、分层读数整段消失，而缓存照样写盘 ——
    第二次栽在「没先看字段的实际编码」上。越界值在这里当场抛，不进缓存。
    """
    a = np.asarray(arr).astype(np.int16)
    bad = (a < 0) | (a >= len(STRATA))
    if bad.any():
        raise EdgeLabelAlignmentError(
            f"dst_stratum 出现越界取值 {np.unique(a[bad])[:5].tolist()}（合法 0..{len(STRATA)-1}）")
    return a.astype(np.int8)


def build_cache(root: str, label_root: str, adjacency_config: str = "default") -> Path:
    """把 S3 标签烘成逐样本的稀疏三元组缓存。返回缓存路径。"""
    ds = AgentAuxGraphDataset(str(root), transform=None, adjacency_config=adjacency_config)
    # 🔴 `token` **不是** `Data` 对象上的属性 —— 必须按下标从 root 解析（与 E1 同一条路径）。
    #    第一版写成 `getattr(d, "token", "")` ⇒ 全部落空、打标边 0.00%，而脚本 rc=0。
    toks = resolve_tokens(str(root), ds.len())
    labels, meta = load_labels_full(Path(label_root))
    entries: List[Dict] = []
    n_hit = n_edge = n_scene_hit = 0
    for i in range(ds.len()):
        d = ds.get(i)
        tok = toks[i]
        ei = np.asarray(d.gnn_edge_index)
        n_edge += ei.shape[1]
        E = int(ei.shape[1])
        U = np.full(E, np.nan, dtype=np.float32)
        Usd = np.full(E, np.nan, dtype=np.float32)
        strat = np.full(E, -1, dtype=np.int8)
        lab = labels.get(tok)
        if lab is not None:
            n_scene_hit += 1
            cols, rows = _match_edges(ei, lab["src"], lab["dst"])
            n_hit += cols.size
            U[cols] = lab["U"][rows]
            Usd[cols] = lab["U_sd"][rows]
            strat[cols] = _stratum_code(lab["stratum"][rows])
        entries.append({
            "U": torch.from_numpy(U), "U_sd": torch.from_numpy(Usd),
            "stratum": torch.from_numpy(strat),
            "n_edges": E, "n_nodes": int(d.gnn_x.shape[0]),
            "gnn_x_sum": float(d.gnn_x.sum())})
        if (i + 1) % 2000 == 0:
            logger.info("  烘缓存 %d/%d", i + 1, ds.len())

    stats = {"n_samples": len(entries), "n_scenes_with_labels": n_scene_hit,
             "n_edges_total": n_edge, "n_edges_labelled": int(n_hit),
             "frac_edges_labelled": float(n_hit) / max(n_edge, 1)}
    # 🔴 空产物守卫（项目规则 10：rc=0 ≠ 跑了）—— 第一版 token 解析写错时脚本正常退出、
    #    缓存正常写盘、打标边 0.00%，只有读 stats 才发现。这道闸就是为它加的。
    if n_scene_hit == 0 or n_hit == 0:
        raise EdgeLabelAlignmentError(
            f"join 全败：打标场景 {n_scene_hit}、打标边 {n_hit} —— 缓存拒绝写盘。"
            f"多半是 token 解析或 S3 目录与 root 不是同一 split")
    cp = label_cache_path(root, adjacency_config)
    cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"entries": entries, "stats": stats,
                "meta": {"label_root": str(label_root), "adjacency_config": adjacency_config,
                         "s3_version": meta.get("version"), "s3_script_sha256": meta.get("script_sha256"),
                         "teacher_sha256": meta.get("teacher_sha256"), "strata": list(STRATA)}}, cp)
    logger.info("✅ 缓存写入 %s｜%d 样本｜打标场景 %d｜打标边占比 %.2f%%",
                cp, len(entries), n_scene_hit, 100 * stats["frac_edges_labelled"])
    return cp


class EdgeLabelGraphDataset(AgentAuxGraphDataset):
    """`AgentAuxGraphDataset` + 三个稠密逐边标签字段（`s4_U` / `s4_Usd` / `s4_stratum`）。"""

    def __init__(self, root, transform=None, pre_transform=None,
                 adjacency_config: str = "default") -> None:
        super().__init__(root, transform, pre_transform, adjacency_config=adjacency_config)
        cp = label_cache_path(root, adjacency_config)
        if not cp.exists():
            raise FileNotFoundError(
                f"缺 S4 标签缓存 {cp} —— 先跑 `magnet_edge_label_dataset.py --root {root} "
                f"--labels <S3 目录> --adjacency_config {adjacency_config}`")
        payload = torch.load(cp)
        self._lab: List[Dict] = payload["entries"]
        self._lab_stats = payload.get("stats", {})
        self._lab_meta = payload.get("meta", {})
        if len(self._lab) != super().len():
            raise EdgeLabelAlignmentError(
                f"标签缓存条数 {len(self._lab)} ≠ dataset 样本数 {super().len()} —— 不是同一批数据")
        s = self._lab_stats
        logger.info("EdgeLabelGraphDataset: %d 样本｜打标场景 %d｜打标边占比 %.2f%%",
                    len(self._lab), s.get("n_scenes_with_labels", 0),
                    100 * float(s.get("frac_edges_labelled", 0.0)))

    def get(self, idx):                                   # noqa: D102  见模块 docstring
        data = super().get(idx)
        e = self._lab[idx]
        if int(e["n_edges"]) != int(data.gnn_edge_index.shape[1]) or \
           int(e["n_nodes"]) != int(data.gnn_x.shape[0]):
            raise EdgeLabelAlignmentError(
                f"样本 {idx} 指纹不符：缓存 E={e['n_edges']} N={e['n_nodes']}，"
                f"dataset E={data.gnn_edge_index.shape[1]} N={data.gnn_x.shape[0]}")
        ref = float(data.gnn_x.sum())
        if abs(ref - float(e["gnn_x_sum"])) > FINGERPRINT_RTOL * max(abs(ref), 1.0):
            raise EdgeLabelAlignmentError(f"样本 {idx} 的 gnn_x_sum 指纹不符 —— 不是同一份特征")
        data.s4_U = e["U"]                  # [E]，未打标 = NaN
        data.s4_Usd = e["U_sd"]             # [E]，跨 teacher 离散度
        data.s4_stratum = e["stratum"]      # [E] int8，见 STRATA；-1 = 未打标
        return data


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description="S4：把 S3 边效用标签烘成逐边稀疏缓存")
    p.add_argument("--root", required=True, help="split leaf root（含 aux 缓存）")
    p.add_argument("--labels", required=True, help="S3 合并产物目录（含 labels_*.npz + meta.json）")
    p.add_argument("--adjacency_config", default="default")
    a = p.parse_args()
    build_cache(a.root, a.labels, a.adjacency_config)


if __name__ == "__main__":
    main()
