"""D0#7 修复：让 imitation 主线的 GNN 分支**逐样本**输出，而不是整个 batch 共享一个向量。

缺陷（阶段14 §1.4，inspect 级复核）
-----------------------------------
`train_magnet_imitation.py:212` 建 DataLoader 时没有 `follow_batch=['gnn_x']`，于是 PyG 不生成
`gnn_x_batch`；`GraphData.__inc__/__cat_dim__` 也不产 `gnn_batch`。父类因此走到：

    transformer_predictor.py:263   gnn_batch = getattr(data, 'gnn_batch', None)   -> None
    transformer_predictor.py:52-55 batch is None -> x.mean(dim=0, keepdim=True)   -> [1,128]
    transformer_predictor.py:270-1 gnn_out.size(0)==1 -> gnn_out.repeat(B, 1)     -> 全 batch 同一向量

即 GNN 分支不携带任何逐样本交互信息。这不污染 2×2 的对照效力（四格共享同一缺陷），
但它使「性能来自 graph message passing」这类归因失去依据（.tex L985）。

修复思路：**不是重写 GNN，而是把缺失的 batch 向量补上**
--------------------------------------------------------
父类第 264 行本来就会把 `data.gnn_batch` 传进 `GNNEncoder`，而 `GNNEncoder` 在 `batch is not None`
时走的正是 `global_mean_pool(x, batch)`（逐图池化）。所以只要 `gnn_batch` 存在且正确，父类**自己
的代码**就是对的。修复因此收敛为一个纯数据侧操作：把 DataLoader `follow_batch=['gnn_x']` 产出的
`gnn_x_batch` 别名为 `gnn_batch`。

这样做的三个好处：
1. 父类 `models/transformer_predictor.py` **一字不改**（硬约束），也不复制它的 forward；
2. `gnn_x_batch` 不存在时（batch=1 的在线 planner、未加 follow_batch 的旧调用方）本模块**什么都不做**，
   父类走原路径 `x.mean(dim=0)` → 与修复前**逐字节一致**，旧行为零改变；
3. 修复点只有一处，可单测。

**未走 `get_encoder_features()`** —— 那条路（transformer_predictor.py:363）判的是
`hasattr(data, 'batch')` 而非 `data.batch is not None`，单样本时 `None == 0` 得 Python bool，
`mask.sum()` 抛异常又被 `except: pass` 吞掉，context 分支静默归零。imitation 主线用 `forward()`
（:232 判了 `is not None`），不能改走那条路。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

# --- sys.path 自包含（同 magnet_relational.py 隔离模板）---
_THIS = Path(__file__).resolve()
_TRAIN_DIR = _THIS.parent
_GNN_ROOT = _THIS.parents[2]
_DEVKIT = _GNN_ROOT.parent
_MODELS = _DEVKIT / "models"
for _p in (str(_TRAIN_DIR), str(_GNN_ROOT), str(_MODELS), str(_DEVKIT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformer_predictor import TransformerPredictor  # noqa: E402  父类 models/（E0 生效副本）

__all__ = ["attach_gnn_batch", "MagnetImitationPooled", "GNN_POOLING_FIX_TAG"]

# 进 result JSON 的 provenance 标记：任何一次 run 都必须能一眼看出 GNN 分支是哪种语义
GNN_POOLING_FIX_TAG = "gnn_pooling_fix_v1"


class GnnBatchCoverageError(RuntimeError):
    """`gnn_x_batch` 未覆盖 batch 内每一个样本 —— 修复无法安全生效，必须暴露而非静默退化。

    父类在 `global_mean_pool` 返回行数 != batch_size 时会退回 `mean(0).repeat(B,1)`
    （transformer_predictor.py:272-274），那正是我们要消除的整 batch 共享。与其让它悄悄发生，
    不如在这里报错：真实数据里每个样本的 `gnn_x` 至少有 1 个节点
    （`dataset.py:640,667` 的兜底就是 `torch.zeros(1, 12)`），所以本异常只应在数据链路出问题时触发。
    """


def attach_gnn_batch(data, batch_size: Optional[int] = None, strict: bool = True):
    """把 `gnn_x_batch`（PyG follow_batch 产物）别名为父类读取的 `gnn_batch`。

    :param data: 一个 PyG ``Batch``/``GraphData``。**原地修改并返回**（与 PyG 惯例一致）。
    :param batch_size: 期望的图数量。``None`` 时从 ``data.num_graphs`` 推断。
    :param strict: 覆盖不全时抛 :class:`GnnBatchCoverageError`；``False`` 则退回旧行为并保持沉默。
    :returns: 同一个 ``data`` 对象。

    没有 ``gnn_x_batch``（单样本 planner / 旧 DataLoader）时直接原样返回 —— 这是修复前的行为，
    也是本函数「默认不改变旧行为」承诺的实现方式。
    """
    gnn_batch = getattr(data, "gnn_x_batch", None)
    if gnn_batch is None:
        return data                      # 旧路径：父类走 x.mean(dim=0)，逐字节一致

    if batch_size is None:
        batch_size = int(getattr(data, "num_graphs", 0)) or int(gnn_batch.max().item()) + 1

    covered = torch.unique(gnn_batch)
    if covered.numel() != batch_size:
        message = (f"gnn_x_batch 只覆盖 {covered.numel()} 个图，但 batch 有 {batch_size} 个样本；"
                   f"父类会退回 mean(0).repeat(B,1)，即修复失效。")
        if strict:
            raise GnnBatchCoverageError(message)
        return data

    data.gnn_batch = gnn_batch
    return data


class MagnetImitationPooled(TransformerPredictor):
    """父类 + 逐样本 GNN 池化。除 `gnn_batch` 的注入外，前向与父类完全相同。

    权重与父类逐键同名，故与既有 `best_model_default.pth` **checkpoint 互相兼容**：
    修复版可以直接加载未修复版训练出的权重，反之亦然（只是 GNN 分支的语义不同）。

    用法要点：DataLoader 必须建成 ``DataLoader(ds, ..., follow_batch=['gnn_x'])``，
    并确保搬到 device 时带上 ``gnn_x_batch``（见本文件末尾的接线清单）。
    """

    def __init__(self, strict_coverage: bool = True) -> None:
        super().__init__()
        self.strict_coverage = strict_coverage

    def forward(self, data):  # type: ignore[override]
        return super().forward(attach_gnn_batch(data, strict=self.strict_coverage))

    def get_encoder_features(self, data) -> torch.Tensor:  # type: ignore[override]
        return super().get_encoder_features(attach_gnn_batch(data, strict=self.strict_coverage))


# ---------------------------------------------------------------------------
# 接线清单（G2 对照 run 起跑前需要在 trainer 侧做的三处改动）
#
#   1. build_loaders():   DataLoader(..., follow_batch=['gnn_x'])   —— train 与 val 都要加
#   2. move_batch_to_device(): custom 列表加 "gnn_x_batch"          —— 否则它留在 CPU
#   3. train():           model = MagnetImitationPooled()          —— 由 CLI 开关选择
#
# 这三处**故意没有直接写进 `train_magnet_imitation.py`**：该文件正被运行中的 dev
# （run_2x2_ablation.py 每个 run 都重新 spawn 一次 python，会读到磁盘上的最新版本）消费，
# 现在改它会让 dev 的 12 个 run 分成两个代码版本。补丁见
# `docs/work_records/paper/MAGNET/VT-2026-04191_修改工作记录/D0_07_gnn_pooling_patch.md`，
# dev 跑完（约 07-31）后再应用。
# ---------------------------------------------------------------------------
