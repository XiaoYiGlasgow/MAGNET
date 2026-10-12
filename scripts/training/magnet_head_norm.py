"""修 A（Q105）：给输出头的**第二个 ReLU 补前置 LayerNorm**，消掉塌陷的共模门。

缺陷（2026-08-21 机制定位，见 `progress_20260821.md` 阶段271）
---------------------------------------------------------------
父类 `TransformerPredictor.output_head` 是::

    [0] Linear(128,128) → [1] ReLU → [2] LayerNorm(128) → [3] Dropout(0.1)
                        → [4] Linear(128,128) → [5] ReLU → [6] Linear(128,120)

`LayerNorm` 只保护了**第一个** ReLU；`Linear[4]` 的输出没有任何东西把它拉回零中心。
实测（`scratchpad/diag_root.py`，300 场景/格，CPU）：

* 预激活均值 = ``w̄·x + b̄``，其中 ``w̄`` 是 ``W[4]`` 的平均行。
  ``corr(w̄·x, pre_mean) = 1.0000``、残差最大 ``4.02e-07`` ⇒ 分解精确；
* ``b̄ = −0.068``、``w̄·β = +0.247``（**正的**）⇒ 负偏移**既不来自 bias 也不来自 LN 的 β**，
  全部来自场景相关的 ``w̄·x ≈ −2``；
* ``‖w̄‖ = 0.5882``，而 128 行方向独立时应为 ``行范数/√128 = 0.104``
  ⇒ **共模对齐 5.66×**：训练学出了一个**全局门**；
* 于是 128 个单元的预激活挤在 ``N(−2, 1)`` 里，逐场景只有 **4–10 个**为正
  （l1 中位 10、fr_base 中位 4），尾巴整体过零时整层输出全零 ⇒ **塌陷**
  （`fr_base` 16.3/19.0/0.3%、旧档 B 格最高 29.1%；未改动基线 `l1_*` 仅 0–0.2%）。
* 塌陷时末层 ``Linear[6]`` 收到零向量，输出退化为纯 bias（实测 |mean| = 4.46e-4）
  ⇒ 预测 6 s 位移仅 **2 cm**，闭环里表现为「不作为失效」。

修法
----
在 ``Linear[4]`` 与 ``ReLU`` 之间插一个 ``LayerNorm(128)``。LayerNorm 减去**每样本**的特征均值
⇒ 共模项 ``w̄·x`` 在预激活里**恒等于 0** ⇒ 均值被钉在 0，128 个单元同时为负在数学上不可能。

🔑 **这与 PlanTF 的做法同构**：``external/planTF/src/models/planTF/modules/trajectory_decoder.py``
的 ``loc`` 头是 ``Linear → LayerNorm → ReLU → Linear``（LN **前置**），结构上排除该失效模式。

设计约束（照本仓库既有旋钮的规矩）
------------------------------------
1. **不改 `models/transformer_predictor.py`** —— 它是 5 个 `devkit_patches` 受管文件之一，
   改父文件要 `sync_devkit_patches.py --check` = 5/5，且会破坏全部历史 ckpt 的可复现性。
2. **继承 `MagnetImitationPooled` 而不是 `TransformerPredictor`** —— 对照基线 `fr_base` 的 recipe 是
   ``class=MagnetImitationPooled, gnn_pooling_fix=true``（读其 sidecar 得来），
   继承父类会同时改动池化语义 ⇒ 差两个变量（Q45 同类坑）。
3. **初始化逐字复制父类的「有效」行为**：父类 `__init__` 里那句
   ``nn.init.normal_(output_head[-1].weight, std=0.1)`` 是**死代码** ——
   紧随其后的 ``for module in self.output_head`` 循环用 ``xavier_normal_`` 把它覆盖了。
   本子类复制**实际生效**的那一版（xavier + zero bias），使唯一变量真的只有 LayerNorm。

state_dict 兼容性
-----------------
插入使下标后移：``output_head.6`` 由 ReLU 变 LayerNorm 的位置、末层 Linear 从 ``.6`` 变 ``.7``，
并新增 ``output_head.5.{weight,bias}``（LN 的 affine）。
⇒ **与基线 ckpt 故意不兼容**：建错类时 ``load_state_dict(strict=True)`` 会**响亮报错**，
这是有意的（本仓库偏好显式失败而非静默错位）。

塌陷探针不受影响：`run_interaction_audit.py::_last_relu` 取的是 ``relus[-1]``，
新头里仍是最后一个 ReLU（下标 6），hook 照常触发。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from magnet_imitation_pooled import MagnetImitationPooled  # noqa: E402

__all__ = ["MagnetHeadNorm", "build_normed_output_head"]

HEAD_NORM_VERSION = "MAGNET_HEAD_NORM_v1_20260821"


def build_normed_output_head(dim: int = 128, out_dim: int = 120,
                             dropout: float = 0.1) -> nn.Sequential:
    """与父类 `output_head` 逐层同构，唯一差别 = 第二个 ReLU 前多一个 `LayerNorm`。

    :param dim: 隐藏维（父类固定 128）。
    :param out_dim: 输出维（父类固定 120 = 60 步 × 2）。
    :param dropout: 与父类一致的 dropout 概率。
    :returns: 新的输出头。初始化与父类**实际生效**的那一版一致（xavier + zero bias）。
    """
    head = nn.Sequential(
        nn.Linear(dim, dim),        # [0] 与父类同
        nn.ReLU(),                  # [1]
        nn.LayerNorm(dim),          # [2] 与父类同（第一个 ReLU 之后）
        nn.Dropout(dropout),        # [3]
        nn.Linear(dim, dim),        # [4]
        nn.LayerNorm(dim),          # [5] ★ 修 A 新增：前置归一化，消掉共模门
        nn.ReLU(),                  # [6]
        nn.Linear(dim, out_dim),    # [7]
    )
    # 🔴 复制父类**生效**的初始化（transformer_predictor.py:127-131 的 xavier 循环），
    #    不复制被它覆盖掉的 std=0.1 死代码 —— 唯一变量必须只有 LayerNorm。
    with torch.no_grad():
        for module in head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return head


class MagnetHeadNorm(MagnetImitationPooled):
    """`MagnetImitationPooled` + 输出头前置 LayerNorm（修 A）。

    与 `fr_base`（`MagnetImitationPooled`，`gnn_pooling_fix=true`）**恰好差一个变量**。
    前向路径一字未改：父类 `forward` 里 ``output = self.output_head(fused_feat)``
    会自动走到本类替换后的头。
    """

    def __init__(self, strict_coverage: bool = True) -> None:
        super().__init__(strict_coverage=strict_coverage)
        self.head_norm = True
        self.output_head = build_normed_output_head()

    def active_head(self) -> nn.Sequential:
        """给塌陷探针用的显式挂点（与 `MagnetImitationMultiModal` 同接口）。"""
        return self.output_head
