"""danger_score.py (numpy) 与 danger_score_torch.py (torch) 配置同步守卫。

两份 danger 评分：
  - `danger_score.py`（numpy）：数据抽取（ttc_filter）+ 评估 / benchmark 测量危险分。
  - `danger_score_torch.py`（torch）：训练损失（模型被训练去超过 GT 危险），求可微梯度。

若关键参数在两份之间漂移，训练优化的危险定义与评估测量的危险定义就**不是一回事**
（静默错配，不报错、难发现）。本测试逐字段断言两份 config 的公共字段值相等，并锁定
已知的字段集差异——出现新的未同步字段即报红，提示复核是否需要同步。
"""
import dataclasses
import os
import sys

# danger_score / danger_score_torch 均在 src/shared/utils
_UTILS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "shared", "utils"))
if _UTILS not in sys.path:
    sys.path.insert(0, _UTILS)

from danger_score import DangerScoreConfig             # noqa: E402
from danger_score_torch import DangerScoreConfigTorch  # noqa: E402

# 已知且有意的字段集差异：numpy 版多 9 个 svo_*_delta_* 参数（仅 numpy 端 SVO 机制用，
# torch 损失不计算这部分）。torch 版不应有 numpy 版没有的字段。
_KNOWN_NUMPY_ONLY = {
    "svo_aggressive_delta_x", "svo_aggressive_delta_y",
    "svo_approach_delta_x", "svo_approach_delta_y",
    "svo_avoidance_delta_x", "svo_avoidance_delta_y",
    "svo_social_force_delta_y",
    "svo_yield_delta_x", "svo_yield_delta_y",
}


def _values(cls) -> dict:
    """实例化 config dataclass，返回 {字段名: 值}（处理 default_factory）。"""
    inst = cls()
    return {f.name: getattr(inst, f.name) for f in dataclasses.fields(cls)}


def test_common_fields_identical():
    """公共字段（collision_distance / near_miss / 全 weights / thresholds 等）值必须逐一相等。"""
    a, b = _values(DangerScoreConfig), _values(DangerScoreConfigTorch)
    common = set(a) & set(b)
    mismatches = {k: (a[k], b[k]) for k in common if a[k] != b[k]}
    assert not mismatches, (
        "danger_score numpy/torch 公共字段漂移（训练与评估会测量不同的危险定义）：\n"
        + "\n".join(f"  {k}: numpy={va} vs torch={vb}" for k, (va, vb) in mismatches.items())
    )


def test_field_set_divergence_is_known():
    """字段集差异必须等于已知白名单——出现新的未同步字段即报红，提示复核。"""
    a, b = _values(DangerScoreConfig), _values(DangerScoreConfigTorch)
    only_numpy = set(a) - set(b)
    only_torch = set(b) - set(a)

    assert only_torch == set(), (
        f"torch 版多出 numpy 版没有的字段（需同步回 danger_score.py）：{sorted(only_torch)}"
    )
    unexpected = only_numpy - _KNOWN_NUMPY_ONLY
    assert unexpected == set(), (
        f"numpy 版新增了未同步到 torch 的字段：{sorted(unexpected)}。\n"
        "若该字段影响 outcome danger，必须同步到 danger_score_torch.py；"
        "若仅 numpy 端 SVO 机制用，请把它加入本测试的 _KNOWN_NUMPY_ONLY 白名单。"
    )
