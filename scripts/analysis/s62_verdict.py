"""§6.2「路线 B 前提」判别实验的**可执行判读**（Q110 §6.2，两种读法事前写死）。

## 🔴 本文件写于 A 臂出数**之前**（2026-08-25 20:5x，A 臂 ETA 21:52）

这是全项目**唯一真正需要预注册**的一条 —— 它锁的是**主张**（「交互使用度低是任务的性质
还是结构的错」），不是「某个子部件坏没坏」。把阈值写进代码，就无法在看到数字之后再挑读法。

## 三个臂

| 臂 | 结构 | 任务 | ckpt |
|---|---|---|---|
| **legacy** | 旧（三次均值池化 + 未 mask 的 padding） | 单 ego 输出 | `knobs/fr_base_L1/seed{1,2,42}` |
| **A** | 新（无池化、已 mask padding） | **单 ego 输出** | `joint_s2_egoonly/seed{1,2,42}` |
| **B** | 新（同上） | **联合多智能体** | `joint_s2_fixed/seed{1,2,42}` |

## 🔴 度量与口径（不得换）

* 度量 = **模型级**使用度 `U = mean_disp / base_scale`，干预 = `count_sat.remove_agent`
  （**真移除**节点 + 其全部边 + 其在 `data.x` 的向量行）。
  🔴 **不是** Q74 的 G2=0.72%（那是 **GNN 通路级**，旧 audit 不动 `data.x`）；模型级基准是 G4=6.87%。
* **主口径 = `U_modes`**（对 M 个模态逐一自比后平均，**完全不依赖选模**）。
  理由：B 臂实测**模态翻转率 53–57%**，argmax 口径有一半「响应」是选模抖动；
  而 legacy 是单输出（M=1）⇒ 三口径恒等 ⇒ **只有 `U_modes` 在三个臂上是同一个 estimand**。
* 点估计取**中位数**（U 分布极右偏，均值到 200%+）；区间用 **log 聚类 bootstrap**（分析单位 = log）。
* 配对：同 token、同 audit seed、seed-matched（treat_seedX ↔ base_seedX）。

## 🔴🔴 两种读法（Q110 §6.2 原文，逐字）

* **A ≫ 旧 且 B ≈ A** ⇒ **路线 B 的前提不成立** —— 是结构的错不是任务的错，
  联合预测只是锦上添花。此时**必须重写动机**（改成「结构缺陷诊断 + 修复」），
  并重新评估是否还要保留联合预测。
* **A ≈ 旧 且 B ≫ A** ⇒ **前提成立，且这是一个很强的正面结果** ——
  结构修好之后交互仍然没负载，说明确实是任务的性质。

## 🔴 操作化（本文件的贡献：把 ≫ / ≈ 写死）

* `X ≫ Y` ⟺ 配对 ΔS(X − Y) 的 **log-cluster 95% CI 排除 0 且为正**；
* `X ≈ Y` ⟺ 该 CI **含 0**。

⚠️ **原预注册只列了两种组合，但逻辑上有四种。** 落在另外两种时**不得**硬塞进已写死的两条里 ——
本脚本会明确报「**落在预注册之外**」并给出该情形的诚实解释，由用户裁定措辞。
这一条同样是事前写下的。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

#: 模型级基准（Q74 三 seed pooled，**旧档 L1**，仅作量级参照，🔴 不得与新档绝对值并列）
G4_MODEL_LEVEL_OLD_TIER = 0.068697


def _ci(d: Dict) -> Optional[Dict[str, float]]:
    """从 `paired_usage_delta.py` 的产物里取主判据（点估计 + log-cluster CI）。

    🔴 **2026-08-25 22:1x 修正的是「解析」，不是「判据」** —— 首版按猜测的键名写，
    与实际产物对不上，脚本因此**正确地拒绝判读**（报「未覆盖组合 ⇒ 停下来查」）而没有瞎猜。
    实际结构是 ``median_of_ratios.{paired_delta_median, paired_delta_ci95_logcluster}``。
    :func:`relation` 与 :func:`verdict`（**决策规则本身**）**一字未动**，
    其函数体 sha256[:16] = ``d9ddb6db6b5e35a0``，与冻结时逐位相同。

    🔴 同时强制主口径：产物的 ``metric`` 必须是 ``modes``（§6.2 预注册主口径，不依赖选模）。
    不是就返回 None ⇒ 判读脚本会拒判，而不是拿别的口径顶上。
    """
    if d.get("metric") not in (None, "modes"):
        return None                       # 口径不对 ⇒ 拒判（None 会被 relation() 变成 "?"）
    mor = d.get("median_of_ratios")
    if isinstance(mor, dict) and "paired_delta_ci95_logcluster" in mor:
        lo, hi = mor["paired_delta_ci95_logcluster"]
        return {"point": float(mor["paired_delta_median"]), "lo": float(lo), "hi": float(hi)}
    return None


def relation(ci: Optional[Dict[str, float]]) -> str:
    """`≫` / `≈` / `≪`（CI 排除 0 且为负）。CI 缺失返回 `?`。"""
    if ci is None:
        return "?"
    if ci["lo"] > 0:
        return "≫"
    if ci["hi"] < 0:
        return "≪"
    return "≈"


def verdict(a_vs_old: str, b_vs_a: str) -> Dict[str, str]:
    """两种事前写死的读法 + 两种「预注册之外」的诚实处置。"""
    if a_vs_old == "≫" and b_vs_a == "≈":
        return {"case": "预注册读法 ①",
                "conclusion": "🔴 **路线 B 的前提不成立** —— 是结构的错不是任务的错。",
                "action": "必须重写动机（改成「结构缺陷诊断 + 修复」），并重新评估是否保留联合预测。"}
    if a_vs_old == "≈" and b_vs_a == "≫":
        return {"case": "预注册读法 ②",
                "conclusion": "🟢 **前提成立，且是一个很强的正面结果** —— 结构修好后交互仍无负载，"
                              "说明确实是任务的性质。",
                "action": "按原计划推进 S3→S5，动机不改。"}
    if a_vs_old == "≫" and b_vs_a == "≫":
        return {"case": "🔴 落在预注册之外（两者都涨）",
                "conclusion": "结构与任务**各自独立地**抬高了交互使用度。",
                "action": "🔴 **不得**塞进上面任一条。诚实写法：两个因素都成立，"
                          "须报**各自的效应量**并说明二者不可互相替代；动机段要同时承认结构缺陷。"}
    if a_vs_old == "≈" and b_vs_a == "≈":
        return {"case": "🔴 落在预注册之外（两者都不动）",
                "conclusion": "改结构与改任务**都没有**抬高交互使用度。",
                "action": "🔴 这会同时否掉「结构是主因」与「任务是主因」两种解释 ⇒ "
                          "须回到测量本身（估计量、剂量、子集是否有功效），先做 MDE 再谈结论。"}
    return {"case": f"🔴 未覆盖组合（A vs 旧 = {a_vs_old}，B vs A = {b_vs_a}）",
            "conclusion": "出现了负向关系或数据缺失。",
            "action": "停下来查数据与通路，不得直接判读。"}


def main() -> None:
    p = argparse.ArgumentParser(description="§6.2 判别实验的预注册判读")
    p.add_argument("--a_vs_legacy", required=True, help="paired_usage_delta 产物：A − legacy")
    p.add_argument("--b_vs_a", required=True, help="paired_usage_delta 产物：B − A")
    p.add_argument("--usage_json", nargs="*", default=[],
                   help="三个臂的 joint_usage_probe 产物（只用于把绝对值一并列出，不参与判读）")
    p.add_argument("--out")
    a = p.parse_args()

    d1 = json.loads(Path(a.a_vs_legacy).read_text())
    d2 = json.loads(Path(a.b_vs_a).read_text())
    ci1, ci2 = _ci(d1), _ci(d2)
    r1, r2 = relation(ci1), relation(ci2)
    v = verdict(r1, r2)

    print("\n=== §6.2 判别实验 —— 预注册判读 ===")
    print("🔴 判据写于 A 臂出数之前（2026-08-25 20:5x），阈值写死在 `s62_verdict.py`，不得事后调整。")
    for nm, ci, r in (("A vs legacy（修结构）", ci1, r1), ("B vs A（换任务）", ci2, r2)):
        if ci is None:
            print(f"  {nm:22s} 读不到 CI ⇒ {r}")
        else:
            # 🔴 产物里存的是**比值**；`pt`（百分点）要 ×100。判据只看符号与是否跨 0 ⇒ **不受影响**。
            print(f"  {nm:22s} ΔS = {ci['point'] * 100:+.4f} pt  "
                  f"CI95[{ci['lo'] * 100:+.4f}, {ci['hi'] * 100:+.4f}] ⇒ **{r}**")
    print(f"\n【{v['case']}】")
    print(f"  结论：{v['conclusion']}")
    print(f"  动作：{v['action']}")
    print("\n🔴 口径提醒：主口径 = `U_modes`（不依赖选模）；点估计取中位数；"
          "A vs legacy 是**跨 ckpt 家族**，只报配对 ΔS，**不得并列绝对使用度**（§8.2）。")
    if a.usage_json:
        print("\n（绝对值仅供上下文，不参与判读）")
        for f in a.usage_json:
            d = json.loads(Path(f).read_text())
            ra = d.get("remove_agent", {})
            print(f"  {Path(f).stem:16s} U_modes 中位 {ra.get('U_modes_median', float('nan')) * 100:7.3f}%"
                  f"  U 中位 {ra.get('U_median', float('nan')) * 100:7.3f}%"
                  f"  翻转率 {ra.get('mode_flip_rate', float('nan')) * 100:6.2f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"a_vs_legacy": ci1, "b_vs_a": ci2, "relation_a_vs_legacy": r1,
             "relation_b_vs_a": r2, **v,
             "preregistered_before": "A 臂出数之前（2026-08-25 20:5x）",
             "primary_caliber": "U_modes（不依赖选模）/ 中位数 / log-cluster CI"},
            ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
