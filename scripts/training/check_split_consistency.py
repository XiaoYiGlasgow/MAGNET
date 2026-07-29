"""校验两个数据集的 split3 manifest 完全一致（VT-2026-04191 阶段⑬）。

2×2 factorial 消融（B/R/P/RP）要求四组共享**同一 480/60/60 log 划分**，否则
「route 有无」「增广有无」的效应会与「数据划分差异」混淆，消融失去可解释性。

判据：
  1. train / val / test 的 **log 集合**分别完全一致；
  2. 三个 split 两两无 log overlap（各自内部的 log-disjoint 性）；
  3. 各 split 的样本数一致；
  4. seed / ratio 一致。

运行：
  /home/team/anaconda3/envs/nuplan/bin/python scripts/training/check_split_consistency.py \
      --a /abs/.../processed_features_magnet_e1_full_split3/split3_manifest.json \
      --b /abs/.../processed_features_magnet_e2_route_v2_split3/split3_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SPLITS = ("train", "val", "test")


def load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="校验两个 split3 manifest 是否完全一致")
    ap.add_argument("--a", required=True, type=str)
    ap.add_argument("--b", required=True, type=str)
    args = ap.parse_args()

    ma, mb = load(Path(args.a)), load(Path(args.b))
    ok = True

    print("=" * 68)
    print("split3 一致性校验")
    print("=" * 68)
    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print()

    # --- 判据 2：各自内部 log-disjoint ---
    for name, m in (("A", ma), ("B", mb)):
        for i, s1 in enumerate(_SPLITS):
            for s2 in _SPLITS[i + 1:]:
                ov = set(m["logs"][s1]) & set(m["logs"][s2])
                if ov:
                    print(f"[2] ❌ {name}: {s1}∩{s2} 有 {len(ov)} 个 log 重叠")
                    ok = False
    print(f"[2] 各自三 split 两两无 log overlap: {'✅' if ok else '❌'}")

    # --- 判据 1：log 集合一致 ---
    for s in _SPLITS:
        la, lb = set(ma["logs"][s]), set(mb["logs"][s])
        same = la == lb
        ok &= same
        extra = f"  A-only={len(la - lb)} B-only={len(lb - la)}" if not same else ""
        print(f"[1] {s:5s} log 集合一致: {'✅' if same else '❌'}  (|A|={len(la)}, |B|={len(lb)}){extra}")

    # --- 判据 3：样本数一致 ---
    for s in _SPLITS:
        ka, kb = ma.get(f"n_samples_{s}"), mb.get(f"n_samples_{s}")
        same = ka == kb
        ok &= same
        print(f"[3] {s:5s} 样本数一致: {'✅' if same else '❌'}  (A={ka}, B={kb})")

    # --- 判据 4：seed / ratio ---
    for k in ("seed", "val_ratio", "test_ratio"):
        same = ma.get(k) == mb.get(k)
        ok &= same
        print(f"[4] {k:11s} 一致: {'✅' if same else '❌'}  (A={ma.get(k)}, B={mb.get(k)})")

    print()
    print(f"总判定: {'✅ 两数据集 split 完全一致，可用于 2×2 factorial 消融' if ok else '❌ 不一致，禁止用于消融'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
