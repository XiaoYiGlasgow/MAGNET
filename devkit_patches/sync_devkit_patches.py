"""方案 A（Q1 裁定）：把 nuplan-devkit 里我们改过的文件 vendor 进 GNN_Trans 并做同步校验。

为什么需要这个
--------------
VT-2026-04191 的实验链横跨三个嵌套 git 仓库：

    /mnt/local/Xiao_Yi                      0 commits   remote=GNN_Trans_Private.git
    /mnt/local/Xiao_Yi/nuplan-devkit        0 commits   remote=<无>
    .../nuplan-devkit/GNN_Trans            17 commits   remote=GNN_Trans.git   ← 唯一有历史

**在线 planner 与父类模型都在中间那个零提交的仓库里。** 于是 D0#2 的验收标准
「命令 + commit 可完整还原实验」对闭环结果**根本无法满足** —— 不是被脏文件挡住，
而是那些文件所在的仓库连一个 commit 都没有。

方案 A 不给 devkit fork 建历史（体量大、与上游关系未理清、且改动只有 3 个文件），
改为：**把这 3 个文件复制进 GNN_Trans 版本库，并强制两边逐字节一致**。
于是 GNN_Trans 的一个 commit 就能完整还原「训练 + 在线 planner」两侧的代码。

三个受管文件
------------
    nuplan/planning/simulation/planner/gnn_trans_planner.py          在线 planner（阶段②③ 重写）
    nuplan/planning/script/config/simulation/planner/gnn_trans_planner.yaml   planner 注册与默认 ckpt
    models/transformer_predictor.py                                  父类模型（E0 后一字未改，但必须锁版本）

用法
----
    python3 devkit_patches/sync_devkit_patches.py --check     # 校验：不一致则非零退出（D0 门用这个）
    python3 devkit_patches/sync_devkit_patches.py --pull      # devkit -> vendor（改了 devkit 后同步进版本库）
    python3 devkit_patches/sync_devkit_patches.py --push      # vendor -> devkit（换机器/重建 devkit 时还原）
    python3 devkit_patches/sync_devkit_patches.py --manifest  # 只刷新 manifest.json

⚠️ `--push` 会覆盖 devkit 里的生效文件。父类 `models/transformer_predictor.py` 属于
「一字不改」硬约束的对象 —— push 前脚本会额外确认它与 vendor 副本一致，不一致直接拒绝。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_THIS = Path(__file__).resolve()
PATCH_DIR = _THIS.parent
FILES_DIR = PATCH_DIR / "files"
MANIFEST = PATCH_DIR / "manifest.json"
GNN_ROOT = PATCH_DIR.parent
DEVKIT_ROOT = GNN_ROOT.parent

# 受管文件：devkit 内的相对路径。新增受管文件只需往这里加一行。
MANAGED: List[str] = [
    "nuplan/planning/simulation/planner/gnn_trans_planner.py",
    "nuplan/planning/script/config/simulation/planner/gnn_trans_planner.yaml",
    # PlanTF 外部 baseline 的 planner 注册（gate⑤，2026-07-31）。planner 实现本身在
    # /mnt/local/Xiao_Yi/external/planTF（上游代码，不 vendor），我方改动只有这个注册文件
    # 与 embedding.py 的 natten try/except —— 后者在 planTF 树内，另行记录。
    "nuplan/planning/script/config/simulation/planner/plantf_planner.yaml",
    "models/transformer_predictor.py",
]

# 「一字不改」硬约束保护的文件：--push 前若与 vendor 不一致，拒绝执行
FROZEN = {"models/transformer_predictor.py"}


def flat_name(rel: str) -> str:
    return rel.replace("/", "__")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            d.update(block)
    return d.hexdigest()


@dataclass
class Pair:
    rel: str
    live: Path
    vendored: Path

    @property
    def live_sha(self) -> str | None:
        return sha256(self.live)

    @property
    def vendored_sha(self) -> str | None:
        return sha256(self.vendored)

    @property
    def in_sync(self) -> bool:
        a, b = self.live_sha, self.vendored_sha
        return a is not None and a == b


def pairs() -> List[Pair]:
    return [Pair(rel, DEVKIT_ROOT / rel, FILES_DIR / flat_name(rel)) for rel in MANAGED]


def write_manifest(ps: List[Pair]) -> Dict:
    data = {
        "scheme": "devkit_patches/A",
        "devkit_root": str(DEVKIT_ROOT),
        "note": ("nuplan-devkit 是零提交仓库；这些文件的权威版本以本目录的 vendor 副本为准，"
                 "由 sync_devkit_patches.py --check 强制两边逐字节一致。"),
        "files": [{
            "devkit_relpath": p.rel,
            "vendored": f"devkit_patches/files/{flat_name(p.rel)}",
            "sha256": p.vendored_sha,
            "frozen": p.rel in FROZEN,
        } for p in ps],
    }
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def cmd_check(ps: List[Pair]) -> int:
    print("=" * 84)
    print(" devkit_patches 同步校验（方案 A）")
    print("=" * 84)
    bad = 0
    for p in ps:
        live, vend = p.live_sha, p.vendored_sha
        if live is None:
            print(f"  ❌ {p.rel}\n       devkit 里的文件不存在: {p.live}")
            bad += 1
        elif vend is None:
            print(f"  ❌ {p.rel}\n       vendor 副本缺失，跑 --pull")
            bad += 1
        elif live != vend:
            print(f"  ❌ {p.rel}\n       devkit={live[:16]}…  vendor={vend[:16]}…  **不一致**")
            print("       改了 devkit 就跑 --pull；改了 vendor 就跑 --push。不要手工二选一。")
            bad += 1
        else:
            frozen = "（frozen）" if p.rel in FROZEN else ""
            print(f"  ✅ {p.rel} {frozen}\n       {live[:16]}…")
    print("-" * 84)
    if bad:
        print(f" ❌ {bad}/{len(ps)} 不同步 —— D0#2 不通过，正式 run 不得起跑。")
    else:
        print(f" ✅ {len(ps)}/{len(ps)} 同步。GNN_Trans 的一个 commit 可完整还原训练 + 在线 planner 两侧代码。")
    print("=" * 84)
    return 1 if bad else 0


def cmd_pull(ps: List[Pair]) -> int:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    for p in ps:
        if not p.live.is_file():
            print(f"  ❌ 跳过（devkit 无此文件）: {p.rel}")
            continue
        changed = p.live_sha != p.vendored_sha
        shutil.copy2(p.live, p.vendored)
        print(("  ✏️ 更新 " if changed else "  · 无变化 ") + p.rel)
    write_manifest(ps)
    print(f"\n  manifest -> {MANIFEST}")
    return 0


def cmd_push(ps: List[Pair]) -> int:
    for p in ps:
        if p.rel in FROZEN and not p.in_sync and p.live.is_file():
            print(f"  ❌ 拒绝 push：{p.rel} 属于「一字不改」保护对象，且当前与 vendor 不一致。")
            print("     先人工确认哪一份是对的，再决定 --pull 还是手工修复。")
            return 2
    for p in ps:
        if not p.vendored.is_file():
            print(f"  ❌ 跳过（无 vendor 副本）: {p.rel}")
            continue
        p.live.parent.mkdir(parents=True, exist_ok=True)
        changed = p.live_sha != p.vendored_sha
        shutil.copy2(p.vendored, p.live)
        print(("  ✏️ 写回 " if changed else "  · 无变化 ") + p.rel)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="devkit 受管文件同步（方案 A）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="校验两边逐字节一致（D0 门）")
    g.add_argument("--pull", action="store_true", help="devkit -> vendor")
    g.add_argument("--push", action="store_true", help="vendor -> devkit（会覆盖生效文件）")
    g.add_argument("--manifest", action="store_true", help="只刷新 manifest.json")
    a = ap.parse_args()

    ps = pairs()
    if a.check:
        return cmd_check(ps)
    if a.pull:
        return cmd_pull(ps)
    if a.push:
        return cmd_push(ps)
    write_manifest(ps)
    print(f"manifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
