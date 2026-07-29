"""Z11 / D0#5：闭环 run 的 provenance 收口校验。

阶段15 §11.6 记录的缺口：阶段⑤/⑧ 的闭环资产（`eval_matrix.sh`、`boston_clean/`、`eval_full/`、
全部 `*weighted_average*.parquet`）随会话级 `scratchpad/` 一起消失。Boston-250 的
`0.122/0.095` 与阶段⑧ 的 `IDM 0.591/0.564`、`LogFuture 0.815/0.543` 因此**只有阶段记录文本、
没有可复现产物** —— 可以继续引用，但**不得声称 reproducible**。

本脚本把「什么算收口完成」变成可执行判据，供 08-01 D0 决策会逐条过：

    P1  每个 tag 都有 `<tag>.provenance.json`
    P2  provenance 里的 checkpoint SHA256 与磁盘上的文件**现在**仍然一致
    P3  产物目录在 `experiments/` 下，**没有任何** scratchpad 路径
    P4  `git_dirty_files == 0` —— 否则「命令 + commit」还原不了这次 run
    P5  feature schema hash 与当前代码一致（输入契约没漂移过）
    P6  每个 tag 都有 `*weighted_average*.parquet`（真的跑出分了，不是只留了个壳）
    P7  route_mode 有据可查（sidecar 记录 > 推断）

用法::

    python3 scripts/eval_closed_loop/verify_provenance.py \
        /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/experiments/closed_loop/pittsburgh_indist_N20

    # 严格模式：P4/P7 由警告升为失败（进论文正式表前必须用这个）
    python3 .../verify_provenance.py <group> --strict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_THIS = Path(__file__).resolve()
_GNN_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_GNN_ROOT / "scripts" / "analysis"))

EXPERIMENTS_ROOT = _GNN_ROOT / "experiments"


@dataclass
class Check:
    code: str
    ok: bool
    message: str
    severity: str = "error"        # error | warn


@dataclass
class TagReport:
    tag: str
    checks: List[Check] = field(default_factory=list)

    @property
    def failed(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "error"]

    @property
    def warned(self) -> List[Check]:
        return [c for c in self.checks if not c.ok and c.severity == "warn"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def current_schema_hash() -> Optional[str]:
    try:
        from result_manifest import schema_hash
        return schema_hash()
    except Exception:                                       # noqa: BLE001
        return None


def find_tags(group: Path) -> Dict[str, Optional[Path]]:
    """tag -> provenance 路径（None 表示该 tag 有产物但没有 provenance）。"""
    tags: Dict[str, Optional[Path]] = {}
    for prov in sorted(group.glob("*.provenance.json")):
        tags[prov.name[: -len(".provenance.json")]] = prov
    # 有 parquet 却没 provenance 的 tag 同样要暴露出来，否则 P1 形同虚设
    for parquet in group.rglob("*weighted_average*.parquet"):
        for part in parquet.relative_to(group).parts:
            if part in tags:
                break
        else:
            rel = parquet.relative_to(group).parts
            if rel:
                tags.setdefault(rel[0], None)
    return tags


def verify_tag(tag: str, prov_path: Optional[Path], group: Path, strict: bool) -> TagReport:
    report = TagReport(tag=tag)
    add = report.checks.append

    if prov_path is None or not prov_path.is_file():
        add(Check("P1", False, "缺 <tag>.provenance.json —— 这个 run 无法追溯"))
        return report
    add(Check("P1", True, f"provenance 存在: {prov_path.name}"))

    try:
        with open(prov_path, "r", encoding="utf-8") as fh:
            prov = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        add(Check("P1", False, f"provenance 不可解析: {exc}"))
        return report

    # P2 checkpoint 指纹
    ckpt = prov.get("checkpoint")
    recorded = prov.get("checkpoint_sha256")
    if not ckpt:
        add(Check("P2", True, "无 checkpoint（规则基线，如 idm/log_future）", severity="warn"))
    else:
        actual = sha256_file(Path(ckpt))
        if actual is None:
            add(Check("P2", False, f"checkpoint 已不存在: {ckpt}"))
        elif actual != recorded:
            add(Check("P2", False, f"checkpoint SHA256 与记录不符（记录 {str(recorded)[:12]}… "
                                   f"实际 {actual[:12]}…）—— 权重被换过"))
        else:
            add(Check("P2", True, f"checkpoint SHA256 一致 ({actual[:12]}…)"))

    # P3 产物位置
    bad_paths = [f"{k}={v}" for k, v in prov.items()
                 if isinstance(v, str) and "scratchpad" in v]
    in_experiments = str(group.resolve()).startswith(str(EXPERIMENTS_ROOT.resolve()))
    add(Check("P3", in_experiments and not bad_paths,
              "产物在 experiments/ 下且无 scratchpad 引用" if (in_experiments and not bad_paths)
              else f"违反 §11.6：in_experiments={in_experiments} scratchpad 引用={bad_paths}"))

    # P4 可从 commit 还原
    dirty = prov.get("git_dirty_files")
    add(Check("P4", dirty == 0,
              "git 工作区干净，命令 + commit 可还原" if dirty == 0
              else f"git 有 {dirty} 个脏文件 —— 此 run 无法仅凭 commit 还原",
              severity="error" if strict else "warn"))

    # P5 schema 契约
    cur, rec = current_schema_hash(), (prov.get("feature_schema") or {}).get("hash")
    if rec is None:
        add(Check("P5", False, "provenance 无 feature_schema.hash（旧版 harness 产出）",
                  severity="error" if strict else "warn"))
    elif cur is None:
        add(Check("P5", False, "当前代码算不出 schema hash —— result_manifest 不可导入"))
    else:
        add(Check("P5", cur == rec,
                  f"schema hash 一致 ({cur})" if cur == rec
                  else f"schema 漂移：run 用 {rec}，当前 {cur} —— 不可与新结果并表"))

    # P6 真的出分了
    parquets = list((group / tag).rglob("*weighted_average*.parquet")) if (group / tag).is_dir() \
        else [p for p in group.rglob("*weighted_average*.parquet") if tag in str(p)]
    add(Check("P6", bool(parquets),
              f"找到 {len(parquets)} 个 weighted_average parquet" if parquets
              else "无 weighted_average parquet —— 没有实际分数产物"))

    # P7 route_mode 溯源
    rm = prov.get("route_mode")
    if isinstance(rm, dict):
        # 精确判 source，不做子串匹配 —— 推断值的 source 文本里同样含 "train_meta.json"
        # （"inferred (no train_meta.json sidecar)"），子串判定会把它误判为已记录。
        recorded_source = rm.get("source", "")
        is_recorded = recorded_source == "train_meta.json"
        add(Check("P7", is_recorded,
                  f"route_mode={rm.get('value')} 来源={recorded_source}"
                  + ("" if is_recorded else " —— 是推断值，不是记录值；正式表须补 sidecar"),
                  severity="error" if strict else "warn"))
    else:
        add(Check("P7", False, "provenance 无 route_mode 字段（旧版 harness）",
                  severity="error" if strict else "warn"))

    # P8 devkit 受管文件（在线 planner / 父类）与 vendor 副本是否一致
    patches = prov.get("devkit_patches")
    if patches is None:
        add(Check("P8", False, "provenance 无 devkit_patches 字段（早于方案 A 的 run）",
                  severity="error" if strict else "warn"))
    elif not patches.get("available"):
        add(Check("P8", False, f"devkit 受管文件状态不可用：{patches.get('reason')}"))
    else:
        add(Check("P8", bool(patches.get("in_sync")),
                  "devkit 生效文件与 vendor 副本一致，在线代码可由 GNN_Trans commit 还原"
                  if patches.get("in_sync")
                  else "devkit 与 vendor 不一致 —— 这次 run 的在线 planner 代码不可追溯"))

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="闭环 run provenance 收口校验（Z11 / D0#5）")
    p.add_argument("group", type=Path, help="闭环 group 目录")
    p.add_argument("--strict", action="store_true",
                   help="P4/P7 由警告升为失败（进论文正式表前必须用）")
    p.add_argument("--json_out", type=Path, default=None)
    args = p.parse_args()

    group = args.group.resolve()
    if not group.is_dir():
        raise SystemExit(f"group 不存在: {group}")

    tags = find_tags(group)
    if not tags:
        print(f"⚠️ {group} 下没有任何 tag（既无 provenance 也无 parquet）—— 这个 group 是空的")
        return 1

    reports = [verify_tag(tag, prov, group, args.strict) for tag, prov in sorted(tags.items())]

    print("=" * 84)
    print(f" 闭环 provenance 校验 —— {group}")
    print(f" {len(reports)} 个 tag   strict={args.strict}")
    print("=" * 84)
    for r in reports:
        status = "❌" if r.failed else ("⚠️" if r.warned else "✅")
        print(f"\n {status} {r.tag}")
        for c in r.checks:
            mark = "✅" if c.ok else ("⚠️" if c.severity == "warn" else "❌")
            print(f"     {mark} {c.code} {c.message}")

    n_fail = sum(1 for r in reports if r.failed)
    n_warn = sum(1 for r in reports if r.warned and not r.failed)
    print("\n" + "-" * 84)
    print(f" 判定: {len(reports) - n_fail - n_warn} 通过 / {n_warn} 警告 / {n_fail} 失败")
    if n_fail:
        print(" ❌ 未收口 —— 这些结果**不得**进重投稿的正式表。")
    elif n_warn:
        print(" ⚠️ 基本收口，但有警告项；进正式表前用 --strict 复跑并清零。")
    else:
        print(" ✅ 收口完成，可进正式表。")
    print("=" * 84)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump([{"tag": r.tag, "checks": [c.__dict__ for c in r.checks]} for r in reports],
                      fh, indent=2, ensure_ascii=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
