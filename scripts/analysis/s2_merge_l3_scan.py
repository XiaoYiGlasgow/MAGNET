"""S2 —— 合并 L3 坏场景扫描的 14 个分片，写出剔除后的共享清单。

对策选型（08-02，冒烟 + 20 log 诊断之后）：

  (A) 从共享清单剔除坏 token  ← **采用**
  (B) 改 PlanTF 的 route_utils 容错
  (C) 回退 remove_invalid_goals=True

选 (A) 的理由：
  * 它保住了第二轮唯一的硬约束 —— **两条管线消费同一份清单**。剔除标准是
    「PlanTF 的 feature builder 能否成功构建」，客观、可复现，两个模型同样施加，
    配对性完全不受影响。
  * (B) 要改别人已发表的代码，且「route 修正失败时退回用什么」没有无争议的定义，
    是对 PlanTF recipe 的第二处偏离；
  * (C) 会把 L3 上限从 245.5 k 打回约 72 k（估计值），Q59 白裁，阶梯被压扁。

⚠️ 剔除是**由 PlanTF 侧的失败定义的**，而 MAGNET 侧这些场景并不崩（其冒烟 [S5] 通过）。
   也就是说 MAGNET 会因为迁就 PlanTF 而少掉这批样本。这对我方是**保守方向**
   （MAGNET 拿到的数据更少却仍要被比较），写作时必须显式声明，不得省略。

用法：
    python s2_merge_l3_scan.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
SCAN_DIR = GNN / "experiments/scaling_ladder/l3_scan"
MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42.json"
OUT_MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42_filtered.json"
OUT_REPORT = GNN / "experiments/scaling_ladder/s2_l3_scan_summary.json"


def main() -> int:
    """合并分片、统计、写出剔除后的清单。"""
    shards = sorted(SCAN_DIR.glob("shard_*.json"))
    if not shards:
        logger.error("没有分片产物，先跑 s2_scan_l3_failures_20260802.sh")
        return 1

    bad_tokens: set[str] = set()
    per_log_fail: Counter = Counter()
    per_log_total: Counter = Counter()
    kinds: Counter = Counter()
    n_scenarios = 0

    for shard in shards:
        data = json.loads(shard.read_text())
        n_scenarios += data["n_scenarios"]
        for failure in data["failures"]:
            bad_tokens.add(failure["token"])
        per_log_fail.update(data["per_log_fail"])
        per_log_total.update(data["per_log_total"])
        kinds.update(data["kinds"])

    manifest = json.loads(MANIFEST.read_text())
    frames_per_log = manifest["frames_per_log"]

    # 逐 log 分类：全崩 / 部分崩 / 干净
    all_bad_logs = [log for log in per_log_total if per_log_fail.get(log, 0) == per_log_total[log]]
    partial_logs = [
        log for log in per_log_total
        if 0 < per_log_fail.get(log, 0) < per_log_total[log]
    ]
    clean_logs = [log for log in per_log_total if per_log_fail.get(log, 0) == 0]

    logger.info("===== L3 坏场景扫描汇总 =====")
    logger.info("  试算 %d 个场景（清单 %d），失败 %d = %.2f%%",
                n_scenarios, sum(len(v) for v in manifest["tokens"].values()),
                len(bad_tokens), 100.0 * len(bad_tokens) / max(n_scenarios, 1))
    logger.info("  异常类型：")
    for kind, count in kinds.most_common(5):
        logger.info("    %6d× %s", count, kind[:120])
    logger.info("  逐 log：全崩 %d｜部分崩 %d｜干净 %d（共 %d）",
                len(all_bad_logs), len(partial_logs), len(clean_logs), len(per_log_total))

    # 写出剔除后的清单：token 级剔除，log 若被剔空则整个移出
    filtered_tokens: Dict[str, List[str]] = {}
    for log, tokens in manifest["tokens"].items():
        kept = [t for t in tokens if t not in bad_tokens]
        if kept:
            filtered_tokens[log] = kept

    # 分级重建：保持 L1⊂L2⊂L3 的嵌套与原来的 log 顺序
    new_tiers: Dict[str, Dict[str, object]] = {}
    for tier, info in manifest["tiers"].items():
        logs = [log for log in info["logs"] if log in filtered_tokens]
        new_tiers[tier] = {
            "n_log": len(logs),
            "n_sample": sum(len(filtered_tokens[log]) for log in logs),
            "logs": logs,
        }
        old = info["n_sample"]
        logger.info("  %s: log %d→%d｜sample %d→%d（−%.2f%%）",
                    tier, info["n_log"], len(logs), old, new_tiers[tier]["n_sample"],
                    100.0 * (old - new_tiers[tier]["n_sample"]) / max(old, 1))

    out = dict(manifest)
    out["tokens"] = filtered_tokens
    out["tiers"] = new_tiers
    out["note"] = (
        "已剔除 PlanTF feature builder 无法构建的场景（rig=False 的代价）。"
        "两条管线必须读本文件，谁都不许自己再抽一次。"
    )
    out["filter_provenance"] = {
        "reason": "remove_invalid_goals=False 下 route_roadblock_correction 抛 "
                  "AttributeError: 'NoneType' object has no attribute 'id'",
        "defined_by": "PlanTF 侧 NuplanFeatureBuilder.get_features_from_scenario 是否抛异常",
        "n_removed": len(bad_tokens),
        "removal_rate": len(bad_tokens) / max(n_scenarios, 1),
        "caveat": "MAGNET 侧这些场景并不崩；剔除是为保持两管线同清单，对我方是保守方向，写作须声明",
        "frames_per_log_original": frames_per_log,
        "note_uneven": "剔除后各 log 帧数不再恒为 100 —— 横轴报 log 数与样本数两条，"
                       "且『固定每 log 100 帧』的说法须改为『上限 100 帧』",
    }
    OUT_MANIFEST.write_text(json.dumps(out, indent=1))
    logger.info("  已写剔除后清单 %s", OUT_MANIFEST)

    OUT_REPORT.write_text(json.dumps(
        {
            "n_scenarios": n_scenarios,
            "n_bad": len(bad_tokens),
            "failure_rate": len(bad_tokens) / max(n_scenarios, 1),
            "kinds": dict(kinds),
            "n_logs_all_bad": len(all_bad_logs),
            "n_logs_partial": len(partial_logs),
            "n_logs_clean": len(clean_logs),
            "logs_all_bad": sorted(all_bad_logs),
            "logs_partial": {log: [per_log_fail[log], per_log_total[log]] for log in sorted(partial_logs)},
            "tiers_before": {k: v["n_sample"] for k, v in manifest["tiers"].items()},
            "tiers_after": {k: v["n_sample"] for k, v in new_tiers.items()},
        },
        indent=1,
    ))
    logger.info("  已写汇总 %s", OUT_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
