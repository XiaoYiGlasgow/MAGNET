"""S2 —— 由共享 (log, token) 清单生成 PlanTF 侧的 scenario_filter yaml。

第二轮 scaling ladder 的硬约束（主控表 §0.0-c）：
**两条管线必须消费同一份清单，谁都不许自己再抽一次。**
不共享时实测同 57 个 log 上 token 交集仅 259/2850。

本脚本只做一件事：把 `s1_shared_token_manifest_seed42.json` 的某一级
（或它的一个子集）翻译成 PlanTF 的 hydra scenario_filter 配置。
filter flags 逐字段照抄清单里的 `filter_flags`，**不得在这里另设默认值** ——
Q59 已裁定 `remove_invalid_goals=False`，写死在清单里，本脚本只负责搬运。

用法：
    python s2_build_plantf_scenario_filter.py --tier L1 --name gnn_scaling_L1
    python s2_build_plantf_scenario_filter.py --tier L1 --n_logs 2 --name gnn_scaling_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GNN = Path("/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans")
MANIFEST = GNN / "experiments/scaling_ladder/s1_shared_token_manifest_seed42.json"
PLANTF_FILTER_DIR = Path("/mnt/local/Xiao_Yi/external/planTF/config/scenario_filter")
SPLITS = Path("/mnt/local/Xiao_Yi/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits")


def _yaml_bool(value: Any) -> str:
    """hydra 侧只认小写 true/false，Python 的 True/False 会被读成字符串。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_filter_yaml(
    manifest: Dict[str, Any],
    tier: str,
    n_logs: int | None,
    name: str,
    only_logs: List[str] | None = None,
) -> tuple[str, List[str], List[str]]:
    """生成 yaml 文本，并返回 (文本, 选中的 log 列表, 选中的 token 列表)。

    Args:
        manifest: 已加载的共享清单。
        tier: 阶梯级别（L1/L2/L3）。
        n_logs: 只取该级前 n 个 log（冒烟用）；None 表示整级。
        name: 生成的 filter 名（同时是文件名）。
        only_logs: 显式指定 log 名（冒烟时用来挑已知干净的 log），优先于 n_logs。

    Returns:
        (yaml 文本, log 名列表, token 列表)。

    Raises:
        KeyError: tier 不在清单里。
        ValueError: only_logs 里有不在该级的 log。
    """
    tier_info = manifest["tiers"][tier]
    logs: List[str] = list(tier_info["logs"])
    if only_logs:
        unknown = [log for log in only_logs if log not in logs]
        if unknown:
            raise ValueError(f"这些 log 不在 {tier} 里：{unknown}")
        logs = list(only_logs)
    elif n_logs is not None:
        logs = logs[:n_logs]

    tokens: List[str] = []
    for log in logs:
        tokens.extend(manifest["tokens"][log])

    flags = manifest["filter_flags"]
    # 清单里没有 scenario_tokens/log_names/map_names 这三项，它们正是本脚本要填的
    header = [
        "# 🔴 由 scripts/analysis/s2_build_plantf_scenario_filter.py 自动生成，勿手工编辑。",
        f"# 来源清单：{MANIFEST}",
        f"# 级别 {tier}｜log 数 {len(logs)}｜token 数 {len(tokens)}｜seed {manifest['seed']}"
        f"｜每 log {manifest['frames_per_log']} 帧",
        "# 两条管线（PlanTF / MAGNET）消费同一份清单，谁都不许自己再抽一次。",
        f"# filter flags 逐字段照抄清单，其中 remove_invalid_goals={flags['remove_invalid_goals']}"
        " 是 Q59 的裁定（对 PlanTF 发表 recipe 的一处偏离，论文须显式声明）。",
        "_target_: nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_: 'all'",
        "",
        f"scenario_types: {_yaml_bool(flags['scenario_types'])}",
        f"num_scenarios_per_type: {_yaml_bool(flags['num_scenarios_per_type'])}",
        f"limit_total_scenarios: {_yaml_bool(flags['limit_total_scenarios'])}",
        f"expand_scenarios: {_yaml_bool(flags['expand_scenarios'])}",
        f"remove_invalid_goals: {_yaml_bool(flags['remove_invalid_goals'])}",
        f"shuffle: {_yaml_bool(flags['shuffle'])}",
        "",
        "map_names: null",
        "timestamp_threshold_s: null",
        "ego_displacement_minimum_m: null",
        "ego_start_speed_threshold: null",
        "ego_stop_speed_threshold: null",
        "speed_noise_tolerance: null",
        "",
        "log_names:",
    ]
    body = [f"  - {log}" for log in logs]
    body.append("")
    body.append("scenario_tokens:")
    # 🔴 token 必须加引号：它是 16 位十六进制串，裸写时 yaml 会把
    # `0150f43e23bb59c6` 这类含 e 的读成科学计数法 float、纯数字串读成 int，
    # devkit 的 build_scenario_filter 会当场抛
    # "Expected all scenario tokens to be 16-character strings"（08-02 冒烟实测）。
    body.extend(f'  - "{token}"' for token in tokens)
    body.append("")

    return "\n".join(header + body), logs, tokens


def main() -> None:
    """入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="L1", choices=["L1", "L2", "L3"])
    parser.add_argument("--n_logs", type=int, default=None, help="只取前 n 个 log（冒烟用）")
    parser.add_argument("--name", required=True, help="生成的 filter 名（不含 .yaml）")
    parser.add_argument("--out_dir", type=Path, default=PLANTF_FILTER_DIR)
    parser.add_argument("--dump_tokens", type=Path, default=None, help="另存一份 token 清单供核对")
    parser.add_argument("--logs", nargs="+", default=None,
                        help="显式指定 log 名（冒烟挑已知干净的 log 用），优先于 --n_logs")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="共享清单路径；剔除坏场景后要指向 *_filtered.json")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    text, logs, tokens = build_filter_yaml(
        manifest, args.tier, args.n_logs, args.name, only_logs=args.logs
    )

    out_path = args.out_dir / f"{args.name}.yaml"
    out_path.write_text(text)
    logger.info("已写 %s（%.1f MB）", out_path, out_path.stat().st_size / 1e6)
    logger.info("  级别 %s｜log %d 个｜token %d 个", args.tier, len(logs), len(tokens))

    # 城市构成（两城裁定 Q58 之后，每次都报一遍，避免又一次「以为只有一城」）
    boston = {p.stem for p in (SPLITS / "train_boston").glob("*.db")}
    pitt = {p.stem for p in (SPLITS / "train_pittsburgh").glob("*.db")}
    n_bos = sum(1 for log in logs if log in boston)
    n_pit = sum(1 for log in logs if log in pitt)
    n_missing = len(logs) - n_bos - n_pit
    logger.info("  城市构成：boston %d｜pittsburgh %d｜两城都找不到 %d", n_bos, n_pit, n_missing)
    if n_missing:
        raise RuntimeError(f"{n_missing} 个 log 在本地两城 split 里都找不到，先查清单再跑")

    if args.dump_tokens is not None:
        args.dump_tokens.write_text(json.dumps({"logs": logs, "tokens": tokens}, indent=1))
        logger.info("  token 清单另存至 %s", args.dump_tokens)


if __name__ == "__main__":
    main()
