#!/usr/bin/env bash
# 官方 nuPlan 闭环评测统一入口（重建版，2026-07-28）
#
# 背景：阶段5 的闭环 harness（scratchpad/eval_matrix.sh、gpu_watcher.sh、extract_scores.py、
#       boston_clean/、eval_full/）在 scratchpad 中已丢失，Boston-250 的 0.122/0.095 目前只有
#       阶段5 记录文本、没有可复现产物。本脚本是可审计的重建版，放在版本可追踪目录下。
#
# 设计约束（阶段15 §3.1 D0）：
#   - 全绝对路径；
#   - 不在运行时自改源文件，全部经 CLI 参数；
#   - 每个 run 落一份 provenance JSON（命令、git commit、ckpt SHA256、环境、场景集）。
#
# 用法：
#   bash run_closed_loop.sh --scenario-set pittsburgh_val --planner gnn_trans \
#        --mode nonreactive --n 20 --device cuda --cuda 0 --group <out_dir>
#
set -euo pipefail

REPO_ROOT="/mnt/local/Xiao_Yi/nuplan-devkit"
GNN_ROOT="${REPO_ROOT}/GNN_Trans"
HARNESS_DIR="${GNN_ROOT}/scripts/eval_closed_loop"
PYTHON="/home/team/anaconda3/envs/nuplan/bin/python"
# PlanTF 走独立 env（torch 2.8.0+cu128）与独立代码树；两者都不污染 `nuplan` env。
# ⚠️ 该 env 里 `import nuplan` 解析到的仍是**本仓的 devkit**（已实测），故 harness、
#    场景集、simulation config、metrics 与 MAGNET 侧逐项相同 —— 这正是 gate⑤ 要证的。
PLANTF_PYTHON="/home/team/anaconda3/envs/plantf/bin/python"
PLANTF_ROOT="/mnt/local/Xiao_Yi/external/planTF"
NUPLAN_SPLITS="${REPO_ROOT}/nuplan/dataset/nuplan-v1.1/splits"
CKPT="${GNN_ROOT}/experiments/magnet_imitation_e1/seed2_gpu2/models/best_model_default.pth"

SCENARIO_SET="pittsburgh_val"
PLANNER="gnn_trans"
MODE="nonreactive"
N=20
DEVICE="cuda"
CUDA_IDX="0"
GROUP=""
WORKER=""
# Q48（2026-08-01）：图拓扑必须能跟着 checkpoint 走。
# C2 三格用 `--dataset_config nearby_only` 训练，而 planner 的 adjacency 默认是 default（完全图）——
# 拿完全图部署等于把 C2 要验证的修复在部署侧原样撤销。planner 侧已加失配即报错的校验
# (`_check_adjacency_consistency`)，这里补上合法的覆盖通道。空 = 用 yaml 默认，行为不变。
ADJACENCY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario-set) SCENARIO_SET="$2"; shift 2 ;;
    --planner)      PLANNER="$2";      shift 2 ;;
    --mode)         MODE="$2";         shift 2 ;;
    --n)            N="$2";            shift 2 ;;
    --device)       DEVICE="$2";       shift 2 ;;
    --cuda)         CUDA_IDX="$2";     shift 2 ;;
    --group)        GROUP="$2";        shift 2 ;;
    --worker)       WORKER="$2";       shift 2 ;;
    --ckpt)         CKPT="$2";         shift 2 ;;
    --adjacency)    ADJACENCY="$2";    shift 2 ;;   # Q48：default / nearby_only / selective
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- 场景集 -> data_root + log_names -------------------------------------
case "${SCENARIO_SET}" in
  pittsburgh_heldout)
    # 阶段8 协议：train_pittsburgh 1560 DB − E1 用过的 600 log = 960 held-out，
    # 深筛后取 300 clean（db_error=0）。目录本身即过滤器，故不再叠加 log_names。
    # 重建脚本：build_pittsburgh_heldout.py --build --validate
    DATA_ROOT="${GNN_ROOT}/data/closed_loop_sets/pittsburgh_heldout_clean"
    LOG_FILE=""
    ;;
  pittsburgh_val)
    DATA_ROOT="${NUPLAN_SPLITS}/train_pittsburgh"
    LOG_FILE="${HARNESS_DIR}/logs_pittsburgh_val60.txt"
    ;;
  pittsburgh_test)
    # 注意：test 为 R4-7 锁定集，仅在方法冻结后一次性使用
    DATA_ROOT="${NUPLAN_SPLITS}/train_pittsburgh"
    LOG_FILE="${HARNESS_DIR}/logs_pittsburgh_test60.txt"
    ;;
  boston)
    # ⚠️ 阶段5 的 boston_clean/（300 个通过 deep-DB 校验的 symlink）已丢失。
    # 直接指向 train_boston 有踩坏 DB 的风险，重跑 Boston 前须先重建 clean 子集。
    DATA_ROOT="${NUPLAN_SPLITS}/train_boston"
    LOG_FILE=""
    ;;
  *) echo "unknown --scenario-set: ${SCENARIO_SET}" >&2; exit 2 ;;
esac

# ---- planner -> hydra override -------------------------------------------
case "${PLANNER}" in
  gnn_trans)   PLANNER_CFG="gnn_trans_planner" ;;
  idm)         PLANNER_CFG="idm_planner" ;;
  log_future)  PLANNER_CFG="log_future_planner" ;;
  plantf)
    # 外部 baseline（gate⑤）：换 env + 换 PYTHONPATH，其余一切保持不变。
    # --ckpt 未显式给出时用 null（随机初始化）—— 只用于接线冒烟，**不得当成结果**。
    PLANNER_CFG="plantf_planner"
    PYTHON="${PLANTF_PYTHON}"
    export PYTHONPATH="${PLANTF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    if [[ "${CKPT}" == "${GNN_ROOT}/experiments/magnet_imitation_e1/seed2_gpu2/models/best_model_default.pth" ]]; then
      CKPT=""   # 默认值是 MAGNET 的 ckpt，对 PlanTF 无意义
    fi
    ;;
  *) echo "unknown --planner: ${PLANNER}" >&2; exit 2 ;;
esac

case "${MODE}" in
  nonreactive) SIM_CFG="closed_loop_nonreactive_agents" ;;
  reactive)    SIM_CFG="closed_loop_reactive_agents" ;;
  *) echo "unknown --mode: ${MODE}" >&2; exit 2 ;;
esac

# 学习型 planner 用 sequential（约 5 s/场景）；规则基线用 CPU ray
if [[ -z "${WORKER}" ]]; then
  case "${PLANNER}" in
    gnn_trans|plantf) WORKER="sequential" ;;
    *)                WORKER="ray_distributed" ;;
  esac
fi

TAG="${SCENARIO_SET}_${PLANNER}_${MODE}_N${N}"
if [[ -z "${GROUP}" ]]; then
  GROUP="${GNN_ROOT}/experiments/closed_loop/$(date +%Y%m%d)_indist"
fi

# 阶段15 §11.6 新增硬约束：任何进入论文表格的闭环结果，产物目录必须落在 experiments/ 下。
# 阶段5/8 的闭环资产就是因为落在会话级 scratchpad/ 才全部丢失（Boston-250 的 0.122/0.095
# 至今只有记录文本、没有可复现产物）。这里用代码强制，不靠自觉。
case "${GROUP}" in
  */scratchpad/*|*/scratchpad)
    echo "REFUSED: --group 落在 scratchpad 下（${GROUP}）。" >&2
    echo "         闭环产物必须落 experiments/，否则会话结束即丢失（§11.6）。" >&2
    exit 3 ;;
  "${GNN_ROOT}/experiments"/*) : ;;
  *)
    echo "REFUSED: --group 必须在 ${GNN_ROOT}/experiments/ 下，当前 ${GROUP}" >&2
    exit 3 ;;
esac
mkdir -p "${GROUP}"

# ---- provenance ----------------------------------------------------------
# 🔴 Q24 修复（2026-07-31）：git 采集原先指 ${REPO_ROOT}（外层 nuplan-devkit），
#    而那是一个**零提交仓库**（devkit_patches/manifest.json 自己写明），于是 P4 恒报
#    `no-git` + 五百多个别人的脏文件 —— 是假警报，不是数据问题。
#    权威仓是 GNN_Trans：闭环用到的 planner / 模型定义 / 训练脚本全部在它里面，
#    devkit 侧的三个受管文件由 devkit_patches 保证同步（见下面的 PATCHES_* 采集）。
#    外层仓的状态仍然记录，但降级为 `devkit_repo` 字段，不再参与 reproducible 判定。
# 注意用 `rev-parse --verify`：零提交仓上裸 `rev-parse HEAD` 会把字面量 "HEAD" 打到 stdout
# 再返回非零，于是 `|| echo no-git` 得到 "HEAD\nno-git" 这种垃圾值。
GIT_COMMIT="$(cd "${GNN_ROOT}" && git rev-parse --verify HEAD 2>/dev/null || echo 'no-git')"
GIT_DIRTY="$(cd "${GNN_ROOT}" && git status --porcelain 2>/dev/null | wc -l)"
DEVKIT_COMMIT="$(cd "${REPO_ROOT}" && git rev-parse --verify HEAD 2>/dev/null || echo 'no-git')"
DEVKIT_DIRTY="$(cd "${REPO_ROOT}" && git status --porcelain 2>/dev/null | wc -l)"

# 🔴 Q24 修复（P8）：harness 此前从不写 `devkit_patches` 字段，而该信息是现成的。
set +e   # --check 失败必须记录进 provenance，而不是让整个 run 中断
PATCHES_OUT="$(cd "${GNN_ROOT}" && $PYTHON "${GNN_ROOT}/devkit_patches/sync_devkit_patches.py" --check 2>&1)"
PATCHES_RC=$?
set -e
PATCHES_SUMMARY="$(printf '%s' "${PATCHES_OUT}" | grep -oE '[0-9]+/[0-9]+ 同步' | tail -1)"
[[ -z "${PATCHES_SUMMARY}" ]] && PATCHES_SUMMARY="unknown"

if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
  CKPT_SHA="$(sha256sum "${CKPT}" | cut -d' ' -f1)"
else
  # PlanTF 冒烟阶段没有权重（官方 ckpt 全 404，自训未完成）—— 显式记 none，不伪造
  CKPT="${CKPT:-none}"
  CKPT_SHA="none"
fi

EXTRA_OVERRIDES=()
if [[ "${PLANNER}" == "gnn_trans" ]]; then
  EXTRA_OVERRIDES+=("planner.gnn_trans_planner.device=${DEVICE}")
  EXTRA_OVERRIDES+=("planner.gnn_trans_planner.checkpoint_path=${CKPT}")
  # 只在显式给了 --adjacency 时才覆盖，不给则沿用 yaml 默认 ⇒ 既有 run 的行为逐字段不变
  if [[ -n "${ADJACENCY}" ]]; then
    EXTRA_OVERRIDES+=("planner.gnn_trans_planner.adjacency_config=${ADJACENCY}")
  fi
fi
if [[ "${PLANNER}" == "plantf" ]]; then
  if [[ "${DEVICE}" == cuda* ]]; then
    EXTRA_OVERRIDES+=("planner.plantf_planner.use_gpu=true")
  else
    EXTRA_OVERRIDES+=("planner.plantf_planner.use_gpu=false")
  fi
  if [[ "${CKPT}" != "none" ]]; then
    EXTRA_OVERRIDES+=("planner.plantf_planner.planner_ckpt=${CKPT}")
  fi
fi
if [[ -n "${LOG_FILE}" ]]; then
  LOG_NAMES="$(paste -sd, "${LOG_FILE}")"
  EXTRA_OVERRIDES+=("scenario_filter.log_names=[${LOG_NAMES}]")
fi

HARNESS_SHA="$(sha256sum "${HARNESS_DIR}/run_closed_loop.sh" | cut -d' ' -f1)"

PROV="${GROUP}/${TAG}.provenance.json"
$PYTHON - "$PROV" "$TAG" "$SCENARIO_SET" "$DATA_ROOT" "${LOG_FILE:-none}" "$PLANNER_CFG" \
                  "$SIM_CFG" "$N" "$DEVICE" "$CUDA_IDX" "$WORKER" "$CKPT" "$CKPT_SHA" \
                  "$GIT_COMMIT" "$GIT_DIRTY" "$HARNESS_SHA" "$GNN_ROOT" \
                  "$DEVKIT_COMMIT" "$DEVKIT_DIRTY" "$PATCHES_SUMMARY" "$PATCHES_RC" \
                  "${ADJACENCY:-}" <<'PYEOF'
import datetime
import json
import os
import platform
import subprocess
import sys

(out, tag, sset, droot, logf, pcfg, scfg, n, dev, cidx, worker,
 ckpt, cksha, commit, dirty, harness_sha, gnn_root,
 devkit_commit, devkit_dirty, patches_summary, patches_rc, adjacency_cli) = sys.argv[1:23]

sys.path.insert(0, os.path.join(gnn_root, "scripts", "analysis"))
try:
    from result_manifest import SCHEMA_VERSION, schema_hash
    schema = {"version": SCHEMA_VERSION, "hash": schema_hash()}
except Exception as exc:                                   # noqa: BLE001
    schema = {"version": None, "hash": None, "error": str(exc)}

# checkpoint 训练时用的 route_mode 只存在于 sidecar 里。没有 sidecar 时 planner 按 "none"
# 处理（gnn_trans_planner.py:126-130）——那是**推断值不是记录值**，必须在清单里标明来源，
# 否则事后无法区分「训练时确实是 none」和「记录丢了」。
meta_path = os.path.join(os.path.dirname(ckpt), "train_meta.json")
if os.path.exists(meta_path):
    with open(meta_path, encoding="utf-8") as fh:
        train_meta = json.load(fh)
    route_mode = {"value": train_meta.get("route_mode"), "source": "train_meta.json"}
else:
    train_meta = None
    route_mode = {"value": "none", "source": "inferred (no train_meta.json sidecar)"}

try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:                                          # noqa: BLE001
    gpu = "n/a"

json.dump({
    "tag": tag, "utc": datetime.datetime.utcnow().isoformat() + "Z",
    "scenario_set": sset, "data_root": droot, "log_names_file": logf,
    "planner_cfg": pcfg, "simulation_cfg": scfg, "num_scenarios_per_type": int(n),
    "device": dev, "cuda_visible": cidx, "worker": worker,
    "checkpoint": ckpt, "checkpoint_sha256": cksha,
    "checkpoint_train_meta": train_meta, "route_mode": route_mode,
    # Q48：图拓扑必须可事后查。`value` 是 planner **实际生效**的 adjacency
    # （给了 --adjacency 就是它，否则是 yaml 默认 "default"）；
    # `checkpoint_dataset_config` 是 ckpt 训练时的拓扑，两者不等时 planner 会直接报错
    # （gnn_trans_planner._check_adjacency_consistency），所以这里记下来即可自证一致。
    "adjacency": {
        "value": adjacency_cli or "default",
        "source": "cli --adjacency" if adjacency_cli else "yaml default",
        "checkpoint_dataset_config": (train_meta or {}).get("dataset_config"),
    },
    "feature_schema": schema,
    "harness": {"script": "scripts/eval_closed_loop/run_closed_loop.sh", "sha256": harness_sha},
    # git_* 指**权威仓 GNN_Trans**（Q24 修复；此前误指零提交的外层 nuplan-devkit）
    "git_repo": "GNN_Trans",
    "git_commit": commit, "git_dirty_files": int(dirty),
    "reproducible_from_commit": int(dirty) == 0,
    # 外层 devkit 仓仅作记录，不参与 reproducible 判定（它是零提交仓）
    "devkit_repo": {"path": os.path.dirname(gnn_root.rstrip("/")),
                    "git_commit": devkit_commit, "git_dirty_files": int(devkit_dirty),
                    "note": "zero-commit vendor tree; managed files tracked via devkit_patches"},
    # P8：受管文件同步状态（`devkit_patches/sync_devkit_patches.py --check`）
    "devkit_patches": {"check": patches_summary, "returncode": int(patches_rc),
                       "in_sync": int(patches_rc) == 0},
    "python": platform.python_version(), "host": platform.node(), "gpu_snapshot": gpu,
}, open(out, "w"), indent=2, ensure_ascii=False)
print("provenance ->", out)
PYEOF

echo "=== ${TAG} ==="
echo "  data_root : ${DATA_ROOT}"
echo "  logs      : ${LOG_FILE:-<all>}"
echo "  planner   : ${PLANNER_CFG} (${DEVICE})  worker=${WORKER}"
echo "  group     : ${GROUP}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_IDX}"
set -x
$PYTHON "${REPO_ROOT}/nuplan/planning/script/run_simulation.py" \
  "+simulation=${SIM_CFG}" \
  "planner=${PLANNER_CFG}" \
  "scenario_builder=nuplan_mini" \
  "scenario_builder.data_root=${DATA_ROOT}" \
  "scenario_builder.db_files=null" \
  "scenario_filter=nuplan_challenge_scenarios" \
  "scenario_filter.num_scenarios_per_type=${N}" \
  "worker=${WORKER}" \
  "group=${GROUP}" \
  "experiment_name=${TAG}" \
  "${EXTRA_OVERRIDES[@]}" \
  2>&1 | tee "${GROUP}/${TAG}.log"
set +x
echo "DONE ${TAG}"
