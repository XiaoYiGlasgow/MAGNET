#!/usr/bin/env bash
# Pittsburgh in-distribution 官方闭环评测 —— GPU 队列排程（2026-07-28）
#
# 目的：用与 Boston-250 完全相同的 checkpoint 与协议，跑同城 held-out 闭环，
#       切开「低分来自跨城 shift」与「模型本身弱」两种解释。
#       该结论同时是候选 C（开环归因排序 vs official closed-loop 后果排序）
#       是否存在地板效应的前置判据。
#
# 排程策略：不抢占运行中的 2×2 dev。轮询三张卡，第一张连续 3 次判定为近乎独占
#          （used < EXCLUSIVE_MIB）的卡触发整批；GPU0 会在 dev 跑完（预计 07-31）后释放。
#
# 预计耗时：our planner GPU sequential ≈5 s/场景 → 280 场景 ≈23 min/模式；
#          两模式 ≈50 min；两条规则基线 CPU-ray 各数分钟。整批 ≈1.5–2 h。
#
# 启动方式（跨会话存活，勿用 pkill -f 杀）：
#   setsid bash /mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans/scripts/eval_closed_loop/queue_pittsburgh_indist.sh &
#
set -u

GNN=/mnt/local/Xiao_Yi/nuplan-devkit/GNN_Trans
HARNESS="$GNN/scripts/eval_closed_loop"
PY=/home/team/anaconda3/envs/nuplan/bin/python
RUN="$HARNESS/run_closed_loop.sh"

SCENARIO_SET="${SCENARIO_SET:-pittsburgh_val}"
N="${N:-20}"
GROUP="${GROUP:-$GNN/experiments/closed_loop/pittsburgh_indist_N${N}}"
LOG="$GROUP/queue.log"

EXCLUSIVE_MIB=4000     # 近乎独占判据（沿用阶段5 watcher v2；旧 util<15% 判据会误触发共卡）
CONSEC_NEEDED=3
POLL_S=120
MAX_WAIT_H=96

BARK=$(grep -oE 'https://api\.day\.app/[A-Za-z0-9_-]+' "$GNN/scratchpad/run_v9_speed_both.sh" | head -1)

mkdir -p "$GROUP"
log()  { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
push() { [ -n "$BARK" ] && curl -s --max-time 20 "$BARK/$1/$2" >/dev/null 2>&1; log "PUSH $1 :: $2"; }

log "================ pittsburgh in-dist closed-loop queue armed ================"
log "scenario_set=$SCENARIO_SET  N=$N  group=$GROUP"
log "等待任一 GPU 近乎独占（used < ${EXCLUSIVE_MIB} MiB，连续 ${CONSEC_NEEDED} 次）"
push "MAGNET-indist-queue-armed" "waiting-for-free-gpu--N${N}"

declare -A HITS=([0]=0 [1]=0 [2]=0)
DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
CHOSEN=""

while [ -z "$CHOSEN" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        log "超过 MAX_WAIT_H=${MAX_WAIT_H}h 仍无空闲卡，放弃"
        push "MAGNET-indist-queue-TIMEOUT" "no-free-gpu-in-${MAX_WAIT_H}h"
        exit 1
    fi
    while IFS=, read -r idx used; do
        idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' MiB')
        if [ "$used" -lt "$EXCLUSIVE_MIB" ]; then
            HITS[$idx]=$(( ${HITS[$idx]} + 1 ))
        else
            HITS[$idx]=0
        fi
        if [ "${HITS[$idx]}" -ge "$CONSEC_NEEDED" ]; then CHOSEN="$idx"; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)
    [ -z "$CHOSEN" ] && sleep "$POLL_S"
done

log "GPU${CHOSEN} 判定为近乎独占，启动整批"
push "MAGNET-indist-eval-START" "gpu${CHOSEN}--${SCENARIO_SET}--N${N}"
BATCH_START=$(date +%s)

# our planner 先跑（占卡），规则基线随后 CPU-ray
for spec in "gnn_trans nonreactive" "gnn_trans reactive" \
            "idm nonreactive" "idm reactive" \
            "log_future nonreactive" "log_future reactive"; do
    set -- $spec
    planner="$1"; mode="$2"
    log "---- run: ${planner} / ${mode} ----"
    bash "$RUN" --scenario-set "$SCENARIO_SET" --planner "$planner" --mode "$mode" \
                --n "$N" --device cuda --cuda "$CHOSEN" --group "$GROUP" >>"$LOG" 2>&1
    rc=$?
    log "---- rc=${rc}: ${planner}/${mode} ----"
    [ "$rc" -ne 0 ] && push "MAGNET-indist-eval-RUNFAIL" "${planner}-${mode}-rc${rc}"
done

MIN=$(( ( $(date +%s) - BATCH_START ) / 60 ))
log "整批结束，用时 ${MIN} min，开始提取分数"
$PY "$HARNESS/extract_scores.py" "$GROUP" --csv "$GROUP/summary.csv" 2>&1 | tee -a "$LOG"

SUMMARY=$($PY - "$GROUP/summary.csv" <<'PYEOF' 2>/dev/null || echo "no-summary"
import sys, pandas as pd
df = pd.read_csv(sys.argv[1]).set_index("tag")
print(";".join(f"{t.split('_')[-3]}-{t.split('_')[-2]}:{v:.3f}"
                for t, v in df["final_score"].items()))
PYEOF
)
push "MAGNET-indist-eval-DONE" "${MIN}min--${SUMMARY}"
log "================ done ================"
