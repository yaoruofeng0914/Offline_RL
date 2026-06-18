#!/bin/bash
ENVS=(
#"door-expert-v0"
#"halfcheetah-medium-replay-v2"
#"hammer-expert-v0"
#"hopper-medium-replay-v2"
#"kitchen-complete-v0"
#"kitchen-mixed-v0"
#"kitchen-partial-v0"
"relocate-expert-v0"
#"walker2d-medium-replay-v2"
)
MODES=(
"random"
#"adversarial"
)

TAGS=(
#"obs"
#"act"
"rew"
)

SEEDS=(
0
#1
)

LOG_DIR="full_logs_udt"          # 改为 UDT 日志目录
mkdir -p "$LOG_DIR"

MAX_PARALLEL=3
current_batch=0
completed_jobs=0

total_jobs=$(( ${#ENVS[@]} * ${#MODES[@]} * ${#TAGS[@]} * ${#SEEDS[@]} ))

echo "日志目录: $LOG_DIR"
echo "计划运行总任务数: $total_jobs"
echo ""

if [ "$total_jobs" -eq 0 ]; then
    echo "没有需要运行的任务，请检查数组配置。"
    exit 0
fi

draw_progress_bar() {
    local _progress=$1
    local _total=$2
    local _percent=$(( ${_progress} * 100 / ${_total} ))
    local _filled=$(( ${_progress} * 40 / ${_total} ))
    local _empty=$(( 40 - ${_filled} ))
    local _bar=$(printf "%${_filled}s" | tr ' ' '█')
    local _space=$(printf "%${_empty}s" | tr ' ' ' ')
    printf "\r[${_bar}${_space}] ${_percent}%% (${_progress}/${_total}) 正在运行中..."
}

draw_progress_bar $completed_jobs $total_jobs

for ENV in "${ENVS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for TAG in "${TAGS[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                LOG_FILE="${LOG_DIR}/${ENV}_${MODE}_${TAG}_${SEED}.txt"

                # 关键：添加 --test_attack_mode nsaop，启用训练时 NSAOP 评估
                python algos/UDT.py \
                    --env "$ENV" \
                    --seed "$SEED" \
                    --corruption_mode "$MODE" \
                    --corruption_tag "$TAG" \
                    --corruption_rate 0.3 \
                    --eval_attack True \
                    --test_attack_mode "nsaop" \
                    --use_wandb 0 \
                    --save_model True \
                    --use_udt True \
                    > "$LOG_FILE" 2>&1 &

                let current_batch++

                if [ "$current_batch" -ge "$MAX_PARALLEL" ]; then
                    wait
                    completed_jobs=$((completed_jobs + current_batch))
                    current_batch=0
                    draw_progress_bar $completed_jobs $total_jobs
                    sleep 2
                fi

            done
        done
    done
done
if [ "$current_batch" -gt 0 ]; then
    wait
    completed_jobs=$((completed_jobs + current_batch))
    draw_progress_bar $completed_jobs $total_jobs
fi

echo -e "\n\n$total_jobs 项训练任务已全部执行完毕！"

echo "正在自动提取各 Case 的最高分..."
SUMMARY_FILE="UDT_Summary_Scores.csv"   # 改为 UDT 汇总文件

# 表头列名改为 UDT
echo "Environment,Seed,Noise_Type,Attack_Type,UDT" > $SUMMARY_FILE

for ENV in "${ENVS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for TAG in "${TAGS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                LOG_FILE="${LOG_DIR}/${ENV}_${MODE}_${TAG}_${SEED}.txt"

                if [ -f "$LOG_FILE" ]; then
                    RUN_DIR=$(grep "Logging to" "$LOG_FILE" | awk '{print $3}')
                    BEST_SCORE="NaN"
                    if [ -n "$RUN_DIR" ] && [ -f "${RUN_DIR}/best_score.txt" ]; then
                        SCORE_RAW=$(cat "${RUN_DIR}/best_score.txt")
                        BEST_SCORE=$(echo "$SCORE_RAW" | cut -d'_' -f1)
                    fi
                    CAP_MODE="$(tr '[:lower:]' '[:upper:]' <<< ${MODE:0:1})${MODE:1}"
                    echo "$ENV,$SEED,$CAP_MODE,$TAG,$BEST_SCORE" >> $SUMMARY_FILE
                fi
            done
        done
    done
done

echo "✅ 收集完成！成绩汇总已保存至: $SUMMARY_FILE"