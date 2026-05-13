#!/bin/bash

# =================================================================
# V4 完全体 (Full) 专项评测：MWPA + Koopman + ASTS 全开
# 9 环境 × 2 模式 × 3 位置 = 54 任务
# =================================================================

ENVS=(
    "door-expert-v0"
    "halfcheetah-medium-replay-v2"
    "hammer-expert-v0"
    "hopper-medium-replay-v2"
    "kitchen-complete-v0"
    "kitchen-mixed-v0"
    "kitchen-partial-v0"
    "relocate-expert-v0"
    "walker2d-medium-replay-v2"
)

MODES=("random" "adversarial")
TAGS=("obs" "act" "rew")

# 核心参数设置：V4 完全体三大模块全开
PARAMS="--use_mwpa True --use_koopman True --use_asts True"

LOG_DIR="v4_full_logs"
mkdir -p "$LOG_DIR"

MAX_PARALLEL=5
current_batch=0
completed_jobs=0
total_jobs=54

echo "启动 V4 完全体对抗评测 (已强制开启 eval_attack)..."
echo "日志目录: $LOG_DIR"
echo ""

# 进度条渲染函数
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

# 初始进度条
draw_progress_bar $completed_jobs $total_jobs

for ENV in "${ENVS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for TAG in "${TAGS[@]}"; do

            LOG_FILE="${LOG_DIR}/${ENV}_${MODE}_${TAG}.txt"

            # 修复：恢复 --eval_attack True 开关，并双重确保 rate 传参
            python algos/WT_RDT.py \
                --env "$ENV" \
                --seed 0 \
                --corruption_mode "$MODE" \
                --corruption_tag "$TAG" \
                --corruption_rate 0.3 \
                --eval_attack True \
                --use_wandb 0 \
                --save_model True \
                $PARAMS > "$LOG_FILE" 2>&1 &

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

if [ "$current_batch" -gt 0 ]; then
    wait
    completed_jobs=$((completed_jobs + current_batch))
    draw_progress_bar $completed_jobs $total_jobs
fi

echo -e "\n\n54 项 V4 完全体任务已全部执行完毕。请检查 $LOG_DIR 查看详细输出。"