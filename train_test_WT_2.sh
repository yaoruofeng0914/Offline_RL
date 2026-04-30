#!/bin/bash

# =================================================================
# WT-RDT 消融实验：仅开启第一个开关 (MWPA)，关闭 Koopman 及其他
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
TAGS=("rew")
#"obs" "act"

# 🌟 核心参数设置：只开 MWPA，不开 Koopman 和 ASTS
PARAMS="--use_mwpa True --use_koopman True --use_asts True"

mkdir -p wt_only_logs

MAX_PARALLEL=3
current_jobs=0

echo "🔥 启动 WT-RDT (小波单模块) 专项评测..."

for ENV in "${ENVS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for TAG in "${TAGS[@]}"; do

            # 构造唯一日志名
            LOG_FILE="wt_only_logs/${ENV}_${MODE}_${TAG}.txt"

            # 启动后台进程
            python algos/WT_RDT.py \
                --env "$ENV" \
                --seed 0 \
                --corruption_mode "$MODE" \
                --corruption_tag "$TAG" \
                --corruption_rate 0.3 \
                --eval_attack True \
                --use_wandb 0 \
                $PARAMS > "$LOG_FILE" 2>&1 &

            echo "🟢 已启动: $ENV | $MODE | $TAG"

            let current_jobs++

            # 并发控制
            if [ "$current_jobs" -ge "$MAX_PARALLEL" ]; then
                wait
                current_jobs=0
                echo "⏳ 批次完成，准备下一组..."
                sleep 5
            fi

            sleep 2
        done
    done
done

wait
echo "任务已全部跑完！"