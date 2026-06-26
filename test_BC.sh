#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export D4RL_SUPPRESS_IMPORT_ERROR=1
BASE_DIR=~/Offline_RL/checkpoint4baseline/BC
LOG_DIR="eval_logs"
SUMMARY_FILE="BC_Summary_Scores.csv"   # 汇总文件

mkdir -p "$LOG_DIR"

# 接收可选参数
TARGET_ENV=$1
TARGET_MODE=$2
TARGET_LOC=$3
TARGET_SEED=$4

# 写入 CSV 表头
echo "Environment,Seed,Noise_Type,Attack_Type,BC" > "$SUMMARY_FILE"

# 1. 统计总任务数（应用所有过滤条件）
echo "正在统计任务总量..."
TOTAL_TASKS=0
for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    [ -n "$TARGET_ENV" ] && [ "$env" != "$TARGET_ENV" ] && continue

    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"BC"_${env}_}"
        [ "$remainder" == "$condition_name" ] && continue
        mode_tag=${remainder:0:3}

        # 判断攻击模式
        if [ "$mode_tag" == "rnd" ] || [ "$mode_tag" == "random" ]; then
            ATTACK_MODE="random"
        elif [ "$mode_tag" == "adv" ] || [ "$mode_tag" == "adversarial" ]; then
            ATTACK_MODE="adversarial"
        else
            continue
        fi
        [ -n "$TARGET_MODE" ] && [ "$ATTACK_MODE" != "$TARGET_MODE" ] && continue

        remainder_1="${remainder#${mode_tag}_}"
        location=${remainder_1:0:3}
        [ -n "$TARGET_LOC" ] && [ "$location" != "$TARGET_LOC" ] && continue

        remainder_2="${remainder_1#${location}_}"
        [[ "${remainder_2:0:1}" == "0" ]] && SEED="0" || SEED="1"
        [ -n "$TARGET_SEED" ] && [ "$SEED" != "$TARGET_SEED" ] && continue

        TOTAL_TASKS=$((TOTAL_TASKS+1))
    done
done

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "没有找到匹配的任务，请检查过滤参数！"
    exit 0
fi

# 2. 执行任务并显示进度，同时记录得分
CURRENT_TASK=0
echo "开始评估 | 匹配到的任务数: $TOTAL_TASKS"
echo "----------------------------------------------------"

for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    [ -n "$TARGET_ENV" ] && [ "$env" != "$TARGET_ENV" ] && continue

    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"BC"_${env}_}"
        [ "$remainder" == "$condition_name" ] && continue
        mode_tag=${remainder:0:3}

        if [ "$mode_tag" == "rnd" ] || [ "$mode_tag" == "random" ]; then
            ATTACK_MODE="random"
        elif [ "$mode_tag" == "adv" ] || [ "$mode_tag" == "adversarial" ]; then
            ATTACK_MODE="adversarial"
        else
            continue
        fi
        [ -n "$TARGET_MODE" ] && [ "$ATTACK_MODE" != "$TARGET_MODE" ] && continue

        remainder_1="${remainder#${mode_tag}_}"
        location=${remainder_1:0:3}
        [ -n "$TARGET_LOC" ] && [ "$location" != "$TARGET_LOC" ] && continue

        remainder_2="${remainder_1#${location}_}"
        [[ "${remainder_2:0:1}" == "0" ]] && SEED="0" || SEED="1"
        [ -n "$TARGET_SEED" ] && [ "$SEED" != "$TARGET_SEED" ] && continue

        # 独立日志文件
        LOG_FILE="${LOG_DIR}/BC_${env}_${ATTACK_MODE}_${location}_seed${SEED}.log"

        ((CURRENT_TASK++))
        PERCENT=$(awk "BEGIN {printf \"%.2f\", $CURRENT_TASK*100/$TOTAL_TASKS}")

        printf "\r进度: [%-50s] %s%% (%d/%d) | 当前: %s / %s / %s / seed=%s" \
            "$(printf '#%.0s' $(seq 1 $(($CURRENT_TASK*50/$TOTAL_TASKS))))" \
            "$PERCENT" "$CURRENT_TASK" "$TOTAL_TASKS" \
            "$env" "$ATTACK_MODE" "$location" "$SEED"

        # 执行评估
        python -m "algos.BC" \
            --test_time $(date +"%Y%m%d_%H%M") \
            --env "$env" \
            --eval_only True \
            --eval_attack True \
            --corruption_mode "$ATTACK_MODE" \
            --corruption_tag "$location" \
            --test_attack_mode "nsaop" \
            --seed "$SEED" \
            --checkpoint_dir "$condition_dir" > "$LOG_FILE" 2>&1

        # 提取最高分
        BEST_SCORE="NaN"
        RUN_DIR=$(grep "Logging to" "$LOG_FILE" | awk '{print $NF}')
        if [ -n "$RUN_DIR" ] && [ -f "${RUN_DIR}/best_score.txt" ]; then
            SCORE_RAW=$(cat "${RUN_DIR}/best_score.txt")
            BEST_SCORE=$(echo "$SCORE_RAW" | cut -d'_' -f1)
        fi

        # 写入 CSV
        echo "$env,$SEED,$ATTACK_MODE,$location,$BEST_SCORE" >> "$SUMMARY_FILE"
    done
done

echo -e "\n----------------------------------------------------"
echo "✅ 所有任务已完成！详细日志保存在: $LOG_DIR"
echo "✅ 成绩汇总已保存至: $SUMMARY_FILE"