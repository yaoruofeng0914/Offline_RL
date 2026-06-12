#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export D4RL_SUPPRESS_IMPORT_ERROR=1
BASE_DIR=~/Offline_RL/checkpoint4baseline/RDT
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

# 接收传入的参数（允许为空）
TARGET_ENV=$1      # 例如: halfcheetah-expert-v2
TARGET_MODE=$2     # 例如: random 或 adversarial
TARGET_LOC=$3      # 例如: obs, act, rew
TARGET_SEED=$4     # 例如: 0 或 1

# 1. 统计总任务数
echo "正在统计任务总量..."
TOTAL_TASKS=0
for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")

    # 【新增】如果指定了环境且当前环境不匹配，则跳过
    [ -n "$TARGET_ENV" ] && [ "$env" != "$TARGET_ENV" ] && continue

    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"RDT"_${env}_}"
        [ "$remainder" == "$condition_name" ] && continue
        mode_tag=${remainder:0:3}

        if [ "$mode_tag" == "rnd" ] || [ "$mode_tag" == "random" ]; then
            ATTACK_MODE="random"
        elif [ "$mode_tag" == "adv" ] || [ "$mode_tag" == "adversarial" ]; then
            ATTACK_MODE="adversarial"
        else
            continue
        fi

        # 【新增】如果指定了攻击模式且当前模式不匹配，则跳过
        [ -n "$TARGET_MODE" ] && [ "$ATTACK_MODE" != "$TARGET_MODE" ] && continue

        remainder_1="${remainder#${mode_tag}_}"
        location=${remainder_1:0:3}

        # 【新增】如果指定了攻击位置且不匹配，则跳过
        [ -n "$TARGET_LOC" ] && [ "$location" != "$TARGET_LOC" ] && continue

        remainder_2="${remainder_1#${location}_}"
        [[ "${remainder_2:0:1}" == "0" ]] && SEED="0" || SEED="1"

        # 【新增】如果指定了 Seed 且不匹配，则跳过
        [ -n "$TARGET_SEED" ] && [ "$SEED" != "$TARGET_SEED" ] && continue

        TOTAL_TASKS=$((TOTAL_TASKS+1))
    done
done

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "没有找到匹配的任务，请检查你输入的过滤参数是否正确！"
    exit 0
fi

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
        remainder="${condition_name#$"RDT"_${env}_}"
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
        LOG_FILE="${LOG_DIR}/RDT_${env}_${ATTACK_MODE}_${location}_seed${SEED}.log"

        ((CURRENT_TASK++))
        PERCENT=$(awk "BEGIN {printf \"%.2f\", $CURRENT_TASK*100/$TOTAL_TASKS}")

        printf "\r进度: [%-50s] %s%% (%d/%d) | 当前: %s / %s / %s / seed=%s" \
            "$(printf '#%.0s' $(seq 1 $(($CURRENT_TASK*50/$TOTAL_TASKS))))" \
            "$PERCENT" "$CURRENT_TASK" "$TOTAL_TASKS" \
            "$env" "$ATTACK_MODE" "$location" "$SEED"

        # 执行评估
        python -m "algos.RDT" \
            --test_time $(date +"%Y%m%d_%H%M") \
            --env "$env" \
            --eval_only True \
            --eval_attack True \
            --corruption_mode "$ATTACK_MODE" \
            --corruption_tag "$location" \
            --test_attack_mode "nsaop" \
            --seed "$SEED" \
            --checkpoint_dir "$condition_dir" > "$LOG_FILE" 2>&1
    done
done

echo -e "\n----------------------------------------------------"
echo "✅ 所有任务已完成！详细日志保存在: $LOG_DIR"

# CSV 提取逻辑保持不变...
# (你可以直接保留你原本的 CSV 收集逻辑，这里为了简洁省略)