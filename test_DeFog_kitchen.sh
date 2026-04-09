#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export D4RL_SUPPRESS_IMPORT_ERROR=1  # 顺便屏蔽 D4RL 的报错刷屏
BASE_DIR=~/Offline_RL/checkpoint4baseline/DeFog
LOG_FILE="eval_progress_DeFog_kitchen.log"
> "$LOG_FILE" # 清空旧日志

# 1. 先统计总任务数（只统计 kitchen 开头的环境下的条件文件夹）
echo "正在统计 kitchen 相关任务总量..."
TOTAL_TASKS=$(find "$BASE_DIR"/kitchen* -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
CURRENT_TASK=0
#安全检查：如果没找到，直接退出，防止后面的进度条除以0报错
if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "未找到任何以 kitchen 开头的环境目录，请检查路径！"
    exit 1
fi

echo "开始全自动评估 | 总任务数: $TOTAL_TASKS"
echo "----------------------------------------------------"
for env_dir in "$BASE_DIR"/kitchen*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    echo $env_dir
    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"DeFog"_${env}_}"
        [ "$remainder" == "$condition_name" ] && continue
        mode_tag=${remainder:0:3}
        remainder_1="${remainder#${mode_tag}_}"
        location=${remainder_1:0:3}
        remainder_2="${remainder_1#${location}_}"
        [[ "${remainder_2:0:1}" == "0" ]] && SEED="0" || SEED="1"
        if [ "$mode_tag" == "rnd" ] || [ "$mode_tag" == "random" ]; then
            ATTACK_MODE="random"
        elif [ "$mode_tag" == "adv" ] || [ "$mode_tag" == "adversarial" ]; then
            ATTACK_MODE="adversarial"
        else
            continue
        fi

            # --- 进度监控逻辑 ---
        ((CURRENT_TASK++))
            # 计算百分比
        PERCENT=$(awk "BEGIN {printf \"%.2f\", $CURRENT_TASK*100/$TOTAL_TASKS}")

            # 在同一行刷新显示进度
        printf "\r\c"
        printf "进度: [%-50s] %s%% (%d/%d) | 当前: %s/%s" \
            "$(printf '#%.0s' $(seq 1 $(($CURRENT_TASK*50/$TOTAL_TASKS))))" \
            "$PERCENT" "$CURRENT_TASK" "$TOTAL_TASKS" "DeFog" "$env"

        # --- 执行 Python 并隐藏详细输出到日志 ---
        python  -m "algos.DeFog" \
            --test_time $(date +"%Y%m%d_%H%M") \
            --env "$env" \
            --eval_only True \
            --eval_attack True \
            --corruption_mode "$ATTACK_MODE" \
            --corruption_tag "$location" \
            --seed "$SEED" \
            --checkpoint_dir "$condition_dir" >> "$LOG_FILE" 2>&1
    done
done

echo -e "\n----------------------------------------------------"
echo "✅ 所有任务已完成！详细报错请查看: $LOG_FILE"