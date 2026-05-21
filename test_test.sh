#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export D4RL_SUPPRESS_IMPORT_ERROR=1  # 顺便屏蔽 D4RL 的报错刷屏
BASE_DIR=~/Offline_RL/results/
LOG_FILE="eval_progress.log"
> "$LOG_FILE" # 清空旧日志

# 1. 先统计总任务数（文件夹总数）
echo "正在统计任务总量..."
TOTAL_TASKS=$(find "$BASE_DIR" -maxdepth 3 -mindepth 3 -type d | wc -l)
CURRENT_TASK=0

echo "开始全自动评估 | 总任务数: $TOTAL_TASKS"
echo "----------------------------------------------------"

for algo_dir in "$BASE_DIR"/*; do
    [ ! -d "$algo_dir" ] && continue
    algo=$(basename "$algo_dir")

    for env_dir in "$algo_dir"/*; do
        [ ! -d "$env_dir" ] && continue
        env=$(basename "$env_dir")
        echo "{$env}"
        for condition_dir in "$env_dir"/*; do
            [ ! -d "$condition_dir" ] && continue
            condition_name=$(basename "$condition_dir")
            # --- 解析逻辑 ---
            remainder="${condition_name#${algo}_${env}_}"
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
                "$PERCENT" "$CURRENT_TASK" "$TOTAL_TASKS" "$algo" "$env"

            # --- 执行 Python 并隐藏详细输出到日志 ---
            python  -m "algos.${algo}" \
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
done

echo -e "\n----------------------------------------------------"
echo "✅ 所有任务已完成！详细报错请查看: $LOG_FILE"