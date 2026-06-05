#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export D4RL_SUPPRESS_IMPORT_ERROR=1
BASE_DIR=~/Offline_RL/checkpoint4baseline/RDT
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

# 1. 统计总任务数
echo "正在统计任务总量..."
TOTAL_TASKS=0
for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"RDT"_${env}_}"
        [ "$remainder" == "$condition_name" ] && continue
        mode_tag=${remainder:0:3}
        remainder_1="${remainder#${mode_tag}_}"
        location=${remainder_1:0:3}
        if [ "$mode_tag" == "rnd" ] || [ "$mode_tag" == "random" ]; then
            TOTAL_TASKS=$((TOTAL_TASKS+1))
        elif [ "$mode_tag" == "adv" ] || [ "$mode_tag" == "adversarial" ]; then
            TOTAL_TASKS=$((TOTAL_TASKS+1))
        fi
    done
done

CURRENT_TASK=0
echo "开始全自动评估 | 总任务数: $TOTAL_TASKS"
echo "----------------------------------------------------"

for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"RDT"_${env}_}"
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
            --test_attack_mode nsaop \
            --seed "$SEED" \
            --checkpoint_dir "$condition_dir" > "$LOG_FILE" 2>&1
    done
done

echo -e "\n----------------------------------------------------"
echo "✅ 所有任务已完成！详细日志保存在: $LOG_DIR"

# ==============================================================
# 自动提取最高分并生成 CSV（对齐第二个脚本的格式）
# ==============================================================
echo "正在自动提取各 Case 的最高分..."
SUMMARY_FILE="RDT_NSAOP_Summary_Scores.csv"
echo "Environment,Seed,Noise_Type,Attack_Type,RDT_NSAOP" > $SUMMARY_FILE

for log_path in "$LOG_DIR"/RDT_*.log; do
    [ ! -f "$log_path" ] && continue
    filename=$(basename "$log_path" .log)
    # 文件名格式: RDT_${env}_${ATTACK_MODE}_${location}_seed${SEED}
    # 例如: RDT_halfcheetah-medium-replay-v2_random_obs_seed0
    # 提取各部分
    IFS='_' read -ra parts <<< "$filename"
    # 最少有 5 段: RDT, env(可能含连字符), mode, tag, seedX
    # 环境名可能包含多个下划线（如 halfcheetah-medium-replay-v2），
    # 所以不能用固定位置，改用已知的模式: 第1段是RDT，最后两段是 tag 和 seedX，倒数第三段是 mode
    # 更稳健的方式：从右往左解析
    seed_str="${parts[-1]}"        # seed0 或 seed1
    SEED="${seed_str#seed}"
    attack_tag="${parts[-2]}"      # obs, act, rew
    attack_mode="${parts[-3]}"     # random 或 adversarial
    # 剩下的部分拼接成环境名（去掉第一个 RDT）
    env_part="${parts[@]:1:${#parts[@]}-4}"
    env_name=$(echo $env_part | tr ' ' '-')   # 重新用连字符拼接（但原本环境名里就有连字符和下划线混杂，这里可能导致错误）

done

SUMMARY_FILE="RDT_NSAOP_Summary_Scores.csv"
echo "Environment,Seed,Noise_Type,Attack_Type,RDT_NSAOP" > $SUMMARY_FILE

for env_dir in "$BASE_DIR"/*; do
    [ ! -d "$env_dir" ] && continue
    env=$(basename "$env_dir")
    for condition_dir in "$env_dir"/*; do
        [ ! -d "$condition_dir" ] && continue
        condition_name=$(basename "$condition_dir")
        remainder="${condition_name#$"RDT"_${env}_}"
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

        # 构建对应的日志文件名
        LOG_FILE="${LOG_DIR}/RDT_${env}_${ATTACK_MODE}_${location}_seed${SEED}.log"
        BEST_SCORE="NaN"

        if [ -f "$LOG_FILE" ]; then
            # 从日志中提取 "Logging to" 行，获取结果目录
            RUN_DIR=$(grep "Logging to" "$LOG_FILE" | tail -1 | awk '{print $3}')
            if [ -n "$RUN_DIR" ] && [ -f "${RUN_DIR}/best_score.txt" ]; then
                SCORE_RAW=$(cat "${RUN_DIR}/best_score.txt")
                BEST_SCORE=$(echo "$SCORE_RAW" | cut -d'_' -f1)
            fi
        fi

        # 格式化 Noise_Type 首字母大写
        CAP_MODE="$(tr '[:lower:]' '[:upper:]' <<< ${ATTACK_MODE:0:1})${ATTACK_MODE:1}"
        echo "$env,$SEED,$CAP_MODE,$location,$BEST_SCORE" >> $SUMMARY_FILE
    done
done

echo "✅ 收集完成！成绩汇总已保存至: $SUMMARY_FILE"