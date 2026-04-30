#!/bin/bash

# ==========================================
# WT_RDT 全规模实验自动化调度脚本 (带进度条版)
# ==========================================

BASE_LOGDIR="Full_Scale_Experiments"
mkdir -p $BASE_LOGDIR

SEEDS=(0 1)
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

# --- 进度计算初始化 ---
TOTAL_CASES=$((${#ENVS[@]} * ${#MODES[@]} * ${#TAGS[@]}))
COMPLETED_CASES=0

function draw_progress_bar() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$((progress * 100 / total))
    local filled=$((progress * width / total))
    local empty=$((width - filled))

    printf "\r进度: ["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf " "; done
    printf "] %d%% (%d/%d)" "$percent" "$progress" "$total"
}

echo "🚀 开始执行并行训练任务..."
echo "📂 所有结果将独立保存在目录: ${BASE_LOGDIR}/"
echo "---------------------------------------------------------"

for env in "${ENVS[@]}"; do
    for mode in "${MODES[@]}"; do
        for tag in "${TAGS[@]}"; do

            # 启动当前 Case 的所有并行 Seed
            for i in "${!SEEDS[@]}"; do
                seed=${SEEDS[$i]}
                GROUP_NAME="seed_${seed}_${mode}_${tag}"

                python algos/WT_RDT.py  \
                    --seed $seed \
                    --use_mwpa True \
                    --use_koopman True \
                    --use_asts True \
                    --env "$env" \
                    --corruption_mode "$mode" \
                    --corruption_tag "$tag" \
                    --logdir "$BASE_LOGDIR" \
                    --group "$GROUP_NAME" \
                    --save_model True > "${BASE_LOGDIR}/${GROUP_NAME}_${env}.log" 2>&1 &

                sleep 1 # 错峰启动
            done

            # 等待当前环境的 Seeds 全部结束
            wait

            # 更新进度
            COMPLETED_CASES=$((COMPLETED_CASES + 1))
            draw_progress_bar $COMPLETED_CASES $TOTAL_CASES
        done
    done
done

echo -e "\n\n🎉 训练任务全部并行执行完毕！"
echo "📊 开始自动汇总最高分并收集 .pth 权重..."

python aggregate_results.py --base_dir "$BASE_LOGDIR"