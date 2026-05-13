#!/bin/bash

# ==========================================
# WT-RDT "自底向上" 基础架构消融实验调度脚本
# 包含四大核心基线: Vanilla DT, Only Koopman, Only MWPA, Only ASTS
# ==========================================

# 捕获 Ctrl+C 信号，确保退出时杀掉所有后台进程，释放显存
trap "echo -e '\n⚠️ 收到终止信号，正在清理后台任务并释放显存...'; kill 0; exit 1" INT TERM

BASE_LOGDIR="Constructive_Experiments"
mkdir -p $BASE_LOGDIR

# ==========================================
# 1. 实验超参数配置
# ==========================================
SEEDS=(0 1)

# 四大极端分组
ABLATIONS=(
    "vanilla_dt"   # 三个模块全关 (绝对地板)
    "only_koopman" # 仅开 Koopman (核心物理骨架)
    "only_mwpa"    # 仅开 小波 (测试纯输入端过滤的局限性)
    "only_asts"    # 仅开 测谎 (测试无物理约束下的盲目干预)
)

# 9 个基础环境
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

# 2 种干扰类型 x 3 个干扰位置
MODES=("random" "adversarial")
TAGS=("obs" "act" "rew")

# ==========================================
# 2. 硬件与进度条初始化
# ==========================================
GPU_ID=0  # 目标 GPU ID

TOTAL_CASES=$((${#ABLATIONS[@]} * ${#ENVS[@]} * ${#MODES[@]} * ${#TAGS[@]}))
COMPLETED_CASES=0

function draw_progress_bar() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$((progress * 100 / total))
    local filled=$((progress * width / total))
    local empty=$((width - filled))

    printf "\r🌟 整体进度: ["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=0; i<empty; i++)); do printf " "; done
    printf "] %d%% (%d/%d Cases)" "$percent" "$progress" "$total"
}

echo "========================================================="
echo "🚀 开始运行 WT-RDT: 四大基础基线消融测试"
echo "组合维度: 4(基线) x 9(环境) x 2(干扰) x 3(位置) = 216 个独立实验条件"
echo "并发策略: 目标 GPU $GPU_ID | 每次下发 2 个 Seed 任务并行"
echo "---------------------------------------------------------"

draw_progress_bar 0 $TOTAL_CASES

# ==========================================
# 3. 核心并发调度循环
# ==========================================

for group in "${ABLATIONS[@]}"; do

    # 动态分配极端的消融参数开关
    if [ "$group" == "vanilla_dt" ]; then
        ARGS="--use_asts=False --use_mwpa=False --use_koopman=False"
    elif [ "$group" == "only_koopman" ]; then
        ARGS="--use_asts=False --use_mwpa=False --use_koopman=True"
    elif [ "$group" == "only_mwpa" ]; then
        ARGS="--use_asts=False --use_mwpa=True --use_koopman=False"
    elif [ "$group" == "only_asts" ]; then
        ARGS="--use_asts=True --use_mwpa=False --use_koopman=False"
    fi

    for env in "${ENVS[@]}"; do
        for mode in "${MODES[@]}"; do
            for tag in "${TAGS[@]}"; do

                # 启动当前 Case 的所有并行 Seed (0 和 1)
                for seed in "${SEEDS[@]}"; do

                    GROUP_NAME="${group}_${mode}_${tag}_seed_${seed}"
                    LOG_FILE="${BASE_LOGDIR}/${env}_${GROUP_NAME}.log"

                    CUDA_VISIBLE_DEVICES=$GPU_ID python algos/WT_RDT.py \
                        --seed $seed \
                        --env "$env" \
                        --corruption_mode "$mode" \
                        --corruption_tag "$tag" \
                        --corruption_rate 0.3 \
                        --eval_attack True \
                        --group "$GROUP_NAME" \
                        --logdir "$BASE_LOGDIR" \
                        --save_model True \
                        $ARGS > "$LOG_FILE" 2>&1 &

                    sleep 1 # 微小错峰启动，防止爆显存
                done

                # 精准挂起：等待当前环境下的 2 个 seed 跑完
                wait

                # 更新进度条
                COMPLETED_CASES=$((COMPLETED_CASES + 1))
                draw_progress_bar $COMPLETED_CASES $TOTAL_CASES

            done
        done
    done
done

echo -e "\n\n🎉 216 组基础消融任务全部执行完毕！"
echo "📊 开始自动提取数据并生成专属 CSV 表格..."

# ==========================================
# 4. 数据收割：提取并生成 Constructive_Ablation_Summary.csv
# ==========================================
cat << 'EOF' > aggregate_constructive.py
import os
import glob
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="Constructive_Experiments")
args = parser.parse_args()

output_csv = "Constructive_Ablation_Summary.csv"

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Ablation_Group', 'Env', 'Corruption_Mode', 'Corruption_Tag', 'Seed', 'Best_Score', 'Best_Epoch', 'Checkpoint_Dir'])

    search_pattern = os.path.join(args.base_dir, "**", "best_score.txt")
    for score_file in glob.glob(search_pattern, recursive=True):
        try:
            with open(score_file, 'r') as sf:
                score, epoch = sf.read().strip().split('_')
        except Exception:
            continue

        ckpt_dir = os.path.dirname(score_file).replace("\\", "/")

        # 提取分组信息
        group = "Unknown"
        for g in ["vanilla_dt", "only_koopman", "only_mwpa", "only_asts"]:
            if g in ckpt_dir:
                group = g
                break

        if group == "Unknown": continue

        env, mode, tag, seed = "Unknown", "Unknown", "Unknown", "Unknown"

        for e in ["door", "halfcheetah", "hammer", "hopper", "kitchen", "relocate", "walker2d"]:
            if e in ckpt_dir:
                env = e
                break

        if "random" in ckpt_dir: mode = "random"
        elif "adversarial" in ckpt_dir: mode = "adversarial"

        if "_obs_" in ckpt_dir or ckpt_dir.endswith("_obs"): tag = "obs"
        elif "_act_" in ckpt_dir or ckpt_dir.endswith("_act"): tag = "act"
        elif "_rew_" in ckpt_dir or ckpt_dir.endswith("_rew"): tag = "rew"

        if "seed_0" in ckpt_dir: seed = "0"
        elif "seed_1" in ckpt_dir: seed = "1"

        writer.writerow([group, env, mode, tag, seed, score, epoch, ckpt_dir])

print(f"✅ 数据提取完成！总共找到 {sum(1 for line in open(output_csv)) - 1} 条结果。")
print(f"📄 汇总结果已保存至: {output_csv}")
EOF

python aggregate_constructive.py --base_dir "$BASE_LOGDIR"
rm aggregate_constructive.py

echo "========================================================="
echo "🎯 大功告成！你可以使用 Constructive_Ablation_Summary.csv 来绘制性能爬坡图了！"
echo "========================================================="

