#!/bin/bash

# ==========================================
# WT-RDT 消融实验自动化调度脚本 (动态组合 & 进度条版)
# ==========================================

# 捕获 Ctrl+C 信号，确保退出时杀掉所有后台进程，释放显存
trap "echo -e '\n⚠️ 收到终止信号，正在清理后台任务并释放显存...'; kill 0; exit 1" INT TERM

BASE_LOGDIR="Ablation_Experiments"
mkdir -p $BASE_LOGDIR

# ==========================================
# 1. 实验超参数配置 (正交组合)
# ==========================================
SEEDS=(0 1)

# 三大消融组
ABLATIONS=(
    "w_o_asts"
    "w_o_mwpa"
    "w_o_koopman"
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
GPU_ID=0

# 每个 Case 指：[1个消融组] 下的 [1个基础环境 + 1种干扰 + 1个位置] 的双 Seed 训练
# 总 Cases = 3 * 9 * 2 * 3 = 162
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
echo "🚀 开始运行 WT-RDT 消融实验矩阵 (单 GPU 双 Seed 并发)"
echo "组合维度: 3(消融) x 9(环境) x 2(干扰) x 3(位置) = 162 个独立实验条件"
echo "并发策略: 目标 GPU $GPU_ID | 每次下发 2 个 Seed 任务"
echo "📂 日志目录: ${BASE_LOGDIR}/"
echo "---------------------------------------------------------"

draw_progress_bar 0 $TOTAL_CASES

# ==========================================
# 3. 核心并发调度循环
# ==========================================

for group in "${ABLATIONS[@]}"; do

    # 动态分配消融参数
    if [ "$group" == "w_o_asts" ]; then
        ARGS="--use_asts=False --use_mwpa=True --use_koopman=True"
    elif [ "$group" == "w_o_mwpa" ]; then
        ARGS="--use_asts=True --use_mwpa=False --use_koopman=True"
    elif [ "$group" == "w_o_koopman" ]; then
        ARGS="--use_asts=True --use_mwpa=True --use_koopman=False"
    fi

    for env in "${ENVS[@]}"; do
        for mode in "${MODES[@]}"; do
            for tag in "${TAGS[@]}"; do

                # 启动当前 Case 的所有并行 Seed (0 和 1)
                for seed in "${SEEDS[@]}"; do

                    GROUP_NAME="${group}_${mode}_${tag}_seed_${seed}"
                    LOG_FILE="${BASE_LOGDIR}/${env}_${GROUP_NAME}.log"

                    # 启动后台训练任务并打到唯一 GPU 上
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

                    sleep 1 # 微小错峰启动，防止同时加载模型瞬间爆显存
                done

                # 🌟 精准挂起：等待当前 (env+mode+tag+group) 下的 2 个 seed 跑完
                # 这保证了你的单张 GPU 永远只有 2 个任务在跑
                wait

                # 更新进度条
                COMPLETED_CASES=$((COMPLETED_CASES + 1))
                draw_progress_bar $COMPLETED_CASES $TOTAL_CASES

            done
        done
    done
done

echo -e "\n\n🎉 162 组消融任务全部执行完毕！"
echo "📊 开始自动提取最高分并生成带有多维信息的 CSV..."

# ==========================================
# 4. 数据收割：提取并生成终极 CSV 表格
# ==========================================
cat << 'EOF' > aggregate_ablations.py
import os
import glob
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="Ablation_Experiments")
args = parser.parse_args()

output_csv = "Ablation_Results_Summary.csv"

with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 表头加入了 Mode 和 Tag，完美适配你的 54 环境组合！
    writer.writerow(['Ablation_Group', 'Env', 'Corruption_Mode', 'Corruption_Tag', 'Seed', 'Best_Score', 'Best_Epoch', 'Checkpoint_Dir'])

    search_pattern = os.path.join(args.base_dir, "**", "best_score.txt")
    for score_file in glob.glob(search_pattern, recursive=True):
        try:
            with open(score_file, 'r') as sf:
                content = sf.read().strip()
                score, epoch = content.split('_')
        except Exception:
            continue

        # 解析路径结构获取信息
        # 你的 Logger 逻辑保存路径通常包含你传进去的 --group 参数
        # 对应上面脚本: GROUP_NAME = w_o_asts_random_obs_seed_0
        ckpt_dir = os.path.dirname(score_file)
        path_parts = ckpt_dir.replace("\\", "/").split('/')

        # 初始化默认值
        group, env, mode, tag, seed = "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"

        # 尝试从路径中智能提取信息 (利用关键字匹配)
        for part in path_parts:
            # 提取 Env
            for e in ["door", "halfcheetah", "hammer", "hopper", "kitchen", "relocate", "walker2d"]:
                if e in part:
                    env = part
                    break

            # 提取 Ablation Group
            for g in ["w_o_asts", "w_o_mwpa", "w_o_koopman"]:
                if g in part:
                    group = g
                    break

            # 提取 Mode
            if "random" in part: mode = "random"
            elif "adversarial" in part: mode = "adversarial"

            # 提取 Tag
            if "_obs_" in part or part.endswith("_obs"): tag = "obs"
            elif "_act_" in part or part.endswith("_act"): tag = "act"
            elif "_rew_" in part or part.endswith("_rew"): tag = "rew"

            # 提取 Seed
            if "seed_0" in part: seed = "0"
            elif "seed_1" in part: seed = "1"

        writer.writerow([group, env, mode, tag, seed, score, epoch, ckpt_dir])

print(f"✅ 数据提取完成！总共找到 {sum(1 for line in open(output_csv)) - 1} 条结果。")
print(f"📄 汇总结果已保存至: {output_csv}")
EOF

python aggregate_ablations.py --base_dir "$BASE_LOGDIR"
rm aggregate_ablations.py

echo "========================================================="
echo "🎯 大功告成！你可以用 Excel 打开 Ablation_Results_Summary.csv 使用透视表进行分析了！"
echo "========================================================="