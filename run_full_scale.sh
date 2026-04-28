#!/bin/bash

# ==========================================
# WT_RDT 全规模实验自动化调度脚本 (并行加速版)
# ==========================================

BASE_LOGDIR="Full_Scale_Experiments"
mkdir -p $BASE_LOGDIR

SEEDS=(0 1)
# 如果你有 3 张显卡，可以解开下面这行的注释，实现 1 个 Seed 占 1 张卡
# GPUS=(0 1 2)

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

echo "🚀 开始执行并行训练任务..."
echo "📂 所有结果将独立保存在目录: ${BASE_LOGDIR}/"

for env in "${ENVS[@]}"; do
    for mode in "${MODES[@]}"; do
        for tag in "${TAGS[@]}"; do

            echo "========================================================="
            echo "▶ 正在 [并行] 启动: Env=$env | Mode=$mode | Tag=$tag"

            for i in "${!SEEDS[@]}"; do
                seed=${SEEDS[$i]}
                GROUP_NAME="seed_${seed}_${mode}_${tag}"

                # 如果你想指定单卡多进程，就不用管 CUDA_VISIBLE_DEVICES
                # 如果你想多卡并行（比如 seed0 跑卡0，seed1 跑卡1），加上 CUDA_VISIBLE_DEVICES=${GPUS[$i]}

                # 在命令末尾加上 '&' 让其在后台并行运行
                python WT_RDT.py \
                    --seed $seed \
                    --use_mwpa True\
                    --use_koopman True\
                    --use_asts True\
                    --env $env \
                    --corruption_mode $mode \
                    --corruption_tag $tag \
                    --logdir $BASE_LOGDIR \
                    --group $GROUP_NAME \
                    --save_model True > "${BASE_LOGDIR}/${GROUP_NAME}_${env}.log" 2>&1 &

                echo "  ↳ 启动了 Seed=$seed (进程已放入后台)"
                sleep 2 # 错开几秒启动，避免同时去读写 dataset 或者瞬间显存尖峰
            done

            # 🌟 关键指令：等待后台的 3 个 Seed 全部跑完，再进入下一个 for 循环
            wait

            echo "✅ 完成: Env=$env | Mode=$mode | Tag=$tag (3 个 Seeds 均已结束)"
        done
    done
done

echo "训练任务全部并行执行完毕！"
echo "📊 开始自动汇总最高分并收集 .pth 权重..."

python aggregate_results.py --base_dir $BASE_LOGDIR