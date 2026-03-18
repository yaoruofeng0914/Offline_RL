#!/bin/bash

group="RDT"
eval_id="attack"
envs=("halfcheetah-medium-replay-v2" "walker2d-medium-replay-v2" "hopper-medium-replay-v2" "kitchen-complete-v0" "kitchen-partial-v0" "kitchen-mixed-v0" "door-expert-v0" "hammer-expert-v0" "relocate-expert-v0")
attack_tag="act"

for env in "${envs[@]}"; do

    env_dir="results/$group/$env"
    if [ ! -d "$env_dir" ]; then
        echo "警告: 目录不存在，跳过: $env_dir"
        continue
    fi

    checkpoint_dirs=()
    while IFS= read -r line; do
        checkpoint_dirs+=("$line")
    done < <(find "$env_dir" -mindepth 1 -maxdepth 1 -type d -name "*$attack_tag*" -printf "%f\n" | sort -V)
    
    count=${#checkpoint_dirs[@]}
    echo "找到的一级子文件夹个数: $count"
    
    for checkpoint_dir in "${checkpoint_dirs[@]}"; do
    
        echo "--------------------------------------------------"
        echo "Env=$env, Checkpoint=$checkpoint_dir, attack_tag=$attack_tag"
        echo "--------------------------------------------------"

        python algos/RDT.py \
            --group "$group" \
            --eval_id "$eval_id" \
            --env "$env" \
            --checkpoint_dir "$checkpoint_dir" \
            --eval_only true \
            --eval_attack true
    
    done
                
done
