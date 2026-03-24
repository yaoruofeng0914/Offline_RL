#!/bin/bash

HDF5_DIR="$HOME/Offline_RL/original"
OUTPUT_DIR="$HOME/Offline_RL/datasets"

mkdir -p "$OUTPUT_DIR"

declare -A RATIOS
RATIOS["door-expert-v0"]=0.01
RATIOS["halfcheetah-medium-replay-v2"]=0.1
RATIOS["hammer-expert-v0"]=0.01
RATIOS["hopper-medium-replay-v2"]=0.1
RATIOS["kitchen-complete-v0"]=1
RATIOS["kitchen-mixed-v0"]=1
RATIOS["kitchen-partial-v0"]=1
RATIOS["relocate-expert-v0"]=0.01
RATIOS["walker2d-medium-replay-v2"]=0.1

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

for ENV_NAME in "${ENVS[@]}"; do
    RATIO=${RATIOS[$ENV_NAME]}
    echo "Generating dataset for $ENV_NAME with ratio $RATIO ..."

    python utils/ratio_dataset.py \
        --env_name "$ENV_NAME" \
        --ratio "$RATIO" \
        --h5path "$HDF5_DIR" \
        --output_path "$OUTPUT_DIR"
done

echo "All datasets generated!"