#!/bin/bash

# 定义显卡数量和每张显卡上运行的程序数量
NUM_GPUS=8
NUM_TASKS_PER_GPU=16 # original: 16
TOTAL_TASKS=$((NUM_GPUS * NUM_TASKS_PER_GPU))

# 基础命令
BASE_CMD="python main_depth.py \
    --dataset-name ego4d_video \
    --image-path /fs/fast/u2023000902/data/ego4d/down_scale_720p \
    --output-path outputs/depth \
    --save-results \
    --depth-input-size 518 \
    --depth-grayscale \
    --depth-encoder vitl \
    --clip-length 8"

# 运行所有任务
for ((task_id=0; task_id<$TOTAL_TASKS; task_id++)); do
    # 计算分配到的显卡ID
    gpu_id=$((task_id / NUM_TASKS_PER_GPU))

    # 计算每个任务的标注文件路径
    anno_id=$((task_id % TOTAL_TASKS))  # 如果标注文件有不同的标号，可以根据需求修改

    # 设置显卡可见性并运行任务
    CUDA_VISIBLE_DEVICES=$gpu_id $BASE_CMD \
        --anno-path /fs/fast/u2023000902/annotations/egovlpv3/egonce++/chunks/egoclip_${anno_id}.csv &
done

# 等待所有后台任务完成
wait