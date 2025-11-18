#!/bin/bash
# 配置3：快速实验型训练脚本
# 适合快速验证和迭代
# 预期性能：MAE < 0.5 eV

echo "=========================================="
echo "开始细粒度注意力模型训练 - 快速实验配置"
echo "数据规模: 18000样本"
echo "用途：快速验证代码和数据"
echo "=========================================="

python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 128 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --warmup_steps 2000 \
    \
    --alignn_layers 3 \
    --gcn_layers 3 \
    --hidden_features 128 \
    \
    --use_fine_grained_attention 1 \
    --fine_grained_hidden_dim 128 \
    --fine_grained_num_heads 4 \
    --fine_grained_dropout 0.1 \
    --fine_grained_use_projection 1 \
    \
    --output_dir ./output_full_training_fast \
    --num_workers 4 \
    --random_seed 42

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_full_training_fast/"
echo "=========================================="
