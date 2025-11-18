#!/bin/bash
# 配置1：稳健型baseline训练脚本
# 适合首次大规模训练，追求稳定性
# 预期性能：MAE < 0.4 eV

echo "=========================================="
echo "开始细粒度注意力模型训练 - Baseline配置"
echo "数据规模: 样本"
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
    --batch_size 32 \
    --epochs 300 \
    --learning_rate 5e-4 \
    --weight_decay 1e-5 \
    --warmup_steps 1500 \
    \
    --alignn_layers 4 \
    --gcn_layers 4 \
    --hidden_features 256 \
    \
    --use_fine_grained_attention 1 \
    --fine_grained_hidden_dim 256 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.15 \
    --fine_grained_use_projection 1 \
    \
    --output_dir ./output_training_baseline \
    --num_workers 4 \
    --random_seed 42

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_full_training_baseline/"
echo "=========================================="
