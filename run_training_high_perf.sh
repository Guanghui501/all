#!/bin/bash
# 配置2：高性能型训练脚本
# 适合GPU内存充足，追求最优指标
# 预期性能：MAE < 0.35 eV

echo "=========================================="
echo "开始细粒度注意力模型训练 - 高性能配置"
echo "数据规模: 18000样本"
echo "注意：需要约16GB GPU内存"
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
    --batch_size 16 \
    --epochs 400 \
    --learning_rate 3e-4 \
    --weight_decay 5e-6 \
    --warmup_steps 5000 \
    \
    --alignn_layers 6 \
    --gcn_layers 6 \
    --hidden_features 384 \
    \
    --use_fine_grained_attention 1 \
    --fine_grained_hidden_dim 384 \
    --fine_grained_num_heads 8 \
    --fine_grained_dropout 0.2 \
    --fine_grained_use_projection 1 \
    \
    --use_cross_modal 1 \
    --cross_modal_hidden_dim 384 \
    --cross_modal_num_heads 8 \
    --cross_modal_dropout 0.15 \
    \
    --output_dir ./output_full_training_high_perf \
    --num_workers 4 \
    --random_seed 42

echo "=========================================="
echo "训练完成！"
echo "模型保存在: ./output_full_training_high_perf/"
echo "=========================================="
