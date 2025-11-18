# 18000样本训练配置建议

## 配置1：稳健型（推荐作为baseline）⭐

适合：首次大规模训练，追求稳定性

```bash
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
    --warmup_steps 3000 \
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
    --output_dir ./output_full_training_v1 \
    --num_workers 4 \
    --random_seed 42
```

**预期性能**: MAE < 0.4 eV (相比测试MAE 0.82)

---

## 配置2：高性能型（追求最佳性能）⭐⭐

适合：GPU内存充足，追求最优指标

```bash
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
    --use_middle_fusion 0 \
    --use_cross_modal 1 \
    --cross_modal_hidden_dim 384 \
    --cross_modal_num_heads 8 \
    --cross_modal_dropout 0.15 \
    \
    --output_dir ./output_full_training_high_perf \
    --num_workers 4 \
    --random_seed 42
```

**预期性能**: MAE < 0.35 eV

**资源需求**: ~16GB GPU内存，训练时间更长

---

## 配置3：快速实验型（快速验证）

适合：快速迭代，验证想法

```bash
python train_with_cross_modal_attention.py \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset \
    --dataset jarvis \
    --property mbj_bandgap \
    \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    \
    --batch_size 64 \
    --epochs 200 \
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
```

**预期性能**: MAE < 0.5 eV

**资源需求**: 更少GPU内存和时间

---

## 参数说明

### 批次大小 (batch_size)
- **32**: 平衡性能和速度，推荐
- **16**: 更稳定的梯度，适合追求最优性能
- **64**: 训练速度快，但可能影响性能

### 学习率 (learning_rate)
- **5e-4 (0.0005)**: 稳健的选择，适合大多数情况
- **3e-4 (0.0003)**: 更保守，训练更稳定
- **1e-3 (0.001)**: 更激进，收敛更快但可能不稳定

### Warmup步数计算
```
每个epoch的步数 = 训练样本数 / batch_size
18000 * 0.8 = 14400训练样本

batch_size=32: 每epoch约450步
- warmup_steps=3000: 约前6-7个epoch做warmup
- warmup_steps=5000: 约前11个epoch做warmup

batch_size=16: 每epoch约900步
- warmup_steps=5000: 约前5-6个epoch做warmup
```

### 模型深度
- **layers=4**: 标准配置，适合大多数情况
- **layers=6**: 更强的表达能力，需要更多数据和训练时间
- **layers=3**: 快速实验，轻量级

### 隐藏层维度
- **256**: 标准配置，5.2M参数
- **384**: 高性能配置，约10M参数
- **128**: 快速实验，约1.5M参数

### Dropout率
- **0.1**: 轻度正则化
- **0.15**: 中度正则化（推荐）
- **0.2**: 强正则化，防止过拟合

---

## 训练监控建议

### 1. 早停策略
建议添加早停逻辑（如果脚本未实现）：
```python
patience = 30  # 30个epoch无改进则停止
min_delta = 0.001  # 最小改进阈值
```

### 2. 学习率调度
推荐策略：
- **ReduceLROnPlateau**: 验证集loss不降则减小lr
  - factor=0.5, patience=10, min_lr=1e-6
- **CosineAnnealingLR**: 余弦退火
  - T_max=epochs, eta_min=1e-6

### 3. 检查点保存
- 保存验证集MAE最低的模型
- 每10个epoch保存一次
- 保留最近5个checkpoint

### 4. TensorBoard监控
```bash
tensorboard --logdir=./output_full_training_v1/tensorboard
```

---

## 预期训练时间

假设单个epoch耗时（取决于硬件）：
- batch_size=32: 约3-5分钟/epoch
- batch_size=16: 约6-10分钟/epoch

**配置1 (300 epochs, bs=32)**:
- 总时间: 15-25小时

**配置2 (400 epochs, bs=16)**:
- 总时间: 40-65小时

**配置3 (200 epochs, bs=64)**:
- 总时间: 6-10小时

---

## 性能基准

基于1000样本50 epochs的测试结果 (MAE=0.82):

18000样本全量训练预期：
- **Baseline (配置1)**: MAE 0.35-0.45 eV
- **High-performance (配置2)**: MAE 0.30-0.40 eV
- **Fast (配置3)**: MAE 0.40-0.50 eV

JARVIS MBJ带隙数据集的SOTA (其他方法):
- ALIGNN: ~0.20 eV MAE
- CGCNN: ~0.35 eV MAE
- SchNet: ~0.30 eV MAE

**目标**: 细粒度注意力机制应该达到或超越CGCNN性能

---

## 推荐执行流程

### Step 1: 快速验证 (配置3)
先用配置3快速训练200轮，验证：
- 代码无bug
- 数据加载正常
- 训练曲线合理
- 预估最终性能

### Step 2: 稳健训练 (配置1) ⭐
使用配置1进行完整训练：
- 更稳定的超参数
- 合理的训练时间
- 作为baseline

### Step 3: 性能优化 (配置2)
如果配置1结果好，使用配置2冲刺最优性能：
- 更大模型
- 更多训练轮数
- 更精细的学习率

### Step 4: 可解释性分析
训练完成后，使用demo脚本进行详细分析：
```bash
# 分析多个样本
for i in {1..20}; do
    python demo_fine_grained_attention.py \
        --model_path output_full_training_v1/best_model.pt \
        --cif_path /path/to/cif/${i}.cif \
        --text "..." \
        --save_dir analysis_sample_${i}
done
```

---

## 常见问题

### Q1: GPU内存不足怎么办？
**方案1**: 减小batch_size到16或8
**方案2**: 减小hidden_features到128
**方案3**: 使用梯度累积（需要修改代码）

### Q2: 训练太慢怎么办？
**方案1**: 增大batch_size到64
**方案2**: 减少epochs到200
**方案3**: 使用预处理数据（--use_preprocessed 1）

### Q3: 过拟合怎么办？
**方案1**: 增大dropout到0.2-0.3
**方案2**: 增大weight_decay到1e-4
**方案3**: 使用数据增强（如果可用）
**方案4**: 减小模型复杂度

### Q4: 欠拟合怎么办？
**方案1**: 增加模型深度 (layers=6)
**方案2**: 增大hidden_features到384
**方案3**: 训练更多epochs
**方案4**: 减小dropout到0.05

---

## 实用工具脚本

### 1. 实时监控训练
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控训练日志
tail -f output_full_training_v1/train.log
```

### 2. 继续训练（如果中断）
```bash
python train_with_cross_modal_attention.py \
    --resume 1 \
    --output_dir ./output_full_training_v1 \
    # ... 其他参数保持一致
```

### 3. 批量评估
```bash
# 评估所有checkpoint
for ckpt in output_full_training_v1/checkpoint_*.pt; do
    echo "Evaluating $ckpt"
    python evaluate.py --model_path $ckpt --dataset test
done
```

---

## 建议的输出目录结构

```
output_full_training_v1/
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
├── ...
├── best_model.pt           # 验证集最优模型
├── final_model.pt          # 最后一个epoch的模型
├── train.log               # 训练日志
├── config.json             # 训练配置
├── train_losses.csv        # 训练loss曲线
├── val_metrics.csv         # 验证指标曲线
└── tensorboard/            # TensorBoard日志
```
