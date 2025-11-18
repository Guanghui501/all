#!/usr/bin/env python
"""
可解释性分析运行脚本

用法示例:
    # 单样本分析
    python run_interpretability_analysis.py --mode single --sample_id 0

    # 批量分析
    python run_interpretability_analysis.py --mode batch --num_samples 50

    # 特征空间可视化
    python run_interpretability_analysis.py --mode feature_space
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from interpretability_enhanced import EnhancedInterpretabilityAnalyzer, batch_interpretability_analysis
from data import get_train_val_loaders
from config import TrainingConfig
from models.alignn import ALIGNN
from jarvis.core.atoms import Atoms


def load_trained_model(checkpoint_path, config_path=None):
    """
    加载训练好的模型

    Args:
        checkpoint_path: checkpoint文件路径
        config_path: 配置文件路径（可选）

    Returns:
        model: 加载的模型
        config: 配置对象
    """
    print(f"\n{'='*80}")
    print("🔄 加载模型...")
    print(f"{'='*80}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载或创建配置
    if config_path and os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        # 使用默认配置
        from models.alignn import ALIGNNConfig
        model_config = ALIGNNConfig(
            name="alignn",
            use_cross_modal_attention=True,
            use_middle_fusion=False,
            classification=False
        )
        config = TrainingConfig(model=model_config)

    # 创建模型
    model = ALIGNN(config.model)

    # 加载权重
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    print(f"✅ 模型加载成功")
    print(f"   - 跨模态注意力: {'✅' if config.model.use_cross_modal_attention else '❌'}")
    print(f"   - 中期融合: {'✅' if config.model.use_middle_fusion else '❌'}")
    print(f"{'='*80}\n")

    return model, config


def single_sample_analysis(
    analyzer,
    test_loader,
    sample_idx=0,
    save_dir='./interpretability_results/single_sample'
):
    """
    单样本可解释性分析

    Args:
        analyzer: 可解释性分析器
        test_loader: 测试数据加载器
        sample_idx: 样本索引
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🔍 单样本分析 - Sample {sample_idx}")
    print(f"{'='*80}\n")

    # 获取样本
    dataset = test_loader.dataset

    if sample_idx >= len(dataset):
        print(f"❌ 样本索引 {sample_idx} 超出范围（最大: {len(dataset)-1}）")
        return

    # 获取原始数据
    sample_data = dataset.dataset[dataset.indices[sample_idx]]

    # 获取图数据
    g, lg = dataset.graphs[sample_idx], dataset.line_graphs[sample_idx]
    text = [sample_data['text']]
    target = dataset.labels[sample_idx].item()
    sample_id = sample_data.get('jid', f'sample_{sample_idx}')

    # 获取atoms对象
    atoms_dict = sample_data['atoms']
    atoms = Atoms.from_dict(atoms_dict)

    print(f"样本信息:")
    print(f"  - ID: {sample_id}")
    print(f"  - 原子数: {len(atoms)}")
    print(f"  - 化学式: {atoms.composition.reduced_formula}")
    print(f"  - 真实值: {target:.4f}")
    print(f"  - 文本: {text[0][:100]}...\n")

    # 执行分析
    explanation = analyzer.explain_single_prediction(
        g, lg, text,
        atoms_object=atoms,
        true_value=target,
        save_dir=save_dir,
        sample_id=sample_id
    )

    print(f"✅ 单样本分析完成！")
    print(f"   结果保存在: {save_dir}\n")

    return explanation


def feature_space_analysis(
    analyzer,
    test_loader,
    num_samples=100,
    save_dir='./interpretability_results/feature_space'
):
    """
    特征空间分析

    Args:
        analyzer: 可解释性分析器
        test_loader: 测试数据加载器
        num_samples: 分析样本数
        save_dir: 保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📊 特征空间分析")
    print(f"{'='*80}\n")

    all_graph_features = []
    all_text_features = []
    all_labels = []
    all_predictions = []

    print(f"收集 {num_samples} 个样本的特征...")

    count = 0
    for batch in test_loader:
        if count >= num_samples:
            break

        g, lg, text, labels = batch

        # 提取特征
        with torch.no_grad():
            result = analyzer.extract_attention_weights(g, lg, text)

        if result['graph_features'] is not None:
            all_graph_features.append(result['graph_features'].cpu())
        if result['text_features'] is not None:
            all_text_features.append(result['text_features'].cpu())

        all_labels.append(labels.cpu())
        all_predictions.append(torch.tensor(result['prediction']))

        count += len(labels)
        print(f"  已收集: {count}/{num_samples}", end='\r')

    print(f"\n\n✅ 特征收集完成！共 {count} 个样本")

    if not all_graph_features or not all_text_features:
        print("❌ 无法提取特征（模型可能未返回中间特征）")
        return

    # 合并特征
    graph_features = torch.cat(all_graph_features, dim=0).numpy()
    text_features = torch.cat(all_text_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    predictions = torch.cat(all_predictions, dim=0).numpy()

    print(f"\n特征维度:")
    print(f"  - 图特征: {graph_features.shape}")
    print(f"  - 文本特征: {text_features.shape}")
    print(f"  - 标签: {labels.shape}\n")

    # t-SNE可视化
    print("📈 生成 t-SNE 可视化...")
    analyzer.visualize_feature_space(
        graph_features, text_features,
        labels=labels,
        predictions=predictions,
        method='tsne',
        save_path=save_dir / 'feature_space_tsne.png'
    )

    # PCA可视化
    print("📈 生成 PCA 可视化...")
    analyzer.visualize_feature_space(
        graph_features, text_features,
        labels=labels,
        predictions=predictions,
        method='pca',
        save_path=save_dir / 'feature_space_pca.png'
    )

    # 保存统计信息
    import numpy as np
    stats = {
        'num_samples': int(count),
        'graph_feature_mean': float(graph_features.mean()),
        'graph_feature_std': float(graph_features.std()),
        'text_feature_mean': float(text_features.mean()),
        'text_feature_std': float(text_features.std()),
        'cosine_similarity': float(np.mean([
            np.dot(graph_features[i], text_features[i]) /
            (np.linalg.norm(graph_features[i]) * np.linalg.norm(text_features[i]))
            for i in range(min(100, len(graph_features)))
        ]))
    }

    import json
    with open(save_dir / 'feature_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ 特征空间分析完成！")
    print(f"   结果保存在: {save_dir}")
    print(f"\n特征统计:")
    for key, value in stats.items():
        print(f"  - {key}: {value:.4f}")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='可解释性分析工具')

    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'batch', 'feature_space'],
                       help='分析模式')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据集路径')
    parser.add_argument('--sample_id', type=int, default=0,
                       help='单样本分析的样本索引')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='批量分析的样本数')
    parser.add_argument('--save_dir', type=str, default='./interpretability_results',
                       help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')

    args = parser.parse_args()

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 加载模型
    model, config = load_trained_model(args.checkpoint, args.config)
    model = model.to(device)
    model.eval()

    # 创建分析器
    analyzer = EnhancedInterpretabilityAnalyzer(model, device=device)

    # 加载数据（这里需要根据实际情况调整）
    # 简化示例：假设有预处理好的数据
    print("\n⚠️  注意: 需要提供实际的数据加载逻辑")
    print("请修改此脚本以适配您的数据格式\n")

    # 示例数据加载（需要替换）
    # train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
    #     dataset_array=dataset_array,
    #     target='target',
    #     ...
    # )

    # 执行分析
    # if args.mode == 'single':
    #     single_sample_analysis(
    #         analyzer, test_loader,
    #         sample_idx=args.sample_id,
    #         save_dir=os.path.join(args.save_dir, 'single_sample')
    #     )
    # elif args.mode == 'batch':
    #     batch_interpretability_analysis(
    #         analyzer, test_loader,
    #         save_dir=os.path.join(args.save_dir, 'batch_analysis'),
    #         num_samples=args.num_samples
    #     )
    # elif args.mode == 'feature_space':
    #     feature_space_analysis(
    #         analyzer, test_loader,
    #         num_samples=args.num_samples,
    #         save_dir=os.path.join(args.save_dir, 'feature_space')
    #     )


if __name__ == '__main__':
    main()
