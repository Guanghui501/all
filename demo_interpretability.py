#!/usr/bin/env python
"""
可解释性分析演示脚本

这是一个完整的演示，展示如何使用所有可解释性功能。
可以直接运行此脚本来测试您的模型。

用法:
    python demo_interpretability.py --checkpoint best_model.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def demo_attention_extraction(model, g, lg, text):
    """演示1: 提取跨模态注意力权重"""

    print("\n" + "="*80)
    print("演示1: 提取跨模态注意力权重")
    print("="*80)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # 🔑 关键: 使用 return_attention=True
        output = model(
            [g.to(device), lg.to(device), text],
            return_features=True,
            return_attention=True
        )

    if isinstance(output, dict):
        print("\n✅ 成功提取输出!")
        print(f"   包含键: {list(output.keys())}")

        if 'attention_weights' in output and output['attention_weights'] is not None:
            attn = output['attention_weights']
            print(f"\n✅ 注意力权重:")

            if 'graph_to_text' in attn:
                g2t = attn['graph_to_text']
                print(f"   - Graph→Text 形状: {g2t.shape}")
                print(f"   - Graph→Text 平均值: {g2t.mean():.4f}")
                print(f"   - Graph→Text 范围: [{g2t.min():.4f}, {g2t.max():.4f}]")

            if 'text_to_graph' in attn:
                t2g = attn['text_to_graph']
                print(f"   - Text→Graph 形状: {t2g.shape}")
                print(f"   - Text→Graph 平均值: {t2g.mean():.4f}")
                print(f"   - Text→Graph 范围: [{t2g.min():.4f}, {t2g.max():.4f}]")

            return output['attention_weights']
        else:
            print("\n⚠️  注意力权重未启用")
            print("   请确保模型配置: use_cross_modal_attention=True")
            return None
    else:
        print("\n⚠️  模型未返回字典格式")
        return None


def demo_attention_visualization(attention_weights, save_dir):
    """演示2: 可视化注意力权重"""

    print("\n" + "="*80)
    print("演示2: 可视化注意力权重")
    print("="*80)

    if attention_weights is None:
        print("\n⚠️  没有可用的注意力权重")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 简单的热图可视化
    import seaborn as sns

    fig = plt.figure(figsize=(14, 6))

    # Graph-to-Text
    if 'graph_to_text' in attention_weights:
        ax1 = fig.add_subplot(121)

        g2t = attention_weights['graph_to_text']
        # [batch, heads, 1, 1] -> [heads, 1]
        if g2t.dim() == 4:
            g2t = g2t[0, :, 0, 0].cpu().numpy()
        else:
            g2t = g2t.cpu().numpy()

        sns.heatmap(
            g2t.reshape(-1, 1),
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax1,
            yticklabels=[f'Head {i+1}' for i in range(len(g2t))]
        )
        ax1.set_title('Graph → Text Attention\n(图关注文本)', fontweight='bold')
        ax1.set_xlabel('Text Features')

    # Text-to-Graph
    if 'text_to_graph' in attention_weights:
        ax2 = fig.add_subplot(122)

        t2g = attention_weights['text_to_graph']
        if t2g.dim() == 4:
            t2g = t2g[0, :, 0, 0].cpu().numpy()
        else:
            t2g = t2g.cpu().numpy()

        sns.heatmap(
            t2g.reshape(-1, 1),
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            ax=ax2,
            yticklabels=[f'Head {i+1}' for i in range(len(t2g))]
        )
        ax2.set_title('Text → Graph Attention\n(文本关注图)', fontweight='bold')
        ax2.set_xlabel('Graph Features')

    plt.tight_layout()

    save_path = save_dir / 'attention_weights.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 注意力可视化已保存: {save_path}")

    plt.close()


def demo_atom_importance(model, g, lg, text, atoms, save_dir):
    """演示3: 计算和可视化原子重要性"""

    print("\n" + "="*80)
    print("演示3: 原子重要性分析")
    print("="*80)

    device = next(model.parameters()).device
    save_dir = Path(save_dir)

    # 梯度法
    print("\n🔄 使用梯度法计算原子重要性...")

    model.eval()
    g = g.to(device)
    lg = lg.to(device)

    # 启用梯度
    node_features = g.ndata['atom_features'].clone().detach().requires_grad_(True)
    original_features = g.ndata['atom_features']
    g.ndata['atom_features'] = node_features

    # Forward
    output = model([g, lg, text])

    if isinstance(output, dict):
        prediction = output['predictions']
    else:
        prediction = output

    # Backward
    loss = prediction.sum()
    loss.backward()

    # 计算重要性
    gradients = node_features.grad
    importance = torch.norm(gradients, dim=1).cpu().numpy()

    # 恢复
    g.ndata['atom_features'] = original_features

    print(f"✅ 原子重要性计算完成")
    print(f"   - 原子数: {len(importance)}")
    print(f"   - 平均重要性: {importance.mean():.4f}")
    print(f"   - 最大重要性: {importance.max():.4f}")
    print(f"   - 最小重要性: {importance.min():.4f}")

    # 归一化
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

    # 打印Top-5
    top_indices = np.argsort(importance_norm)[::-1][:5]
    print(f"\n📊 Top-5 重要原子:")
    print(f"{'='*50}")
    for rank, idx in enumerate(top_indices, 1):
        element = atoms.elements[idx]
        score = importance_norm[idx]
        print(f"   {rank}. 原子 {idx} ({element}): {score:.4f}")
    print(f"{'='*50}")

    # 可视化
    print(f"\n🎨 可视化原子重要性...")

    import pandas as pd
    coords = atoms.cart_coords
    elements = list(atoms.elements)

    fig = plt.figure(figsize=(15, 5))

    # 1. 分布图
    ax1 = fig.add_subplot(131)
    bars = ax1.bar(range(len(importance_norm)), importance_norm,
                  color=plt.cm.YlOrRd(importance_norm))
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Importance Score')
    ax1.set_title('Atom Importance Distribution')
    ax1.grid(True, alpha=0.3)

    # 高亮top-5
    for idx in top_indices:
        ax1.axvline(idx, color='red', alpha=0.3, linestyle='--')

    # 2. 按元素统计
    ax2 = fig.add_subplot(132)
    df = pd.DataFrame({
        'Element': elements,
        'Importance': importance_norm
    })
    element_avg = df.groupby('Element')['Importance'].mean().sort_values(ascending=False)

    ax2.barh(range(len(element_avg)), element_avg.values,
            color=plt.cm.viridis(np.linspace(0, 1, len(element_avg))))
    ax2.set_yticks(range(len(element_avg)))
    ax2.set_yticklabels(element_avg.index)
    ax2.set_xlabel('Average Importance')
    ax2.set_title('Importance by Element')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. 空间分布
    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(coords[:, 0], coords[:, 1],
                         c=importance_norm, s=300,
                         cmap='YlOrRd', alpha=0.7,
                         edgecolors='black', linewidth=1.5)

    # 标注top-3
    for idx in top_indices[:3]:
        ax3.annotate(elements[idx],
                    (coords[idx, 0], coords[idx, 1]),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax3.set_xlabel('X (Å)')
    ax3.set_ylabel('Y (Å)')
    ax3.set_title('Spatial Distribution')
    plt.colorbar(scatter, ax=ax3, label='Importance')

    plt.tight_layout()

    save_path = save_dir / 'atom_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 原子重要性可视化已保存: {save_path}")

    plt.close()

    return importance_norm


def demo_feature_space(model, test_loader, save_dir, num_samples=50):
    """演示4: 特征空间可视化"""

    print("\n" + "="*80)
    print("演示4: 特征空间可视化")
    print("="*80)

    device = next(model.parameters()).device
    save_dir = Path(save_dir)

    print(f"\n🔄 收集 {num_samples} 个样本的特征...")

    all_graph_features = []
    all_text_features = []
    all_labels = []

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in test_loader:
            if count >= num_samples:
                break

            g, lg, text, labels = batch

            output = model(
                [g.to(device), lg.to(device), text],
                return_features=True
            )

            if isinstance(output, dict):
                if 'graph_features' in output and output['graph_features'] is not None:
                    all_graph_features.append(output['graph_features'].cpu())
                if 'text_features' in output and output['text_features'] is not None:
                    all_text_features.append(output['text_features'].cpu())

            all_labels.append(labels.cpu())
            count += len(labels)

            print(f"   收集进度: {count}/{num_samples}", end='\r')

    print(f"\n✅ 特征收集完成: {count} 个样本")

    if not all_graph_features or not all_text_features:
        print("⚠️  无法提取特征（模型可能未返回中间特征）")
        return

    # 合并
    graph_features = torch.cat(all_graph_features, dim=0).numpy()
    text_features = torch.cat(all_text_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    print(f"\n📊 特征统计:")
    print(f"   - 图特征形状: {graph_features.shape}")
    print(f"   - 文本特征形状: {text_features.shape}")

    # 计算余弦相似度
    cosine_sim = []
    for i in range(len(graph_features)):
        sim = np.dot(graph_features[i], text_features[i]) / \
              (np.linalg.norm(graph_features[i]) * np.linalg.norm(text_features[i]))
        cosine_sim.append(sim)

    print(f"   - 平均余弦相似度: {np.mean(cosine_sim):.4f}")
    print(f"   - 相似度范围: [{np.min(cosine_sim):.4f}, {np.max(cosine_sim):.4f}]")

    # t-SNE可视化
    print(f"\n🎨 生成 t-SNE 可视化...")
    from sklearn.manifold import TSNE

    all_features = np.vstack([graph_features, text_features])
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(all_features)

    n = len(graph_features)
    graph_emb = embedded[:n]
    text_emb = embedded[n:]

    # 可视化
    fig = plt.figure(figsize=(14, 6))

    # 左图: 模态分离
    ax1 = fig.add_subplot(121)
    ax1.scatter(graph_emb[:, 0], graph_emb[:, 1],
               c='#3498db', alpha=0.6, s=80, label='Graph',
               marker='o', edgecolors='black', linewidth=0.5)
    ax1.scatter(text_emb[:, 0], text_emb[:, 1],
               c='#e74c3c', alpha=0.6, s=80, label='Text',
               marker='^', edgecolors='black', linewidth=0.5)

    # 连线
    for i in np.random.choice(n, min(30, n), replace=False):
        ax1.plot([graph_emb[i, 0], text_emb[i, 0]],
                [graph_emb[i, 1], text_emb[i, 1]],
                'gray', alpha=0.2, linewidth=0.8)

    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_title('Graph-Text Alignment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 按标签着色
    ax2 = fig.add_subplot(122)
    scatter1 = ax2.scatter(graph_emb[:, 0], graph_emb[:, 1],
                          c=labels, cmap='viridis', alpha=0.6, s=80,
                          marker='o', edgecolors='black', linewidth=0.5)
    ax2.scatter(text_emb[:, 0], text_emb[:, 1],
               c=labels, cmap='viridis', alpha=0.6, s=80,
               marker='^', edgecolors='black', linewidth=0.5)

    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.set_title('Features Colored by Target')
    plt.colorbar(scatter1, ax=ax2, label='Target Value')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = save_dir / 'feature_space_tsne.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 特征空间可视化已保存: {save_path}")

    plt.close()


def main():
    """主函数 - 运行所有演示"""

    print("\n" + "="*80)
    print("🔍 可解释性分析演示")
    print("="*80)

    # 这里需要加载您的实际模型和数据
    print("\n⚠️  注意: 这是一个演示脚本模板")
    print("请修改以下部分以适配您的实际模型和数据:\n")

    print("TODO:")
    print("1. 加载训练好的模型")
    print("2. 加载测试数据")
    print("3. 准备一个样本用于演示")
    print("4. 运行演示函数\n")

    # 示例代码（需要替换）:
    """
    # 1. 加载模型
    checkpoint = torch.load('best_model.pt')
    model = ALIGNN(config.model)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 2. 加载数据
    test_loader = ...

    # 3. 获取一个样本
    g, lg, text, label = next(iter(test_loader))
    atoms = ...  # 从数据中获取Atoms对象

    # 4. 运行演示
    save_dir = Path('./demo_results')

    # 演示1: 注意力提取
    attention = demo_attention_extraction(model, g, lg, text)

    # 演示2: 注意力可视化
    demo_attention_visualization(attention, save_dir)

    # 演示3: 原子重要性
    demo_atom_importance(model, g, lg, text, atoms, save_dir)

    # 演示4: 特征空间
    demo_feature_space(model, test_loader, save_dir, num_samples=50)

    print(f"\n✅ 所有演示完成！结果保存在: {save_dir}")
    """

    print("="*80)


if __name__ == '__main__':
    main()
