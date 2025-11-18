"""
增强的可解释性分析工具 - 完整支持跨模态注意力可视化

主要功能：
1. 跨模态注意力权重提取和可视化
2. 原子重要性分析（梯度法、积分梯度法、Layer-wise Relevance Propagation）
3. 文本Token重要性分析
4. 特征空间对齐可视化
5. 单样本完整解释报告
6. 批量可解释性分析

作者: Enhanced Interpretability Module
日期: 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm


class EnhancedInterpretabilityAnalyzer:
    """增强版可解释性分析器 - 支持完整的跨模态注意力分析"""

    def __init__(self, model, tokenizer=None, device='cuda'):
        """
        Args:
            model: 训练好的 ALIGNN 模型
            tokenizer: 文本tokenizer（可选，用于token级别分析）
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # 检查模型是否支持注意力提取
        self.has_cross_modal = hasattr(model, 'use_cross_modal_attention') and \
                              model.use_cross_modal_attention
        self.has_middle_fusion = hasattr(model, 'use_middle_fusion') and \
                                model.use_middle_fusion

        print(f"\n🔍 可解释性分析器初始化:")
        print(f"  - 跨模态注意力: {'✅ 支持' if self.has_cross_modal else '❌ 未启用'}")
        print(f"  - 中期融合: {'✅ 支持' if self.has_middle_fusion else '❌ 未启用'}")
        print(f"  - 设备: {device}\n")

    def extract_attention_weights(self, g, lg, text, return_prediction=True):
        """
        提取跨模态注意力权重（完整版）

        Args:
            g: DGL graph
            lg: Line graph
            text: 文本列表
            return_prediction: 是否返回预测值

        Returns:
            Dict包含:
                - attention_weights: 注意力权重字典
                - prediction: 预测值
                - graph_features: 图特征
                - text_features: 文本特征
        """
        self.model.eval()

        with torch.no_grad():
            # 🔑 关键：使用 return_attention=True
            output = self.model(
                [g.to(self.device), lg.to(self.device), text],
                return_features=True,
                return_attention=True  # 返回注意力权重
            )

        result = {}

        if isinstance(output, dict):
            result['prediction'] = output['predictions'].cpu().numpy()
            result['graph_features'] = output.get('graph_features', None)
            result['text_features'] = output.get('text_features', None)

            # 提取全局注意力权重（向后兼容）
            if 'attention_weights' in output:
                attn = output['attention_weights']
                result['attention_weights'] = {
                    'graph_to_text': attn.get('graph_to_text', None),
                    'text_to_graph': attn.get('text_to_graph', None)
                }
            else:
                result['attention_weights'] = None

            # 提取细粒度注意力权重（新增）
            if 'fine_grained_attention_weights' in output:
                fg_attn = output['fine_grained_attention_weights']
                result['fine_grained_attention_weights'] = {
                    'atom_to_text': fg_attn.get('atom_to_text', None),  # [batch, heads, num_atoms, seq_len]
                    'text_to_atom': fg_attn.get('text_to_atom', None)   # [batch, heads, seq_len, num_atoms]
                }
            else:
                result['fine_grained_attention_weights'] = None
        else:
            result['prediction'] = output.cpu().numpy()
            result['attention_weights'] = None

        return result

    def compute_atom_importance(self, g, lg, text, method='gradient', target_class=None):
        """
        计算原子重要性分数

        Args:
            g: DGL graph
            lg: Line graph
            text: 文本
            method: 'gradient' 或 'integrated_gradients'
            target_class: 目标类别（分类任务）

        Returns:
            importance_scores: [num_atoms] 重要性分数
        """
        self.model.eval()

        if method == 'gradient':
            return self._gradient_importance(g, lg, text, target_class)
        elif method == 'integrated_gradients':
            return self._integrated_gradients(g, lg, text, target_class)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _gradient_importance(self, g, lg, text, target_class=None):
        """梯度法计算原子重要性"""
        g = g.to(self.device)
        lg = lg.to(self.device)

        # 启用梯度
        node_features = g.ndata['atom_features'].clone().detach().requires_grad_(True)
        original_features = g.ndata['atom_features']
        g.ndata['atom_features'] = node_features

        # Forward
        output = self.model([g, lg, text])

        if isinstance(output, dict):
            prediction = output['predictions']
        else:
            prediction = output

        # 选择目标输出
        if target_class is not None:
            if prediction.dim() > 1:
                prediction = prediction[:, target_class]

        # 对预测求和（如果是batch）
        loss = prediction.sum()

        # Backward
        loss.backward()

        # 计算重要性（梯度L2范数）
        gradients = node_features.grad
        importance = torch.norm(gradients, dim=1).cpu().numpy()

        # 恢复
        g.ndata['atom_features'] = original_features

        return importance

    def _integrated_gradients(self, g, lg, text, target_class=None, steps=50):
        """积分梯度法计算原子重要性"""
        g = g.to(self.device)
        lg = lg.to(self.device)

        original_features = g.ndata['atom_features'].clone()
        baseline = torch.zeros_like(original_features)

        integrated_grads = torch.zeros_like(original_features)

        for alpha in torch.linspace(0, 1, steps):
            # 插值
            interpolated = baseline + alpha * (original_features - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            g.ndata['atom_features'] = interpolated

            # Forward
            output = self.model([g, lg, text])
            if isinstance(output, dict):
                prediction = output['predictions']
            else:
                prediction = output

            if target_class is not None:
                if prediction.dim() > 1:
                    prediction = prediction[:, target_class]

            loss = prediction.sum()
            loss.backward()

            integrated_grads += interpolated.grad

        # 平均并缩放
        integrated_grads = integrated_grads / steps
        importance = torch.norm(integrated_grads * (original_features - baseline), dim=1)

        # 恢复
        g.ndata['atom_features'] = original_features

        return importance.cpu().numpy()

    def visualize_cross_modal_attention(
        self,
        attention_weights,
        atom_symbols=None,
        text_tokens=None,
        save_path=None,
        figsize=(14, 10)
    ):
        """
        可视化跨模态注意力权重（增强版）

        Args:
            attention_weights: 注意力权重字典
            atom_symbols: 原子符号列表
            text_tokens: 文本token列表
            save_path: 保存路径
            figsize: 图像大小
        """
        if attention_weights is None:
            print("⚠️  没有可用的注意力权重（模型可能未启用跨模态注意力）")
            return

        fig = plt.figure(figsize=figsize)

        # 1. Graph-to-Text Attention
        if 'graph_to_text' in attention_weights and attention_weights['graph_to_text'] is not None:
            ax1 = fig.add_subplot(211)

            g2t_attn = attention_weights['graph_to_text']
            # [batch, heads, 1, 1] -> 取第一个样本，平均所有头
            if g2t_attn.dim() == 4:
                g2t_attn = g2t_attn[0].mean(dim=0).cpu().numpy()  # [1, 1]
            else:
                g2t_attn = g2t_attn.cpu().numpy()

            sns.heatmap(
                g2t_attn,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax1
            )
            ax1.set_title('Graph-to-Text Attention\n(图关注文本的强度)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Text Features')
            ax1.set_ylabel('Graph Features')

        # 2. Text-to-Graph Attention
        if 'text_to_graph' in attention_weights and attention_weights['text_to_graph'] is not None:
            ax2 = fig.add_subplot(212)

            t2g_attn = attention_weights['text_to_graph']
            if t2g_attn.dim() == 4:
                t2g_attn = t2g_attn[0].mean(dim=0).cpu().numpy()
            else:
                t2g_attn = t2g_attn.cpu().numpy()

            sns.heatmap(
                t2g_attn,
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'},
                ax=ax2
            )
            ax2.set_title('Text-to-Graph Attention\n(文本关注图的强度)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Graph Features')
            ax2.set_ylabel('Text Features')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 注意力可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_attention_by_heads(
        self,
        attention_weights,
        save_path=None,
        figsize=(16, 4)
    ):
        """
        按注意力头分别可视化

        Args:
            attention_weights: 注意力权重字典
            save_path: 保存路径
            figsize: 图像大小
        """
        if attention_weights is None or 'graph_to_text' not in attention_weights:
            print("⚠️  没有可用的多头注意力权重")
            return

        g2t = attention_weights['graph_to_text']  # [batch, heads, 1, 1]

        if g2t.dim() != 4:
            print("⚠️  注意力权重维度不符合多头格式")
            return

        num_heads = g2t.shape[1]
        g2t = g2t[0].cpu().numpy()  # [heads, 1, 1]

        fig, axes = plt.subplots(1, num_heads, figsize=figsize)

        for i in range(num_heads):
            ax = axes[i] if num_heads > 1 else axes

            sns.heatmap(
                g2t[i],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar=False,
                ax=ax
            )
            ax.set_title(f'Head {i+1}', fontweight='bold')
            ax.set_xlabel('Text')
            ax.set_ylabel('Graph')

        plt.suptitle('Multi-Head Attention Weights (Graph → Text)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 多头注意力可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_atom_importance(
        self,
        atoms_object,
        importance_scores,
        save_path=None,
        top_k=10,
        figsize=(16, 5)
    ):
        """
        可视化原子重要性（增强版）

        Args:
            atoms_object: jarvis.core.atoms.Atoms对象
            importance_scores: 重要性分数
            save_path: 保存路径
            top_k: 显示top-k重要原子
            figsize: 图像大小

        Returns:
            df: 包含原子信息和重要性的DataFrame
        """
        # 归一化
        importance_scores = (importance_scores - importance_scores.min()) / \
                          (importance_scores.max() - importance_scores.min() + 1e-8)

        # 创建DataFrame
        elements = list(atoms_object.elements)
        coords = atoms_object.cart_coords

        df = pd.DataFrame({
            'Index': range(len(elements)),
            'Element': elements,
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'Importance': importance_scores
        })

        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)

        # 打印Top-k
        print(f"\n{'='*70}")
        print(f"Top {top_k} Most Important Atoms")
        print(f"{'='*70}")
        print(df.head(top_k)[['Index', 'Element', 'Importance']].to_string(index=False))
        print(f"{'='*70}\n")

        # 可视化
        fig = plt.figure(figsize=figsize)

        # 1. 重要性分布
        ax1 = fig.add_subplot(131)
        bars = ax1.bar(range(len(importance_scores)), importance_scores,
                      color=plt.cm.YlOrRd(importance_scores))
        ax1.set_xlabel('Atom Index', fontsize=11)
        ax1.set_ylabel('Importance Score', fontsize=11)
        ax1.set_title('Atom Importance Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 高亮top-k
        top_indices = df.head(top_k)['Index'].values
        for idx in top_indices:
            ax1.axvline(idx, color='red', alpha=0.3, linestyle='--', linewidth=1)

        # 2. 按元素类型统计
        ax2 = fig.add_subplot(132)
        element_stats = df.groupby('Element')['Importance'].agg(['mean', 'std', 'count'])
        element_stats = element_stats.sort_values('mean', ascending=False)

        x_pos = np.arange(len(element_stats))
        ax2.barh(x_pos, element_stats['mean'].values,
                color=plt.cm.viridis(np.linspace(0, 1, len(element_stats))))
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels([f"{elem} (n={int(count)})"
                             for elem, count in zip(element_stats.index, element_stats['count'])])
        ax2.set_xlabel('Average Importance', fontsize=11)
        ax2.set_title('Importance by Element Type', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

        # 3. 空间分布（2D投影）
        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(
            coords[:, 0], coords[:, 1],
            c=importance_scores,
            s=300,
            cmap='YlOrRd',
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        # 标注top-k原子
        for idx in top_indices[:min(top_k, 5)]:  # 只标注前5个避免拥挤
            ax3.annotate(
                f"{elements[idx]}",
                (coords[idx, 0], coords[idx, 1]),
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
            )

        ax3.set_xlabel('X Coordinate (Å)', fontsize=11)
        ax3.set_ylabel('Y Coordinate (Å)', fontsize=11)
        ax3.set_title('Spatial Distribution (X-Y Projection)', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax3, label='Importance Score')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 原子重要性可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

        return df

    def visualize_feature_space(
        self,
        graph_features,
        text_features,
        labels=None,
        predictions=None,
        method='tsne',
        save_path=None,
        figsize=(14, 6)
    ):
        """
        可视化特征空间（增强版）

        Args:
            graph_features: 图特征 [N, D]
            text_features: 文本特征 [N, D]
            labels: 真实标签
            predictions: 预测值
            method: 'tsne' 或 'pca'
            save_path: 保存路径
            figsize: 图像大小
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA

        # 转换为numpy
        if isinstance(graph_features, torch.Tensor):
            graph_features = graph_features.cpu().numpy()
        if isinstance(text_features, torch.Tensor):
            text_features = text_features.cpu().numpy()

        # 合并特征
        all_features = np.vstack([graph_features, text_features])

        # 降维
        print(f"\n🔄 使用 {method.upper()} 进行降维...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features)-1))
        else:
            reducer = PCA(n_components=2)

        embedded = reducer.fit_transform(all_features)

        # 分离
        n = len(graph_features)
        graph_emb = embedded[:n]
        text_emb = embedded[n:]

        # 可视化
        fig = plt.figure(figsize=figsize)

        # 左图：模态分离
        ax1 = fig.add_subplot(121)
        ax1.scatter(graph_emb[:, 0], graph_emb[:, 1],
                   c='#3498db', alpha=0.6, s=80, label='Graph',
                   marker='o', edgecolors='black', linewidth=0.5)
        ax1.scatter(text_emb[:, 0], text_emb[:, 1],
                   c='#e74c3c', alpha=0.6, s=80, label='Text',
                   marker='^', edgecolors='black', linewidth=0.5)

        # 绘制配对连线（采样避免过密）
        sample_size = min(50, n)
        sample_indices = np.random.choice(n, sample_size, replace=False)
        for i in sample_indices:
            ax1.plot([graph_emb[i, 0], text_emb[i, 0]],
                    [graph_emb[i, 1], text_emb[i, 1]],
                    'gray', alpha=0.2, linewidth=0.8)

        ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
        ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
        ax1.set_title('Graph-Text Feature Alignment', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 右图：按标签着色
        ax2 = fig.add_subplot(122)
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            scatter1 = ax2.scatter(graph_emb[:, 0], graph_emb[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6, s=80,
                                  marker='o', edgecolors='black', linewidth=0.5)
            scatter2 = ax2.scatter(text_emb[:, 0], text_emb[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6, s=80,
                                  marker='^', edgecolors='black', linewidth=0.5)
            cbar = plt.colorbar(scatter1, ax=ax2, label='Target Value')
            ax2.set_title('Features Colored by Target', fontsize=12, fontweight='bold')
        else:
            ax2.scatter(graph_emb[:, 0], graph_emb[:, 1],
                       c='#3498db', alpha=0.6, s=80, label='Graph', marker='o')
            ax2.scatter(text_emb[:, 0], text_emb[:, 1],
                       c='#e74c3c', alpha=0.6, s=80, label='Text', marker='^')
            ax2.set_title('Feature Distribution', fontsize=12, fontweight='bold')
            ax2.legend()

        ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=11)
        ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 特征空间可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def explain_single_prediction(
        self,
        g, lg, text,
        atoms_object,
        true_value=None,
        save_dir=None,
        sample_id='sample'
    ):
        """
        为单个样本生成完整解释报告

        Args:
            g, lg, text: 模型输入
            atoms_object: Atoms对象
            true_value: 真实值
            save_dir: 保存目录
            sample_id: 样本ID

        Returns:
            explanation: 解释字典
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🔍 样本 {sample_id} 的可解释性分析")
        print(f"{'='*80}")

        explanation = {}

        # 1. 提取注意力和预测
        print("\n1️⃣  提取注意力权重和预测...")
        result = self.extract_attention_weights(g, lg, text, return_prediction=True)

        prediction = result['prediction'][0] if len(result['prediction'].shape) > 0 else result['prediction']
        explanation['prediction'] = float(prediction)
        explanation['true_value'] = float(true_value) if true_value is not None else None

        if true_value is not None:
            error = abs(prediction - true_value)
            explanation['error'] = float(error)
            print(f"   预测值: {prediction:.4f}")
            print(f"   真实值: {true_value:.4f}")
            print(f"   误差: {error:.4f}")
        else:
            print(f"   预测值: {prediction:.4f}")

        # 2. 可视化注意力
        if result['attention_weights'] is not None:
            print("\n2️⃣  可视化跨模态注意力...")
            attn_path = save_dir / f'{sample_id}_attention.png' if save_dir else None
            self.visualize_cross_modal_attention(
                result['attention_weights'],
                save_path=attn_path
            )

            # 多头注意力
            heads_path = save_dir / f'{sample_id}_attention_heads.png' if save_dir else None
            self.visualize_attention_by_heads(
                result['attention_weights'],
                save_path=heads_path
            )
        else:
            print("\n2️⃣  ⚠️  跨模态注意力未启用")

        # 3. 计算原子重要性
        print("\n3️⃣  计算原子重要性...")
        atom_importance_grad = self.compute_atom_importance(g, lg, text, method='gradient')
        explanation['atom_importance'] = atom_importance_grad.tolist()

        # 可视化
        atom_path = save_dir / f'{sample_id}_atom_importance.png' if save_dir else None
        atom_df = self.visualize_atom_importance(
            atoms_object,
            atom_importance_grad,
            save_path=atom_path,
            top_k=10
        )

        # 4. 保存解释
        if save_dir:
            with open(save_dir / f'{sample_id}_explanation.json', 'w') as f:
                json.dump(explanation, f, indent=2)
            print(f"\n✅ 解释报告已保存到: {save_dir}")

        print(f"\n{'='*80}\n")

        return explanation

    def visualize_fine_grained_attention(
        self,
        attention_weights,
        atoms_object,
        text_tokens,
        save_path=None,
        top_k_atoms=10,
        top_k_words=15,
        show_all_heads=False
    ):
        """
        可视化细粒度注意力权重（原子-文本token级别）

        Args:
            attention_weights: 细粒度注意力权重字典
                - 'atom_to_text': [batch, heads, num_atoms, seq_len]
                - 'text_to_atom': [batch, heads, seq_len, num_atoms]
            atoms_object: Atoms对象（JARVIS）
            text_tokens: 文本tokens列表（解码后的词语）
            save_path: 保存路径
            top_k_atoms: 显示top-k重要的原子
            top_k_words: 显示top-k重要的词语
            show_all_heads: 是否显示所有注意力头

        Returns:
            分析结果字典
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if attention_weights is None:
            print("⚠️  没有细粒度注意力权重")
            return None

        atom_to_text = attention_weights.get('atom_to_text', None)
        text_to_atom = attention_weights.get('text_to_atom', None)

        if atom_to_text is None and text_to_atom is None:
            print("⚠️  没有找到细粒度注意力权重")
            return None

        # Convert to numpy
        if atom_to_text is not None:
            atom_to_text = atom_to_text.cpu().numpy()  # [batch, heads, num_atoms, seq_len]
        if text_to_atom is not None:
            text_to_atom = text_to_atom.cpu().numpy()  # [batch, heads, seq_len, num_atoms]

        # For single sample (batch_size=1)
        if atom_to_text is not None:
            atom_to_text = atom_to_text[0]  # [heads, num_atoms, seq_len]
        if text_to_atom is not None:
            text_to_atom = text_to_atom[0]  # [heads, seq_len, num_atoms]

        num_heads = atom_to_text.shape[0] if atom_to_text is not None else text_to_atom.shape[0]
        num_atoms = atom_to_text.shape[1] if atom_to_text is not None else text_to_atom.shape[2]
        seq_len = atom_to_text.shape[2] if atom_to_text is not None else text_to_atom.shape[1]

        # Get atom elements
        elements = [str(atoms_object.elements[i]) for i in range(num_atoms)]

        # Average over heads
        atom_to_text_avg = atom_to_text.mean(axis=0) if atom_to_text is not None else None  # [num_atoms, seq_len]
        text_to_atom_avg = text_to_atom.mean(axis=0) if text_to_atom is not None else None  # [seq_len, num_atoms]

        # Create visualization
        if show_all_heads:
            # Show each head separately
            fig, axes = plt.subplots(2, num_heads//2 + num_heads%2, figsize=(20, 8))
            axes = axes.flatten()

            for head in range(num_heads):
                sns.heatmap(
                    atom_to_text[head],
                    xticklabels=text_tokens[:seq_len],
                    yticklabels=elements,
                    cmap='YlOrRd',
                    ax=axes[head],
                    cbar=True
                )
                axes[head].set_title(f'Head {head+1}')
                axes[head].set_xlabel('Text Tokens')
                axes[head].set_ylabel('Atoms')

            plt.tight_layout()
        else:
            # Show averaged attention
            fig, axes = plt.subplots(1, 2, figsize=(20, max(8, num_atoms * 0.5)))

            # Atom-to-Text attention heatmap
            if atom_to_text_avg is not None:
                sns.heatmap(
                    atom_to_text_avg,
                    xticklabels=text_tokens[:seq_len],
                    yticklabels=elements,
                    cmap='YlOrRd',
                    ax=axes[0],
                    cbar=True,
                    annot=num_atoms <= 10 and seq_len <= 20  # Only annotate if small enough
                )
                axes[0].set_title('Atom → Text Attention\n(Which words does each atom attend to?)', fontsize=12)
                axes[0].set_xlabel('Text Tokens', fontsize=10)
                axes[0].set_ylabel('Atoms (Element)', fontsize=10)
                plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)

            # Text-to-Atom attention heatmap
            if text_to_atom_avg is not None:
                sns.heatmap(
                    text_to_atom_avg,
                    xticklabels=elements,
                    yticklabels=text_tokens[:seq_len],
                    cmap='YlGnBu',
                    ax=axes[1],
                    cbar=True,
                    annot=num_atoms <= 10 and seq_len <= 20
                )
                axes[1].set_title('Text → Atom Attention\n(Which atoms does each word attend to?)', fontsize=12)
                axes[1].set_xlabel('Atoms (Element)', fontsize=10)
                axes[1].set_ylabel('Text Tokens', fontsize=10)
                plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=8)

            plt.suptitle('Fine-Grained Cross-Modal Attention', fontsize=14, fontweight='bold')
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 细粒度注意力可视化已保存: {save_path}")

        plt.close()

        # Analyze patterns
        analysis = {}

        if atom_to_text_avg is not None:
            # Top words attended by each atom
            analysis['atom_top_words'] = {}
            for i, element in enumerate(elements):
                top_word_indices = atom_to_text_avg[i].argsort()[-top_k_words:][::-1]
                top_words = [(text_tokens[idx], atom_to_text_avg[i, idx]) for idx in top_word_indices]
                analysis['atom_top_words'][f"{element}_{i}"] = top_words

            # Overall most important words (averaged over all atoms)
            word_importance = atom_to_text_avg.mean(axis=0)  # [seq_len]
            top_word_indices = word_importance.argsort()[-top_k_words:][::-1]
            analysis['overall_top_words'] = [
                (text_tokens[idx], word_importance[idx]) for idx in top_word_indices
            ]

        if text_to_atom_avg is not None:
            # Top atoms attended by each word
            analysis['word_top_atoms'] = {}
            for i, token in enumerate(text_tokens[:seq_len]):
                top_atom_indices = text_to_atom_avg[i].argsort()[-top_k_atoms:][::-1]
                top_atoms = [(f"{elements[idx]}_{idx}", text_to_atom_avg[i, idx]) for idx in top_atom_indices]
                analysis['word_top_atoms'][token] = top_atoms

            # Overall most important atoms (averaged over all words)
            atom_importance = text_to_atom_avg.mean(axis=0)  # [num_atoms]
            top_atom_indices = atom_importance.argsort()[-top_k_atoms:][::-1]
            analysis['overall_top_atoms'] = [
                (f"{elements[idx]}_{idx}", atom_importance[idx]) for idx in top_atom_indices
            ]

        return analysis


def batch_interpretability_analysis(
    analyzer,
    test_loader,
    save_dir,
    num_samples=10,
    analyze_feature_space=True
):
    """
    批量可解释性分析

    Args:
        analyzer: EnhancedInterpretabilityAnalyzer
        test_loader: 测试数据加载器
        save_dir: 保存目录
        num_samples: 分析样本数
        analyze_feature_space: 是否分析特征空间

    Returns:
        summary: 分析摘要
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"📊 批量可解释性分析")
    print(f"{'='*80}")
    print(f"  样本数: {num_samples}")
    print(f"  保存目录: {save_dir}")
    print(f"{'='*80}\n")

    all_graph_features = []
    all_text_features = []
    all_labels = []
    all_predictions = []

    # 分析每个样本
    for i, batch in enumerate(tqdm(test_loader, desc="分析样本", total=num_samples)):
        if i >= num_samples:
            break

        g, lg, text, labels = batch

        # 提取特征
        result = analyzer.extract_attention_weights(g, lg, text)

        if result['graph_features'] is not None:
            all_graph_features.append(result['graph_features'].cpu())
        if result['text_features'] is not None:
            all_text_features.append(result['text_features'].cpu())

        all_labels.append(labels.cpu())
        all_predictions.append(torch.tensor(result['prediction']))

    # 特征空间可视化
    if analyze_feature_space and all_graph_features and all_text_features:
        print("\n📈 可视化特征空间...")

        graph_features = torch.cat(all_graph_features, dim=0).numpy()
        text_features = torch.cat(all_text_features, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        # t-SNE
        analyzer.visualize_feature_space(
            graph_features, text_features, labels,
            method='tsne',
            save_path=save_dir / 'feature_space_tsne.png'
        )

        # PCA
        analyzer.visualize_feature_space(
            graph_features, text_features, labels,
            method='pca',
            save_path=save_dir / 'feature_space_pca.png'
        )

    print(f"\n✅ 批量分析完成！结果保存在: {save_dir}\n")

    return {
        'num_samples': min(num_samples, len(test_loader)),
        'save_dir': str(save_dir)
    }
