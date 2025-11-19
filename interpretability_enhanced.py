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

    # 停用词将从文件加载
    _STOPWORDS = None

    @classmethod
    def load_stopwords(cls, stopwords_dir=None):
        """从文件加载停用词

        Args:
            stopwords_dir: 停用词目录路径，默认为 ./stopwords/en/

        Returns:
            停用词集合
        """
        if cls._STOPWORDS is not None:
            return cls._STOPWORDS

        stopwords = set()

        # 默认停用词目录
        if stopwords_dir is None:
            # 尝试多个可能的路径
            possible_dirs = [
                Path(__file__).parent / 'stopwords' / 'en',
                Path('./stopwords/en'),
                Path('../stopwords/en'),
            ]
            for d in possible_dirs:
                if d.exists():
                    stopwords_dir = d
                    break

        if stopwords_dir is None or not Path(stopwords_dir).exists():
            print(f"⚠️ 停用词目录不存在，使用默认停用词")
            # 使用基本停用词
            stopwords = {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                'and', 'or', 'but', 'not', 'no', 'if', 'that', 'this', 'it', 'its',
            }
        else:
            # 从所有txt文件加载停用词
            stopwords_path = Path(stopwords_dir)
            txt_files = list(stopwords_path.glob('*.txt'))

            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip().lower()
                            if word and not word.startswith('#'):  # 跳过空行和注释
                                stopwords.add(word)
                except Exception as e:
                    pass  # 静默处理读取错误

            print(f"✅ 从 {len(txt_files)} 个文件加载了 {len(stopwords)} 个停用词")

        # 添加 BERT 特殊 token 和标点符号
        special_tokens = {
            '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]',
            '.', ',', '(', ')', '-', '–', ':', ';', '"', "'", '/', '\\',
            '##s', '##ed', '##ing', '##ly', '##er', '##est', '##tion', '##ment',
        }
        stopwords.update(special_tokens)

        cls._STOPWORDS = stopwords
        return stopwords

    @property
    def STOPWORDS(self):
        """获取停用词集合（懒加载）"""
        if self._STOPWORDS is None:
            self.load_stopwords()
        return self._STOPWORDS

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

        # 加载停用词
        self.load_stopwords()

        print(f"\n🔍 可解释性分析器初始化:")
        print(f"  - 跨模态注意力: {'✅ 支持' if self.has_cross_modal else '❌ 未启用'}")
        print(f"  - 中期融合: {'✅ 支持' if self.has_middle_fusion else '❌ 未启用'}")
        print(f"  - 设备: {device}")
        print(f"  - 停用词过滤: ✅ 已启用 ({len(self.STOPWORDS)} 个停用词)\n")

    @staticmethod
    def _merge_tokens_and_weights(tokens, weights):
        """Merge WordPiece tokens and their corresponding attention weights.

        Handles space groups (F-43m, P63/mmc), N-coordinate (12-coordinate),
        atom labels with site numbers (Ba(1)), and element symbols with digits (Ba4).

        Args:
            tokens: list of token strings
            weights: numpy array of shape [..., seq_len]

        Returns:
            merged_tokens: list of merged token strings
            merged_weights: numpy array with merged weights
            token_mapping: list mapping merged index to original indices
        """
        import numpy as np

        # Space group starting letters (Bravais lattice symbols)
        space_group_starters = {'P', 'I', 'F', 'R', 'C', 'A', 'B'}
        # Characters that are part of space group notation
        space_group_chars = {'-', '/', 'm', 'n', 'c', 'a', 'b', 'd', 'e'}
        # Common atom symbols (1-2 letters)
        atom_symbols = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                       'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                       'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                       'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                       'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                       'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                       'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                       'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                       'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                       'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'}

        merged_tokens = []
        token_mapping = []  # Each element is a list of original indices
        current_token = ""
        current_indices = []
        in_space_group = False  # Track if we're in a space group symbol
        in_coordinate = False  # Track if we're in a N-coordinate pattern
        in_parentheses = False  # Track if we're inside parentheses like Ba(1)

        for i, token in enumerate(tokens):
            if token.startswith("##"):
                # Continue previous token (WordPiece continuation)
                current_token += token[2:]
                current_indices.append(i)
            elif token == '-' and current_token:
                # Check if this is part of space group or N-coordinate pattern
                # Use upper() for case-insensitive space group detection (BERT lowercases)
                if current_token[0].upper() in space_group_starters and len(current_token) <= 4:
                    # Space group pattern (F-43m, P-1, f-43m)
                    current_token += token
                    current_indices.append(i)
                    in_space_group = True
                elif current_token.isdigit():
                    # N-coordinate pattern (12-coordinate)
                    current_token += token
                    current_indices.append(i)
                    in_coordinate = True
                else:
                    # Don't merge "-" with regular words like "bonded"
                    if current_token:
                        merged_tokens.append(current_token)
                        token_mapping.append(current_indices)
                        in_space_group = False
                        in_coordinate = False
                    current_token = token
                    current_indices = [i]
            elif token == '/' and current_token:
                # Merge "/" for space groups (I4/mmm, P63/mmc)
                # Use upper() for case-insensitive detection
                if current_token[0].upper() in space_group_starters:
                    current_token += token
                    current_indices.append(i)
                    in_space_group = True
                else:
                    if current_token:
                        merged_tokens.append(current_token)
                        token_mapping.append(current_indices)
                    current_token = token
                    current_indices = [i]
            elif in_coordinate and token.isalpha():
                # Continue N-coordinate pattern (12-coordinate)
                current_token += token
                current_indices.append(i)
                in_coordinate = False  # End after the word
            elif in_space_group and (token.isdigit() or (token.lower() in space_group_chars) or
                                     (token.isalpha() and len(token) <= 2)):
                # Continue space group: merge numbers, m/n/c/a/b/d letters
                current_token += token
                current_indices.append(i)
            elif token == '(' and current_token:
                # Merge opening parentheses (e.g., for atom labels like Li(1))
                current_token += token
                current_indices.append(i)
                in_parentheses = True
            elif token == ')' and current_token and in_parentheses:
                # Merge closing parentheses
                current_token += token
                current_indices.append(i)
                in_parentheses = False
            elif in_parentheses and (token.isdigit() or token.isalpha()):
                # Merge content inside parentheses (digits or letters)
                current_token += token
                current_indices.append(i)
            elif token.isdigit() and current_token and not current_token[-1].isdigit():
                # Only merge numbers with atom symbols or short space group starters
                # NOT with regular words like "bonded"
                # Check both original case and capitalized for atom symbols
                # For space group starters, only merge if token is short (<=2 chars, like P, I, F, Pm)
                is_atom = current_token in atom_symbols or current_token.capitalize() in atom_symbols
                is_short_space_group = (len(current_token) <= 2 and current_token[0].upper() in space_group_starters)
                if is_atom or is_short_space_group:
                    current_token += token
                    current_indices.append(i)
                else:
                    # Save previous and start new with the digit
                    if current_token:
                        merged_tokens.append(current_token)
                        token_mapping.append(current_indices)
                        in_space_group = False
                    current_token = token
                    current_indices = [i]
            elif token == '.' and current_token:
                # Merge decimal points
                current_token += token
                current_indices.append(i)
            else:
                # Save previous token if exists
                if current_token:
                    merged_tokens.append(current_token)
                    token_mapping.append(current_indices)
                    in_space_group = False
                    in_coordinate = False
                    in_parentheses = False
                # Start new token
                current_token = token
                current_indices = [i]
                # Check if starting a space group (case-insensitive)
                if token.upper() in space_group_starters:
                    in_space_group = True
                else:
                    in_space_group = False

        # Don't forget the last token
        if current_token:
            merged_tokens.append(current_token)
            token_mapping.append(current_indices)

        # Merge weights by taking maximum over grouped indices
        # Using max preserves the strongest attention signal in merged tokens
        if weights is not None and len(weights.shape) >= 1:
            # Handle different weight shapes
            if len(weights.shape) == 1:
                # [seq_len]
                merged_weights = np.array([weights[indices].max() for indices in token_mapping])
            elif len(weights.shape) == 2:
                # [num_atoms, seq_len] or [seq_len, num_atoms]
                if weights.shape[-1] == len(tokens):
                    # Last dim is seq_len
                    merged_weights = np.array([[weights[i, indices].max() for indices in token_mapping]
                                               for i in range(weights.shape[0])])
                else:
                    # First dim is seq_len
                    merged_weights = np.array([[weights[indices, i].max() for indices in token_mapping]
                                               for i in range(weights.shape[1])]).T
            else:
                merged_weights = weights  # Don't merge for complex shapes
        else:
            merged_weights = weights

        return merged_tokens, merged_weights, token_mapping

    def is_stopword(self, word):
        """检查是否为停用词"""
        word_lower = word.lower().strip()
        # 检查是否在停用词列表中
        if word_lower in self.STOPWORDS or word in self.STOPWORDS:
            return True
        # 检查是否为WordPiece碎片（以##开头且长度小于4）
        if word.startswith('##') and len(word) < 5:
            return True
        # 检查是否为单个字符（除了元素符号）
        if len(word_lower) == 1 and not word_lower.isupper():
            return True
        return False

    def filter_stopwords_from_analysis(self, word_importance_pairs):
        """过滤分析结果中的停用词

        Args:
            word_importance_pairs: [(word, importance), ...] 列表

        Returns:
            过滤后的列表
        """
        return [(word, imp) for word, imp in word_importance_pairs
                if not self.is_stopword(word)]

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
        show_all_heads=False,
        filter_stopwords=True,
        merge_wordpiece=True
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
            filter_stopwords: 是否过滤停用词（默认True）
            merge_wordpiece: 是否合并WordPiece tokens用于显示（如F-43m）

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

        # Merge WordPiece tokens if requested
        original_tokens = text_tokens.copy() if isinstance(text_tokens, list) else list(text_tokens)
        if merge_wordpiece and atom_to_text_avg is not None:
            text_tokens, atom_to_text_avg, token_mapping = self._merge_tokens_and_weights(original_tokens, atom_to_text_avg)
            if text_to_atom_avg is not None:
                # text_to_atom_avg is [seq_len, num_atoms], merge along first dim
                merged_t2a = []
                for indices in token_mapping:
                    merged_t2a.append(text_to_atom_avg[indices, :].max(axis=0))
                text_to_atom_avg = np.array(merged_t2a)
            seq_len = len(text_tokens)
            print(f"   - Merged tokens: {seq_len} (from {len(original_tokens)} original)")

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
            # Show averaged attention with only top-k atoms and words
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))

            # Select top-k atoms and words based on importance
            if atom_to_text_avg is not None:
                # Calculate importance scores
                atom_importance = atom_to_text_avg.mean(axis=1)  # [num_atoms]
                word_importance = atom_to_text_avg.mean(axis=0)  # [seq_len]

                # Get top-k indices
                top_atom_indices = atom_importance.argsort()[-top_k_atoms:][::-1]
                top_word_indices = word_importance.argsort()[-top_k_words:][::-1]

                # Sort indices for better visualization
                top_atom_indices = np.sort(top_atom_indices)
                top_word_indices = np.sort(top_word_indices)

                # Extract sub-matrix
                atom_to_text_sub = atom_to_text_avg[np.ix_(top_atom_indices, top_word_indices)]

                # Get labels for selected items
                selected_atoms = [f"{elements[i]}-{i}" for i in top_atom_indices]
                selected_words = [text_tokens[i] if i < len(text_tokens) else f"[{i}]" for i in top_word_indices]

                # Atom-to-Text attention heatmap (filtered)
                sns.heatmap(
                    atom_to_text_sub,
                    xticklabels=selected_words,
                    yticklabels=selected_atoms,
                    cmap='YlOrRd',
                    ax=axes[0],
                    cbar=True,
                    annot=top_k_atoms <= 15 and top_k_words <= 15,
                    fmt='.2f' if top_k_atoms <= 15 and top_k_words <= 15 else None
                )
                axes[0].set_title(f'Atom → Text Attention\n(Top {top_k_atoms} atoms × Top {top_k_words} words)', fontsize=12)
                axes[0].set_xlabel('Text Tokens (High Importance)', fontsize=10)
                axes[0].set_ylabel('Atoms (High Importance)', fontsize=10)
                plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
                plt.setp(axes[0].get_yticklabels(), fontsize=9)

            # Text-to-Atom attention heatmap (filtered)
            if text_to_atom_avg is not None:
                # Calculate importance scores
                word_importance_t2a = text_to_atom_avg.mean(axis=1)  # [seq_len]
                atom_importance_t2a = text_to_atom_avg.mean(axis=0)  # [num_atoms]

                # Get top-k indices
                top_word_indices_t2a = word_importance_t2a.argsort()[-top_k_words:][::-1]
                top_atom_indices_t2a = atom_importance_t2a.argsort()[-top_k_atoms:][::-1]

                # Sort indices
                top_word_indices_t2a = np.sort(top_word_indices_t2a)
                top_atom_indices_t2a = np.sort(top_atom_indices_t2a)

                # Extract sub-matrix
                text_to_atom_sub = text_to_atom_avg[np.ix_(top_word_indices_t2a, top_atom_indices_t2a)]

                # Get labels
                selected_words_t2a = [text_tokens[i] if i < len(text_tokens) else f"[{i}]" for i in top_word_indices_t2a]
                selected_atoms_t2a = [f"{elements[i]}-{i}" for i in top_atom_indices_t2a]

                sns.heatmap(
                    text_to_atom_sub,
                    xticklabels=selected_atoms_t2a,
                    yticklabels=selected_words_t2a,
                    cmap='YlGnBu',
                    ax=axes[1],
                    cbar=True,
                    annot=top_k_atoms <= 15 and top_k_words <= 15,
                    fmt='.2f' if top_k_atoms <= 15 and top_k_words <= 15 else None
                )
                axes[1].set_title(f'Text → Atom Attention\n(Top {top_k_words} words × Top {top_k_atoms} atoms)', fontsize=12)
                axes[1].set_xlabel('Atoms (High Importance)', fontsize=10)
                axes[1].set_ylabel('Text Tokens (High Importance)', fontsize=10)
                plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
                plt.setp(axes[1].get_yticklabels(), fontsize=9)

            plt.suptitle('Fine-Grained Cross-Modal Attention (Filtered by Importance)', fontsize=14, fontweight='bold')
            plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 细粒度注意力可视化已保存: {save_path}")

        plt.close()

        # Analyze patterns
        analysis = {}

        if atom_to_text_avg is not None:
            # Top words attended by each atom (with stopword filtering)
            analysis['atom_top_words'] = {}
            for i, element in enumerate(elements):
                # 获取所有词的重要性
                all_words = [(text_tokens[idx], float(atom_to_text_avg[i, idx]))
                            for idx in range(len(text_tokens[:seq_len]))]
                # 按重要性排序
                all_words.sort(key=lambda x: x[1], reverse=True)
                # 过滤停用词
                if filter_stopwords:
                    filtered_words = self.filter_stopwords_from_analysis(all_words)
                else:
                    filtered_words = all_words
                analysis['atom_top_words'][f"{element}_{i}"] = filtered_words[:top_k_words]

            # Overall most important words (averaged over all atoms)
            word_importance = atom_to_text_avg.mean(axis=0)  # [seq_len]
            all_words = [(text_tokens[idx], float(word_importance[idx]))
                        for idx in range(len(text_tokens[:seq_len]))]
            all_words.sort(key=lambda x: x[1], reverse=True)
            # 过滤停用词
            if filter_stopwords:
                filtered_words = self.filter_stopwords_from_analysis(all_words)
            else:
                filtered_words = all_words
            analysis['overall_top_words'] = filtered_words[:top_k_words]

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

    def analyze_attention_head_specialization(
        self,
        attention_weights,
        atoms_object,
        text_tokens,
        save_path=None
    ):
        """
        分析注意力头的专业化模式

        Args:
            attention_weights: 细粒度注意力权重字典
            atoms_object: Atoms对象
            text_tokens: 文本tokens列表
            save_path: 保存路径（可选）

        Returns:
            分析结果字典，包含：
            - head_patterns: 每个头的主要关注模式
            - head_diversity: 头之间的多样性分数
            - head_importance: 每个头的重要性
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.spatial.distance import cosine
        from scipy.stats import entropy

        atom_to_text = attention_weights.get('atom_to_text', None)
        if atom_to_text is None:
            print("⚠️  没有atom_to_text注意力权重")
            return None

        # Convert to numpy and get single sample
        atom_to_text = atom_to_text.cpu().numpy()[0]  # [heads, num_atoms, seq_len]
        num_heads, num_atoms, seq_len = atom_to_text.shape

        # Get atom elements
        elements = [str(atoms_object.elements[i]) for i in range(num_atoms)]

        # Merge WordPiece tokens for better display
        # Use average over all heads for merging
        avg_attn = atom_to_text.mean(axis=0)  # [num_atoms, seq_len]
        merged_tokens, merged_weights, token_mapping = self._merge_tokens_and_weights(text_tokens, avg_attn)

        # Merge attention for each head
        merged_head_attn = []
        for head in range(num_heads):
            head_attn = atom_to_text[head]  # [num_atoms, seq_len]
            # Merge weights for this head
            merged_head = np.array([[head_attn[atom, indices].max() for indices in token_mapping]
                                    for atom in range(num_atoms)])
            merged_head_attn.append(merged_head)
        merged_head_attn = np.array(merged_head_attn)  # [num_heads, num_atoms, merged_seq_len]

        analysis = {
            'head_patterns': {},
            'head_diversity': 0.0,
            'head_importance': {},
            'head_entropy': {}
        }

        # Analyze each head
        head_vectors = []  # For diversity calculation

        for head in range(num_heads):
            head_attn = merged_head_attn[head]  # [num_atoms, merged_seq_len]

            # Calculate entropy (lower = more focused)
            head_entropy_val = entropy(head_attn.flatten() + 1e-10)
            analysis['head_entropy'][f'head_{head+1}'] = float(head_entropy_val)

            # Find top words for this head
            word_importance = head_attn.mean(axis=0)  # Average over atoms
            top_word_indices = word_importance.argsort()[-5:][::-1]
            top_words = [(merged_tokens[idx], float(word_importance[idx])) for idx in top_word_indices]

            # Find top atoms for this head
            atom_importance = head_attn.mean(axis=1)  # Average over words
            top_atom_indices = atom_importance.argsort()[-3:][::-1]
            top_atoms = [(f"{elements[idx]}_{idx}", float(atom_importance[idx])) for idx in top_atom_indices]

            # Overall importance (sum of attention)
            head_importance_score = float(head_attn.sum())
            analysis['head_importance'][f'head_{head+1}'] = head_importance_score

            # Store pattern
            analysis['head_patterns'][f'head_{head+1}'] = {
                'top_words': top_words,
                'top_atoms': top_atoms,
                'entropy': float(head_entropy_val),
                'focus_level': 'high' if head_entropy_val < np.median([analysis['head_entropy'][f'head_{h+1}'] for h in range(head+1)]) else 'low'
            }

            # Store flattened attention for diversity calculation
            head_vectors.append(head_attn.flatten())

        # Calculate head diversity (average pairwise cosine distance)
        head_vectors = np.array(head_vectors)
        diversity_scores = []
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                dist = cosine(head_vectors[i], head_vectors[j])
                diversity_scores.append(dist)

        analysis['head_diversity'] = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        # Visualize head specialization
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. Head importance bar chart
            heads = list(analysis['head_importance'].keys())
            importances = list(analysis['head_importance'].values())
            axes[0, 0].bar(range(len(heads)), importances, color='steelblue')
            axes[0, 0].set_xticks(range(len(heads)))
            axes[0, 0].set_xticklabels(heads, rotation=45)
            axes[0, 0].set_ylabel('Total Attention Weight')
            axes[0, 0].set_title('Attention Head Importance')
            axes[0, 0].grid(axis='y', alpha=0.3)

            # 2. Head entropy (focus level)
            entropies = list(analysis['head_entropy'].values())
            axes[0, 1].bar(range(len(heads)), entropies, color='coral')
            axes[0, 1].set_xticks(range(len(heads)))
            axes[0, 1].set_xticklabels(heads, rotation=45)
            axes[0, 1].set_ylabel('Entropy (lower = more focused)')
            axes[0, 1].set_title('Attention Head Focus Level')
            axes[0, 1].grid(axis='y', alpha=0.3)

            # 3. Top words per head heatmap
            top_words_matrix = np.zeros((num_heads, 5))
            word_labels = []
            for head in range(num_heads):
                head_key = f'head_{head+1}'
                top_words = analysis['head_patterns'][head_key]['top_words']
                for i, (word, score) in enumerate(top_words):
                    top_words_matrix[head, i] = score
                    if head == 0:
                        word_labels.append(word[:15])  # Truncate long words

            sns.heatmap(top_words_matrix, ax=axes[1, 0], cmap='YlOrRd',
                       xticklabels=word_labels, yticklabels=heads,
                       annot=True, fmt='.3f', cbar_kws={'label': 'Attention Weight'})
            axes[1, 0].set_title('Top Words per Head')
            axes[1, 0].set_xlabel('Words')
            axes[1, 0].set_ylabel('Attention Head')

            # 4. Head diversity visualization (similarity matrix)
            similarity_matrix = np.zeros((num_heads, num_heads))
            for i in range(num_heads):
                for j in range(num_heads):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity_matrix[i, j] = 1.0 - cosine(head_vectors[i], head_vectors[j])

            sns.heatmap(similarity_matrix, ax=axes[1, 1], cmap='coolwarm',
                       xticklabels=heads, yticklabels=heads,
                       annot=True, fmt='.2f', vmin=0, vmax=1,
                       cbar_kws={'label': 'Similarity (1-cosine distance)'})
            axes[1, 1].set_title(f'Head Similarity Matrix\nDiversity Score: {analysis["head_diversity"]:.3f}')
            axes[1, 1].set_xlabel('Attention Head')
            axes[1, 1].set_ylabel('Attention Head')

            plt.suptitle('Attention Head Specialization Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 注意力头专业化分析已保存: {save_path}")
            plt.close()

        return analysis

    def analyze_key_atom_word_pairs(
        self,
        attention_weights,
        atoms_object,
        text_tokens,
        top_k=20,
        save_path=None,
        filter_stopwords=True
    ):
        """
        识别最关键的原子-词语对及其语义类别

        Args:
            attention_weights: 细粒度注意力权重字典
            atoms_object: Atoms对象
            text_tokens: 文本tokens列表
            top_k: 返回top-k对
            save_path: 保存路径（可选）
            filter_stopwords: 是否过滤停用词（默认True）

        Returns:
            分析结果字典，包含：
            - top_pairs: 最强的原子-词语对
            - semantic_categories: 自动分类的语义类别
        """
        import matplotlib.pyplot as plt
        import numpy as np

        atom_to_text = attention_weights.get('atom_to_text', None)
        if atom_to_text is None:
            print("⚠️  没有atom_to_text注意力权重")
            return None

        # Convert to numpy and get single sample
        atom_to_text = atom_to_text.cpu().numpy()[0]  # [heads, num_atoms, seq_len]

        # Average over heads
        atom_to_text_avg = atom_to_text.mean(axis=0)  # [num_atoms, seq_len]

        # Merge WordPiece tokens for better display
        merged_tokens, merged_weights, token_mapping = self._merge_tokens_and_weights(text_tokens, atom_to_text_avg)
        text_tokens = merged_tokens
        atom_to_text_avg = merged_weights

        num_atoms, seq_len = atom_to_text_avg.shape

        # Get atom elements
        elements = [str(atoms_object.elements[i]) for i in range(num_atoms)]

        # Find top-k atom-word pairs (with optional stopword filtering)
        flat_attention = atom_to_text_avg.flatten()
        # 获取所有索引按注意力权重排序
        sorted_indices = flat_attention.argsort()[::-1]

        top_pairs = []
        for idx in sorted_indices:
            if len(top_pairs) >= top_k:
                break

            atom_idx = idx // seq_len
            word_idx = idx % seq_len
            word = text_tokens[word_idx]

            # 过滤停用词
            if filter_stopwords and self.is_stopword(word):
                continue

            atom_name = f"{elements[atom_idx]}_{atom_idx}"
            weight = float(atom_to_text_avg[atom_idx, word_idx])
            top_pairs.append({
                'atom': atom_name,
                'word': word,
                'weight': weight,
                'atom_idx': int(atom_idx),
                'word_idx': int(word_idx)
            })

        # Semantic categorization (rule-based with detailed categories)
        # Element identification
        element_keywords = set(['mg', 'sn', 'ge', 'o', 'na', 'ba', 'bi', 'si', 'al', 'fe', 'cu', 'zn',
                               'li', 'hf', 'k', 'ca', 'ti', 'v', 'cr', 'mn', 'co', 'ni', 'ga', 'as',
                               'se', 'br', 'rb', 'sr', 'y', 'zr', 'nb', 'mo', 'ag', 'cd', 'in', 'sb',
                               'te', 'cs', 'la', 'ce', 'pr', 'nd', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho',
                               'er', 'tm', 'yb', 'lu', 'ta', 'w', 're', 'os', 'ir', 'pt', 'au', 'hg',
                               'tl', 'pb', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 'pa', 'u', 'np', 'pu'])

        # Crystal structure and symmetry
        structure_keywords = set(['cubic', 'monoclinic', 'orthorhombic', 'hexagonal', 'tetragonal',
                                  'triclinic', 'rhombohedral', 'trigonal', 'symmetry', 'lattice', 'cell',
                                  'crystallizes', 'crystal', 'phase'])

        # Space group related
        space_group_keywords = set(['group', 'space', 'p-1', 'pm-3m', 'fm-3m', 'fd-3m', 'i4/mmm',
                                   'p63/mmc', 'r-3m', 'c2/m', 'cmcm', 'pnma', 'pbca'])

        # Bonding and coordination
        bonding_keywords = set(['bond', 'bonded', 'bonding', 'length', 'distance', 'å', 'coordination',
                               'coordinate', 'valence', 'oxidation'])

        # Geometry and polyhedra
        geometry_keywords = set(['geometry', 'octahedral', 'tetrahedral', 'planar', 'trigonal',
                                'square', 'pyramidal', 'bipyramidal', 'prismatic', 'antiprismatic',
                                'linear', 'bent', 'seesaw', 't-shaped'])

        # Polyhedral connectivity
        connectivity_keywords = set(['sharing', 'corner', 'edge', 'face', 'vertex', 'connected',
                                    'linked', 'bridging', 'terminal'])

        # Distortion and disorder
        distortion_keywords = set(['distorted', 'distortion', 'tilted', 'rotated', 'displaced',
                                  'disordered', 'ordered', 'regular', 'irregular'])

        # Framework and structural units
        framework_keywords = set(['framework', 'cluster', 'chain', 'layer', 'sheet', 'network',
                                 'cage', 'channel', 'pore', 'void', 'consists', 'composed'])

        # Equivalence and multiplicity
        equivalence_keywords = set(['equivalent', 'inequivalent', 'unique', 'distinct', 'identical',
                                   'similar', 'different'])

        # Composition and formula
        composition_keywords = set(['formula', 'composition', 'stoichiometry', 'ratio', 'content',
                                   'concentration', 'doping', 'substitution'])

        semantic_categories = {
            'element_identification': [],
            'structure_symmetry': [],
            'space_group': [],
            'bonding_coordination': [],
            'geometry_polyhedra': [],
            'connectivity': [],
            'distortion': [],
            'framework_units': [],
            'equivalence': [],
            'composition': [],
            'other': []
        }

        for pair in top_pairs:
            word_lower = pair['word'].lower().replace('#', '').replace('-', '')
            categorized = False

            # Check each category
            if word_lower in element_keywords or any(word_lower.startswith(elem) for elem in element_keywords if len(elem) <= 2):
                semantic_categories['element_identification'].append(pair)
                categorized = True
            elif word_lower in space_group_keywords or any(sg in pair['word'].lower() for sg in ['43m', '63', 'mmm', '3m']):
                semantic_categories['space_group'].append(pair)
                categorized = True
            elif word_lower in structure_keywords or 'crystal' in word_lower:
                semantic_categories['structure_symmetry'].append(pair)
                categorized = True
            elif word_lower in bonding_keywords or 'bond' in word_lower or 'coordinate' in word_lower:
                semantic_categories['bonding_coordination'].append(pair)
                categorized = True
            elif word_lower in distortion_keywords or 'distort' in word_lower:
                semantic_categories['distortion'].append(pair)
                categorized = True
            elif word_lower in geometry_keywords or 'hedral' in word_lower:
                semantic_categories['geometry_polyhedra'].append(pair)
                categorized = True
            elif word_lower in connectivity_keywords:
                semantic_categories['connectivity'].append(pair)
                categorized = True
            elif word_lower in framework_keywords:
                semantic_categories['framework_units'].append(pair)
                categorized = True
            elif word_lower in equivalence_keywords:
                semantic_categories['equivalence'].append(pair)
                categorized = True
            elif word_lower in composition_keywords:
                semantic_categories['composition'].append(pair)
                categorized = True

            if not categorized:
                semantic_categories['other'].append(pair)

        # Calculate category statistics
        category_stats = {}
        for cat, pairs in semantic_categories.items():
            if pairs:
                total_weight = sum(p['weight'] for p in pairs)
                category_stats[cat] = {
                    'count': len(pairs),
                    'total_weight': float(total_weight),
                    'percentage': float(len(pairs) / len(top_pairs) * 100)
                }

        analysis = {
            'top_pairs': top_pairs,
            'semantic_categories': semantic_categories,
            'category_stats': category_stats
        }

        # Visualize
        if save_path:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # 1. Top pairs bar chart
            pairs_labels = [f"{p['atom']}→{p['word'][:10]}" for p in top_pairs[:15]]
            pairs_weights = [p['weight'] for p in top_pairs[:15]]

            axes[0].barh(range(len(pairs_labels)), pairs_weights, color='steelblue')
            axes[0].set_yticks(range(len(pairs_labels)))
            axes[0].set_yticklabels(pairs_labels, fontsize=9)
            axes[0].set_xlabel('Attention Weight', fontsize=10)
            axes[0].set_title(f'Top {len(pairs_labels)} Atom-Word Pairs', fontsize=12, fontweight='bold')
            axes[0].invert_yaxis()
            axes[0].grid(axis='x', alpha=0.3)

            # 2. Semantic category pie chart
            cat_labels = []
            cat_sizes = []
            cat_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

            for i, (cat, stats) in enumerate(category_stats.items()):
                cat_name = cat.replace('_', ' ').title()
                cat_labels.append(f"{cat_name}\n({stats['count']} pairs)")
                cat_sizes.append(stats['percentage'])

            if cat_sizes:
                axes[1].pie(cat_sizes, labels=cat_labels, autopct='%1.1f%%',
                           colors=cat_colors[:len(cat_sizes)], startangle=90)
                axes[1].set_title('Semantic Category Distribution', fontsize=12, fontweight='bold')

            plt.suptitle('Key Atom-Word Pairs Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 关键原子-词语对分析已保存: {save_path}")
            plt.close()

        return analysis

    def analyze_attention_statistics(
        self,
        attention_weights,
        atoms_object,
        text_tokens,
        save_path=None
    ):
        """
        分析注意力分布的统计特性

        Args:
            attention_weights: 细粒度注意力权重字典
            atoms_object: Atoms对象
            text_tokens: 文本tokens列表
            save_path: 保存路径（可选）

        Returns:
            统计分析结果字典
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from scipy.stats import entropy

        atom_to_text = attention_weights.get('atom_to_text', None)
        if atom_to_text is None:
            print("⚠️  没有atom_to_text注意力权重")
            return None

        # Convert to numpy and get single sample
        atom_to_text = atom_to_text.cpu().numpy()[0]  # [heads, num_atoms, seq_len]
        num_heads, num_atoms, seq_len = atom_to_text.shape

        # Get atom elements
        elements = [str(atoms_object.elements[i]) for i in range(num_atoms)]

        # Calculate statistics
        stats = {
            'global_stats': {},
            'per_atom_stats': {},
            'per_head_stats': {}
        }

        # Global statistics
        all_attention = atom_to_text.flatten()
        stats['global_stats'] = {
            'mean': float(np.mean(all_attention)),
            'std': float(np.std(all_attention)),
            'min': float(np.min(all_attention)),
            'max': float(np.max(all_attention)),
            'median': float(np.median(all_attention)),
            'entropy': float(entropy(all_attention + 1e-10)),
            'sparsity': float(np.sum(all_attention < 0.01) / len(all_attention) * 100),  # % below threshold
            'effective_connections': float(np.sum(all_attention >= 0.01) / len(all_attention) * 100)
        }

        # Per-atom statistics (averaged over heads)
        atom_to_text_avg = atom_to_text.mean(axis=0)  # [num_atoms, seq_len]
        for i, element in enumerate(elements):
            atom_attn = atom_to_text_avg[i]
            stats['per_atom_stats'][f"{element}_{i}"] = {
                'entropy': float(entropy(atom_attn + 1e-10)),
                'max_attention': float(np.max(atom_attn)),
                'focus_level': 'high' if entropy(atom_attn + 1e-10) < stats['global_stats']['entropy'] else 'low',
                'top_word': text_tokens[np.argmax(atom_attn)],
                'top_word_weight': float(np.max(atom_attn))
            }

        # Per-head statistics
        for head in range(num_heads):
            head_attn = atom_to_text[head]
            stats['per_head_stats'][f'head_{head+1}'] = {
                'entropy': float(entropy(head_attn.flatten() + 1e-10)),
                'mean': float(np.mean(head_attn)),
                'sparsity': float(np.sum(head_attn < 0.01) / head_attn.size * 100)
            }

        # Visualize
        if save_path:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            # 1. Attention distribution histogram
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.hist(all_attention, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax1.axvline(stats['global_stats']['mean'], color='red', linestyle='--',
                       label=f"Mean: {stats['global_stats']['mean']:.4f}")
            ax1.axvline(stats['global_stats']['median'], color='green', linestyle='--',
                       label=f"Median: {stats['global_stats']['median']:.4f}")
            ax1.set_xlabel('Attention Weight')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Global Attention Distribution')
            ax1.legend()
            ax1.grid(alpha=0.3)

            # 2. Global stats table
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.axis('off')
            stats_text = f"""
Global Statistics:
─────────────────
Mean:     {stats['global_stats']['mean']:.4f}
Std:      {stats['global_stats']['std']:.4f}
Entropy:  {stats['global_stats']['entropy']:.2f}
Sparsity: {stats['global_stats']['sparsity']:.1f}%
Effective: {stats['global_stats']['effective_connections']:.1f}%
"""
            ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # 3. Per-atom entropy
            ax3 = fig.add_subplot(gs[1, :])
            atom_names = list(stats['per_atom_stats'].keys())
            atom_entropies = [stats['per_atom_stats'][a]['entropy'] for a in atom_names]
            colors = ['coral' if stats['per_atom_stats'][a]['focus_level'] == 'high' else 'steelblue'
                     for a in atom_names]

            ax3.bar(range(len(atom_names)), atom_entropies, color=colors)
            ax3.axhline(stats['global_stats']['entropy'], color='red', linestyle='--',
                       label=f"Global entropy: {stats['global_stats']['entropy']:.2f}")
            ax3.set_xticks(range(len(atom_names)))
            ax3.set_xticklabels(atom_names, rotation=45, ha='right')
            ax3.set_ylabel('Entropy (lower = more focused)')
            ax3.set_title('Per-Atom Attention Entropy (Coral = High Focus, Blue = Low Focus)')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)

            # 4. Per-head statistics
            ax4 = fig.add_subplot(gs[2, 0])
            head_names = list(stats['per_head_stats'].keys())
            head_entropies = [stats['per_head_stats'][h]['entropy'] for h in head_names]
            ax4.bar(range(len(head_names)), head_entropies, color='#99cc99')
            ax4.set_xticks(range(len(head_names)))
            ax4.set_xticklabels(head_names, rotation=45, ha='right')
            ax4.set_ylabel('Entropy')
            ax4.set_title('Per-Head Entropy')
            ax4.grid(axis='y', alpha=0.3)

            # 5. Per-head sparsity
            ax5 = fig.add_subplot(gs[2, 1])
            head_sparsity = [stats['per_head_stats'][h]['sparsity'] for h in head_names]
            ax5.bar(range(len(head_names)), head_sparsity, color='#ff9999')
            ax5.set_xticks(range(len(head_names)))
            ax5.set_xticklabels(head_names, rotation=45, ha='right')
            ax5.set_ylabel('Sparsity (%)')
            ax5.set_title('Per-Head Sparsity')
            ax5.grid(axis='y', alpha=0.3)

            # 6. Box plot for attention distribution
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.boxplot(all_attention, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
            ax6.set_ylabel('Attention Weight')
            ax6.set_title('Attention Distribution\n(Box Plot)')
            ax6.grid(axis='y', alpha=0.3)

            plt.suptitle('Attention Distribution Statistics', fontsize=14, fontweight='bold')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 注意力统计分析已保存: {save_path}")
            plt.close()

        return stats

    def analyze_text_semantic_regions(
        self,
        attention_weights,
        atoms_object,
        text,
        text_tokens,
        save_path=None
    ):
        """
        分析文本不同语义区域的重要性

        Args:
            attention_weights: 细粒度注意力权重字典
            atoms_object: Atoms对象
            text: 原始文本字符串
            text_tokens: 文本tokens列表
            save_path: 保存路径（可选）

        Returns:
            分析结果字典
        """
        import matplotlib.pyplot as plt
        import numpy as np

        atom_to_text = attention_weights.get('atom_to_text', None)
        if atom_to_text is None:
            print("⚠️  没有atom_to_text注意力权重")
            return None

        # Convert to numpy and get single sample
        atom_to_text = atom_to_text.cpu().numpy()[0]  # [heads, num_atoms, seq_len]

        # Average over heads and atoms
        word_importance = atom_to_text.mean(axis=(0, 1))  # [seq_len]

        # Simple sentence segmentation (by period, or fixed length)
        sentences = text.split('. ')

        # Map tokens to sentences
        token_to_sentence = []
        current_pos = 0
        current_sentence = 0

        for token in text_tokens:
            # Simple heuristic: check if we've moved to next sentence
            token_clean = token.replace('#', '').lower()
            if current_sentence < len(sentences) - 1:
                if sentences[current_sentence].lower().find(token_clean) == -1:
                    current_sentence += 1
            token_to_sentence.append(current_sentence)

        # Aggregate importance by sentence
        sentence_importance = {}
        for i, sent_idx in enumerate(token_to_sentence):
            if i < len(word_importance):
                if sent_idx not in sentence_importance:
                    sentence_importance[sent_idx] = []
                sentence_importance[sent_idx].append(word_importance[i])

        # Calculate sentence-level statistics
        regions = []
        for sent_idx in sorted(sentence_importance.keys()):
            if sent_idx < len(sentences):
                importance_scores = sentence_importance[sent_idx]
                regions.append({
                    'region_id': sent_idx,
                    'text': sentences[sent_idx][:100] + ('...' if len(sentences[sent_idx]) > 100 else ''),
                    'avg_importance': float(np.mean(importance_scores)),
                    'max_importance': float(np.max(importance_scores)),
                    'num_tokens': len(importance_scores),
                    'contribution': 'high' if np.mean(importance_scores) > word_importance.mean() else 'medium'
                })

        # Sort by importance
        regions = sorted(regions, key=lambda x: x['avg_importance'], reverse=True)

        analysis = {
            'regions': regions,
            'word_importance': word_importance.tolist(),
            'text_tokens': text_tokens
        }

        # Visualize
        if save_path:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 1. Token-level importance over sequence
            axes[0].plot(range(len(word_importance)), word_importance, color='steelblue', linewidth=2)
            axes[0].fill_between(range(len(word_importance)), word_importance, alpha=0.3)
            axes[0].axhline(word_importance.mean(), color='red', linestyle='--',
                          label=f'Mean: {word_importance.mean():.4f}')
            axes[0].set_xlabel('Token Position', fontsize=10)
            axes[0].set_ylabel('Attention Weight', fontsize=10)
            axes[0].set_title('Token-Level Importance Over Sequence', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # 2. Sentence/region importance
            if regions:
                region_labels = [f"Region {r['region_id']+1}" for r in regions[:10]]
                region_scores = [r['avg_importance'] for r in regions[:10]]
                colors_map = {'high': 'coral', 'medium': 'steelblue', 'low': 'lightgray'}
                colors = [colors_map.get(r['contribution'], 'lightgray') for r in regions[:10]]

                axes[1].barh(range(len(region_labels)), region_scores, color=colors)
                axes[1].set_yticks(range(len(region_labels)))
                axes[1].set_yticklabels(region_labels, fontsize=9)
                axes[1].set_xlabel('Average Attention Weight', fontsize=10)
                axes[1].set_title('Semantic Region Importance', fontsize=12, fontweight='bold')
                axes[1].invert_yaxis()
                axes[1].grid(axis='x', alpha=0.3)

                # Add text preview on the right
                for i, region in enumerate(regions[:10]):
                    text_preview = region['text'][:40] + '...'
                    axes[1].text(region_scores[i] + 0.001, i, text_preview,
                               va='center', fontsize=7, style='italic')

            plt.suptitle('Text Semantic Region Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 文本语义区域分析已保存: {save_path}")
            plt.close()

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
