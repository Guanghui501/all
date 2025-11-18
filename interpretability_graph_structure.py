"""
图结构可解释性分析 - 超越单原子分析

本模块提供更全面的图结构分析，包括：
1. 边重要性分析（化学键、原子间距离）
2. 角度/三元组重要性分析（ALIGNN特有）
3. 局部配位环境分析
4. 子结构/基序识别
5. 结构层次化分析

作者: Graph Structure Interpretability Module
日期: 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import networkx as nx
from collections import defaultdict, Counter
import json


class GraphStructureAnalyzer:
    """图结构可解释性分析器 - 超越单原子分析"""

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: 训练好的 ALIGNN 模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()

        print(f"\n🔬 图结构分析器初始化")
        print(f"   设备: {device}")

    def compute_edge_importance(self, g, lg, text, method='gradient'):
        """
        计算边（化学键）的重要性

        Args:
            g: DGL graph
            lg: Line graph
            text: 文本
            method: 'gradient' 或 'integrated_gradients'

        Returns:
            edge_importance: [num_edges] 边重要性分数
            edge_info: 边的详细信息（原子对、距离等）
        """
        self.model.eval()
        g = g.to(self.device)
        lg = lg.to(self.device)

        print(f"\n🔗 计算边重要性...")
        print(f"   边数量: {g.num_edges()}")

        # 获取边特征并启用梯度
        edge_features = g.edata['r'].clone().detach().requires_grad_(True)
        original_edge_features = g.edata['r']
        g.edata['r'] = edge_features

        # Forward
        output = self.model([g, lg, text])

        if isinstance(output, dict):
            prediction = output['predictions']
        else:
            prediction = output

        # Backward
        loss = prediction.sum()
        loss.backward()

        # 计算边重要性（梯度的L2范数）
        edge_gradients = edge_features.grad
        # 处理不同维度的梯度
        if edge_gradients.dim() == 1:
            edge_importance = torch.abs(edge_gradients).cpu().numpy()
        else:
            edge_importance = torch.norm(edge_gradients, dim=1).cpu().numpy()

        # 恢复
        g.edata['r'] = original_edge_features

        # 提取边信息
        src, dst = g.edges()
        edge_info = []

        for i, (s, d) in enumerate(zip(src.cpu().numpy(), dst.cpu().numpy())):
            # 计算边长（原子间距离）
            edge_vector = original_edge_features[i].cpu().numpy()
            distance = np.linalg.norm(edge_vector)

            edge_info.append({
                'edge_id': i,
                'src_atom': int(s),
                'dst_atom': int(d),
                'distance': float(distance),
                'importance': float(edge_importance[i]),
                'vector': edge_vector.tolist()
            })

        return edge_importance, edge_info

    def compute_angle_importance(self, g, lg, text):
        """
        计算角度/三元组的重要性（ALIGNN特有）

        Line graph 的节点对应原始图的边
        Line graph 的边对应原始图的角度（三元组）

        Args:
            g: DGL graph
            lg: Line graph
            text: 文本

        Returns:
            angle_importance: [num_angles] 角度重要性
            angle_info: 角度详细信息
        """
        self.model.eval()
        g = g.to(self.device)
        lg = lg.to(self.device)

        print(f"\n📐 计算角度/三元组重要性...")
        print(f"   Line graph 节点数: {lg.num_nodes()} (= 原图边数)")
        print(f"   Line graph 边数: {lg.num_edges()} (= 角度数)")

        # Line graph 的边特征就是角度信息
        if 'h' not in lg.edata:
            print("⚠️  Line graph 没有边特征，跳过角度分析")
            return None, None

        angle_features = lg.edata['h'].clone().detach().requires_grad_(True)
        original_angle_features = lg.edata['h']
        lg.edata['h'] = angle_features

        # Forward
        output = self.model([g, lg, text])

        if isinstance(output, dict):
            prediction = output['predictions']
        else:
            prediction = output

        # Backward
        loss = prediction.sum()
        loss.backward()

        # 计算角度重要性
        angle_gradients = angle_features.grad
        # 处理不同维度的梯度
        if angle_gradients.dim() == 1:
            angle_importance = torch.abs(angle_gradients).cpu().numpy()
        else:
            angle_importance = torch.norm(angle_gradients, dim=1).cpu().numpy()

        # 恢复
        lg.edata['h'] = original_angle_features

        # 提取角度信息
        src_edges, dst_edges = lg.edges()
        angle_info = []

        # Line graph 的边连接两条原图的边，这两条边共享一个原子，形成角度
        for i, (e1, e2) in enumerate(zip(src_edges.cpu().numpy(), dst_edges.cpu().numpy())):
            angle_info.append({
                'angle_id': i,
                'edge1': int(e1),
                'edge2': int(e2),
                'importance': float(angle_importance[i])
            })

        return angle_importance, angle_info

    def analyze_coordination_environment(self, g, atoms_object, atom_importance):
        """
        分析局部配位环境的重要性

        Args:
            g: DGL graph
            atoms_object: Atoms对象
            atom_importance: 原子重要性分数

        Returns:
            coord_analysis: 配位环境分析结果
        """
        print(f"\n🔮 分析配位环境...")

        # 转换为NetworkX图便于分析
        nx_g = self._dgl_to_networkx(g)

        coord_analysis = []
        elements = list(atoms_object.elements)

        for atom_idx in range(g.num_nodes()):
            # 获取邻居
            neighbors = list(nx_g.neighbors(atom_idx))

            # 配位数
            coord_num = len(neighbors)

            # 邻居元素
            neighbor_elements = [elements[n] for n in neighbors]
            neighbor_counts = Counter(neighbor_elements)

            # 邻居的平均重要性
            neighbor_importance = [atom_importance[n] for n in neighbors]
            avg_neighbor_importance = np.mean(neighbor_importance) if neighbors else 0.0

            # 配位环境描述
            coord_env = {
                'atom_id': atom_idx,
                'element': elements[atom_idx],
                'atom_importance': float(atom_importance[atom_idx]),
                'coordination_number': coord_num,
                'neighbor_elements': dict(neighbor_counts),
                'neighbor_importance_avg': float(avg_neighbor_importance),
                'neighbor_importance_max': float(np.max(neighbor_importance)) if neighbors else 0.0,
                'neighbors': neighbors
            }

            coord_analysis.append(coord_env)

        return coord_analysis

    def identify_important_substructures(
        self,
        g,
        atoms_object,
        atom_importance,
        edge_importance,
        subgraph_size=3,
        top_k=10
    ):
        """
        识别重要的子结构/基序

        Args:
            g: DGL graph
            atoms_object: Atoms对象
            atom_importance: 原子重要性
            edge_importance: 边重要性
            subgraph_size: 子图大小
            top_k: 返回top-k个子结构

        Returns:
            important_substructures: 重要子结构列表
        """
        print(f"\n🧩 识别重要子结构（大小={subgraph_size}）...")

        # 转换为NetworkX
        nx_g = self._dgl_to_networkx(g)
        elements = list(atoms_object.elements)

        # 枚举所有大小为subgraph_size的连通子图
        substructures = []

        # 使用k-hop邻居生成子图
        for center_node in range(g.num_nodes()):
            # BFS获取k-hop邻居
            subgraph_nodes = self._get_k_hop_neighbors(
                nx_g, center_node, k=subgraph_size-1
            )

            if len(subgraph_nodes) >= subgraph_size:
                # 限制为subgraph_size个节点
                subgraph_nodes = sorted(subgraph_nodes)[:subgraph_size]

                # 计算子图的总重要性
                subgraph_atom_importance = sum([atom_importance[n] for n in subgraph_nodes])

                # 计算子图内部边的重要性
                subgraph = nx_g.subgraph(subgraph_nodes)
                subgraph_edge_importance = 0.0

                for u, v in subgraph.edges():
                    # 找到对应的DGL边
                    edge_ids = g.edge_ids(u, v)
                    # Handle different return types from edge_ids (int, scalar tensor, or 1D tensor)
                    if hasattr(edge_ids, 'dim'):
                        # It's a tensor
                        if edge_ids.dim() == 0:
                            # Scalar tensor
                            edge_id = edge_ids.item()
                        else:
                            # 1D tensor
                            edge_id = edge_ids[0].item()
                    elif isinstance(edge_ids, int):
                        # Already an int
                        edge_id = edge_ids
                    else:
                        # List or other sequence
                        edge_id = edge_ids[0]
                    subgraph_edge_importance += edge_importance[edge_id]

                # 总重要性
                total_importance = subgraph_atom_importance + subgraph_edge_importance

                # 子结构签名（用于去重）
                signature = tuple(sorted([elements[n] for n in subgraph_nodes]))

                substructures.append({
                    'nodes': subgraph_nodes,
                    'elements': [elements[n] for n in subgraph_nodes],
                    'signature': signature,
                    'total_importance': total_importance,
                    'atom_importance': subgraph_atom_importance,
                    'edge_importance': subgraph_edge_importance,
                    'num_edges': subgraph.number_of_edges()
                })

        # 按总重要性排序
        substructures.sort(key=lambda x: x['total_importance'], reverse=True)

        # 去重（保留重要性最高的）
        seen_signatures = set()
        unique_substructures = []

        for sub in substructures:
            if sub['signature'] not in seen_signatures:
                seen_signatures.add(sub['signature'])
                unique_substructures.append(sub)

            if len(unique_substructures) >= top_k:
                break

        print(f"   发现 {len(unique_substructures)} 个独特的重要子结构")

        return unique_substructures

    def _dgl_to_networkx(self, g):
        """将DGL图转换为NetworkX图"""
        import networkx as nx

        nx_g = nx.Graph()

        # 添加节点
        nx_g.add_nodes_from(range(g.num_nodes()))

        # 添加边
        src, dst = g.edges()
        edges = list(zip(src.cpu().numpy(), dst.cpu().numpy()))
        nx_g.add_edges_from(edges)

        return nx_g

    def _get_k_hop_neighbors(self, nx_g, center, k):
        """获取k-hop邻居"""
        neighbors = {center}
        current_layer = {center}

        for _ in range(k):
            next_layer = set()
            for node in current_layer:
                next_layer.update(nx_g.neighbors(node))
            neighbors.update(next_layer)
            current_layer = next_layer

        return list(neighbors)

    def visualize_edge_importance(
        self,
        g,
        atoms_object,
        edge_importance,
        edge_info,
        save_path=None,
        top_k=20
    ):
        """
        可视化边重要性

        Args:
            g: DGL graph
            atoms_object: Atoms对象
            edge_importance: 边重要性分数
            edge_info: 边信息
            save_path: 保存路径
            top_k: 显示top-k重要的边
        """
        # 归一化
        edge_importance_norm = (edge_importance - edge_importance.min()) / \
                              (edge_importance.max() - edge_importance.min() + 1e-8)

        # 创建DataFrame
        df = pd.DataFrame(edge_info)
        df['importance_norm'] = edge_importance_norm

        # 添加元素信息
        elements = list(atoms_object.elements)
        df['src_element'] = df['src_atom'].apply(lambda x: elements[x])
        df['dst_element'] = df['dst_atom'].apply(lambda x: elements[x])
        df['bond_type'] = df.apply(lambda x: f"{x['src_element']}-{x['dst_element']}", axis=1)

        df = df.sort_values('importance_norm', ascending=False).reset_index(drop=True)

        # 打印Top-k
        print(f"\n{'='*80}")
        print(f"Top {top_k} Most Important Edges (Chemical Bonds)")
        print(f"{'='*80}")
        print(df.head(top_k)[['edge_id', 'bond_type', 'distance', 'importance_norm']].to_string(index=False))
        print(f"{'='*80}\n")

        # 可视化
        fig = plt.figure(figsize=(16, 5))

        # 1. 边重要性分布
        ax1 = fig.add_subplot(131)
        ax1.hist(edge_importance_norm, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(edge_importance_norm.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {edge_importance_norm.mean():.3f}')
        ax1.set_xlabel('Edge Importance Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Edge Importance Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 按键类型统计
        ax2 = fig.add_subplot(132)
        bond_importance = df.groupby('bond_type')['importance_norm'].agg(['mean', 'count'])
        bond_importance = bond_importance[bond_importance['count'] >= 3]  # 至少3条边
        bond_importance = bond_importance.sort_values('mean', ascending=False).head(15)

        y_pos = np.arange(len(bond_importance))
        ax2.barh(y_pos, bond_importance['mean'].values,
                color=plt.cm.viridis(np.linspace(0, 1, len(bond_importance))))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{bt} (n={int(c)})"
                             for bt, c in zip(bond_importance.index, bond_importance['count'])])
        ax2.set_xlabel('Average Importance', fontsize=11)
        ax2.set_title('Importance by Bond Type', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. 距离 vs 重要性
        ax3 = fig.add_subplot(133)
        scatter = ax3.scatter(df['distance'], df['importance_norm'],
                            c=df['importance_norm'], s=50,
                            cmap='YlOrRd', alpha=0.6, edgecolors='black', linewidth=0.5)

        # 高亮top-10边
        top_edges = df.head(10)
        ax3.scatter(top_edges['distance'], top_edges['importance_norm'],
                   s=200, edgecolors='red', facecolors='none', linewidth=2,
                   label='Top-10 Edges')

        ax3.set_xlabel('Bond Distance (Å)', fontsize=11)
        ax3.set_ylabel('Importance Score', fontsize=11)
        ax3.set_title('Distance vs Importance', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 边重要性可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

        return df

    def visualize_coordination_environment(
        self,
        coord_analysis,
        save_path=None,
        top_k=15
    ):
        """
        可视化配位环境分析

        Args:
            coord_analysis: 配位环境分析结果
            save_path: 保存路径
            top_k: 显示top-k重要的配位环境
        """
        df = pd.DataFrame(coord_analysis)
        df = df.sort_values('atom_importance', ascending=False).reset_index(drop=True)

        # 打印Top-k
        print(f"\n{'='*80}")
        print(f"Top {top_k} Most Important Coordination Environments")
        print(f"{'='*80}")
        display_cols = ['atom_id', 'element', 'coordination_number',
                       'atom_importance', 'neighbor_importance_avg']
        print(df.head(top_k)[display_cols].to_string(index=False))
        print(f"{'='*80}\n")

        # 可视化
        fig = plt.figure(figsize=(16, 5))

        # 1. 配位数分布
        ax1 = fig.add_subplot(131)
        coord_counts = df['coordination_number'].value_counts().sort_index()
        ax1.bar(coord_counts.index, coord_counts.values,
               color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Coordination Number', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Coordination Number Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 配位数 vs 重要性
        ax2 = fig.add_subplot(132)
        coord_importance = df.groupby('coordination_number')['atom_importance'].agg(['mean', 'std', 'count'])
        coord_importance = coord_importance[coord_importance['count'] >= 3]

        x = coord_importance.index
        y = coord_importance['mean']
        yerr = coord_importance['std']

        ax2.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5,
                    color='darkblue', markersize=8, linewidth=2)
        ax2.set_xlabel('Coordination Number', fontsize=11)
        ax2.set_ylabel('Average Atom Importance', fontsize=11)
        ax2.set_title('Coordination Number vs Importance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. 元素-配位数矩阵
        ax3 = fig.add_subplot(133)
        element_coord = df.groupby(['element', 'coordination_number'])['atom_importance'].mean().unstack(fill_value=0)

        # 只显示主要元素
        main_elements = df['element'].value_counts().head(8).index
        element_coord = element_coord.loc[main_elements]

        sns.heatmap(element_coord, cmap='YlOrRd', annot=True, fmt='.2f',
                   cbar_kws={'label': 'Avg Importance'}, ax=ax3)
        ax3.set_xlabel('Coordination Number', fontsize=11)
        ax3.set_ylabel('Element', fontsize=11)
        ax3.set_title('Element-Coordination Matrix', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 配位环境可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

        return df

    def visualize_substructures(
        self,
        substructures,
        save_path=None
    ):
        """
        可视化重要子结构

        Args:
            substructures: 重要子结构列表
            save_path: 保存路径
        """
        if not substructures:
            print("⚠️  没有发现重要子结构")
            return

        print(f"\n{'='*80}")
        print(f"Top {len(substructures)} Most Important Substructures")
        print(f"{'='*80}")

        for i, sub in enumerate(substructures[:10], 1):
            motif = '-'.join(sub['elements'])
            print(f"{i}. {motif}")
            print(f"   节点: {sub['nodes']}")
            print(f"   总重要性: {sub['total_importance']:.4f}")
            print(f"   边数: {sub['num_edges']}")
            print()

        print(f"{'='*80}\n")

        # 可视化
        fig = plt.figure(figsize=(16, 5))

        # 1. 子结构类型分布
        ax1 = fig.add_subplot(131)
        motif_counts = Counter(['-'.join(sorted(sub['elements'])) for sub in substructures])
        top_motifs = dict(motif_counts.most_common(15))

        y_pos = np.arange(len(top_motifs))
        ax1.barh(y_pos, list(top_motifs.values()),
                color=plt.cm.viridis(np.linspace(0, 1, len(top_motifs))))
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(list(top_motifs.keys()), fontsize=9)
        ax1.set_xlabel('Frequency', fontsize=11)
        ax1.set_title('Most Common Substructure Motifs', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. 重要性分布
        ax2 = fig.add_subplot(132)
        importances = [sub['total_importance'] for sub in substructures]
        ax2.hist(importances, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(importances), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(importances):.3f}')
        ax2.set_xlabel('Total Importance', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Substructure Importance Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Top-10 子结构重要性分解
        ax3 = fig.add_subplot(133)
        top_10 = substructures[:10]
        x = np.arange(len(top_10))
        width = 0.35

        atom_imp = [s['atom_importance'] for s in top_10]
        edge_imp = [s['edge_importance'] for s in top_10]

        ax3.bar(x - width/2, atom_imp, width, label='Atom Importance',
               color='skyblue', edgecolor='black')
        ax3.bar(x + width/2, edge_imp, width, label='Edge Importance',
               color='salmon', edgecolor='black')

        ax3.set_xlabel('Substructure Rank', fontsize=11)
        ax3.set_ylabel('Importance', fontsize=11)
        ax3.set_title('Top-10 Substructures: Atom vs Edge', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"#{i+1}" for i in range(len(top_10))])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 子结构可视化已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def comprehensive_structure_analysis(
        self,
        g, lg, text,
        atoms_object,
        atom_importance,
        save_dir=None,
        sample_id='sample'
    ):
        """
        综合图结构分析

        Args:
            g, lg, text: 模型输入
            atoms_object: Atoms对象
            atom_importance: 原子重要性
            save_dir: 保存目录
            sample_id: 样本ID

        Returns:
            analysis: 完整分析结果
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🔬 样本 {sample_id} 的综合图结构分析")
        print(f"{'='*80}")

        analysis = {
            'sample_id': sample_id,
            'num_atoms': g.num_nodes(),
            'num_edges': g.num_edges()
        }

        # 1. 边重要性
        edge_importance, edge_info = self.compute_edge_importance(g, lg, text)
        analysis['edge_importance'] = edge_importance.tolist()
        analysis['edge_info'] = edge_info

        edge_df = self.visualize_edge_importance(
            g, atoms_object, edge_importance, edge_info,
            save_path=save_dir / f'{sample_id}_edge_importance.png' if save_dir else None
        )

        # 2. 角度重要性
        angle_importance, angle_info = self.compute_angle_importance(g, lg, text)
        if angle_importance is not None:
            analysis['angle_importance'] = angle_importance.tolist()
            analysis['angle_info'] = angle_info

        # 3. 配位环境
        coord_analysis = self.analyze_coordination_environment(g, atoms_object, atom_importance)
        analysis['coordination_environments'] = coord_analysis

        coord_df = self.visualize_coordination_environment(
            coord_analysis,
            save_path=save_dir / f'{sample_id}_coordination.png' if save_dir else None
        )

        # 4. 重要子结构
        substructures = self.identify_important_substructures(
            g, atoms_object, atom_importance, edge_importance,
            subgraph_size=3, top_k=20
        )
        analysis['important_substructures'] = substructures

        self.visualize_substructures(
            substructures,
            save_path=save_dir / f'{sample_id}_substructures.png' if save_dir else None
        )

        # 5. 保存分析结果
        if save_dir:
            with open(save_dir / f'{sample_id}_structure_analysis.json', 'w') as f:
                # 移除不可JSON序列化的部分
                json_safe = {
                    'sample_id': analysis['sample_id'],
                    'num_atoms': analysis['num_atoms'],
                    'num_edges': analysis['num_edges'],
                    'top_edges': edge_df.head(20).to_dict('records'),
                    'top_coord_envs': coord_df.head(20).to_dict('records'),
                    'top_substructures': substructures[:10]
                }
                json.dump(json_safe, f, indent=2)
            print(f"\n✅ 结构分析结果已保存到: {save_dir}")

        print(f"\n{'='*80}\n")

        return analysis


# 便捷函数
def analyze_graph_structure(
    model, g, lg, text,
    atoms_object,
    atom_importance,
    save_dir='./structure_analysis',
    sample_id='sample'
):
    """
    一键执行完整的图结构分析

    Args:
        model: 训练好的模型
        g, lg, text: 模型输入
        atoms_object: Atoms对象
        atom_importance: 原子重要性
        save_dir: 保存目录
        sample_id: 样本ID

    Returns:
        analysis: 完整分析结果
    """
    analyzer = GraphStructureAnalyzer(model)

    analysis = analyzer.comprehensive_structure_analysis(
        g, lg, text,
        atoms_object,
        atom_importance,
        save_dir=save_dir,
        sample_id=sample_id
    )

    return analysis
