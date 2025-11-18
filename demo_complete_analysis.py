#!/usr/bin/env python
"""
完整可解释性分析演示 - 超越单原子分析

展示如何综合分析：
1. 原子重要性
2. 边（化学键）重要性
3. 角度/三元组重要性
4. 配位环境
5. 子结构/基序
6. 跨模态注意力

提供材料属性预测的全方位解释
"""

import torch
import numpy as np
from pathlib import Path

from interpretability_enhanced import EnhancedInterpretabilityAnalyzer
from interpretability_graph_structure import GraphStructureAnalyzer


def complete_interpretability_analysis(
    model,
    g, lg, text,
    atoms_object,
    true_value=None,
    save_dir='./complete_analysis',
    sample_id='sample'
):
    """
    完整的可解释性分析流程

    Args:
        model: 训练好的模型
        g, lg, text: 模型输入
        atoms_object: Atoms对象
        true_value: 真实值
        save_dir: 保存目录
        sample_id: 样本ID

    Returns:
        complete_report: 完整分析报告
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    print("\n" + "="*80)
    print("🔬 完整可解释性分析")
    print("="*80)
    print(f"样本ID: {sample_id}")
    print(f"原子数: {atoms_object.num_atoms}")
    print(f"化学式: {atoms_object.composition.reduced_formula}")
    print("="*80 + "\n")

    # ========== 第1部分: 预测和跨模态注意力 ==========
    print("【第1部分】预测和跨模态注意力分析")
    print("-" * 80)

    analyzer = EnhancedInterpretabilityAnalyzer(model, device=device)

    # 提取注意力和预测
    result = analyzer.extract_attention_weights(g, lg, text, return_prediction=True)

    prediction = result['prediction'][0] if len(result['prediction'].shape) > 0 else result['prediction']

    print(f"\n📊 预测结果:")
    print(f"   预测值: {prediction:.4f}")
    if true_value is not None:
        error = abs(prediction - true_value)
        print(f"   真实值: {true_value:.4f}")
        print(f"   误差: {error:.4f}")
        print(f"   相对误差: {100*error/abs(true_value):.2f}%")

    # 可视化注意力
    if result['attention_weights'] is not None:
        print(f"\n🔍 跨模态注意力:")
        attn = result['attention_weights']

        if 'graph_to_text' in attn and attn['graph_to_text'] is not None:
            g2t_mean = attn['graph_to_text'].mean().item()
            print(f"   Graph→Text: {g2t_mean:.4f} (图对文本的依赖)")

        if 'text_to_graph' in attn and attn['text_to_graph'] is not None:
            t2g_mean = attn['text_to_graph'].mean().item()
            print(f"   Text→Graph: {t2g_mean:.4f} (文本对图的依赖)")

        analyzer.visualize_cross_modal_attention(
            result['attention_weights'],
            save_path=save_dir / f'{sample_id}_attention.png'
        )

        analyzer.visualize_attention_by_heads(
            result['attention_weights'],
            save_path=save_dir / f'{sample_id}_attention_heads.png'
        )

    # ========== 第2部分: 原子重要性 ==========
    print("\n" + "="*80)
    print("【第2部分】原子重要性分析")
    print("-" * 80)

    # 梯度法
    atom_importance = analyzer.compute_atom_importance(g, lg, text, method='gradient')

    print(f"\n📊 原子重要性统计:")
    print(f"   平均值: {atom_importance.mean():.4f}")
    print(f"   标准差: {atom_importance.std():.4f}")
    print(f"   范围: [{atom_importance.min():.4f}, {atom_importance.max():.4f}]")

    # 可视化
    atom_df = analyzer.visualize_atom_importance(
        atoms_object,
        atom_importance,
        save_path=save_dir / f'{sample_id}_atom_importance.png',
        top_k=10
    )

    # ========== 第3部分: 边（化学键）重要性 ==========
    print("\n" + "="*80)
    print("【第3部分】边（化学键）重要性分析")
    print("-" * 80)

    struct_analyzer = GraphStructureAnalyzer(model, device=device)

    edge_importance, edge_info = struct_analyzer.compute_edge_importance(g, lg, text)

    print(f"\n📊 边重要性统计:")
    print(f"   边数量: {len(edge_importance)}")
    print(f"   平均值: {edge_importance.mean():.4f}")
    print(f"   标准差: {edge_importance.std():.4f}")

    edge_df = struct_analyzer.visualize_edge_importance(
        g, atoms_object, edge_importance, edge_info,
        save_path=save_dir / f'{sample_id}_edge_importance.png',
        top_k=20
    )

    # ========== 第4部分: 角度/三元组重要性 ==========
    print("\n" + "="*80)
    print("【第4部分】角度/三元组重要性分析（ALIGNN特有）")
    print("-" * 80)

    angle_importance, angle_info = struct_analyzer.compute_angle_importance(g, lg, text)

    if angle_importance is not None:
        print(f"\n📊 角度重要性统计:")
        print(f"   角度数量: {len(angle_importance)}")
        print(f"   平均值: {angle_importance.mean():.4f}")
        print(f"   标准差: {angle_importance.std():.4f}")
    else:
        print("\n⚠️  角度信息不可用")

    # ========== 第5部分: 配位环境分析 ==========
    print("\n" + "="*80)
    print("【第5部分】配位环境分析")
    print("-" * 80)

    coord_analysis = struct_analyzer.analyze_coordination_environment(
        g, atoms_object, atom_importance
    )

    coord_df = struct_analyzer.visualize_coordination_environment(
        coord_analysis,
        save_path=save_dir / f'{sample_id}_coordination.png',
        top_k=15
    )

    # 配位环境统计
    coord_nums = [c['coordination_number'] for c in coord_analysis]
    print(f"\n📊 配位环境统计:")
    print(f"   平均配位数: {np.mean(coord_nums):.2f}")
    print(f"   配位数范围: [{min(coord_nums)}, {max(coord_nums)}]")

    # ========== 第6部分: 重要子结构识别 ==========
    print("\n" + "="*80)
    print("【第6部分】重要子结构/基序识别")
    print("-" * 80)

    substructures = struct_analyzer.identify_important_substructures(
        g, atoms_object, atom_importance, edge_importance,
        subgraph_size=3, top_k=20
    )

    struct_analyzer.visualize_substructures(
        substructures,
        save_path=save_dir / f'{sample_id}_substructures.png'
    )

    # ========== 第7部分: 特征空间（如果有多样本）==========
    if result['graph_features'] is not None and result['text_features'] is not None:
        print("\n" + "="*80)
        print("【第7部分】特征空间分析")
        print("-" * 80)
        print("   （需要多个样本才能有效可视化）")

    # ========== 生成综合报告 ==========
    print("\n" + "="*80)
    print("📝 生成综合分析报告")
    print("="*80)

    # 综合洞察
    insights = generate_insights(
        prediction, true_value,
        atom_importance, atom_df,
        edge_importance, edge_df,
        coord_analysis,
        substructures,
        result['attention_weights']
    )

    # 保存文本报告
    report_path = save_dir / f'{sample_id}_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"完整可解释性分析报告 - {sample_id}\n")
        f.write("="*80 + "\n\n")

        f.write(f"样本信息:\n")
        f.write(f"  化学式: {atoms_object.composition.reduced_formula}\n")
        f.write(f"  原子数: {atoms_object.num_atoms}\n")
        f.write(f"  边数: {g.num_edges()}\n\n")

        f.write(f"预测结果:\n")
        f.write(f"  预测值: {prediction:.4f}\n")
        if true_value is not None:
            f.write(f"  真实值: {true_value:.4f}\n")
            f.write(f"  误差: {abs(prediction - true_value):.4f}\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("关键洞察\n")
        f.write("="*80 + "\n\n")

        for i, insight in enumerate(insights, 1):
            f.write(f"{i}. {insight}\n\n")

    print(f"\n✅ 综合报告已保存: {report_path}")

    # 创建总结
    complete_report = {
        'sample_id': sample_id,
        'prediction': float(prediction),
        'true_value': float(true_value) if true_value is not None else None,
        'num_atoms': atoms_object.num_atoms,
        'num_edges': g.num_edges(),
        'insights': insights,
        'files_generated': [
            f'{sample_id}_attention.png',
            f'{sample_id}_attention_heads.png',
            f'{sample_id}_atom_importance.png',
            f'{sample_id}_edge_importance.png',
            f'{sample_id}_coordination.png',
            f'{sample_id}_substructures.png',
            f'{sample_id}_analysis_report.txt'
        ]
    }

    print(f"\n{'='*80}")
    print(f"✅ 完整分析完成！")
    print(f"   结果保存在: {save_dir}")
    print(f"   生成了 {len(complete_report['files_generated'])} 个文件")
    print(f"{'='*80}\n")

    return complete_report


def generate_insights(
    prediction, true_value,
    atom_importance, atom_df,
    edge_importance, edge_df,
    coord_analysis,
    substructures,
    attention_weights
):
    """
    从分析结果中生成关键洞察

    Returns:
        insights: 洞察列表
    """
    insights = []

    # 1. 预测准确性
    if true_value is not None:
        error = abs(prediction - true_value)
        rel_error = 100 * error / abs(true_value)

        if rel_error < 5:
            insights.append(f"✅ 预测非常准确（相对误差 {rel_error:.2f}%）")
        elif rel_error < 15:
            insights.append(f"⚠️  预测较准确（相对误差 {rel_error:.2f}%）")
        else:
            insights.append(f"❌ 预测误差较大（相对误差 {rel_error:.2f}%），需要进一步分析")

    # 2. 最重要的原子
    top_atoms = atom_df.head(3)
    elements = ', '.join(top_atoms['Element'].tolist())
    avg_importance = top_atoms['Importance'].mean()
    insights.append(f"🔬 最重要的原子: {elements}（平均重要性 {avg_importance:.3f}）")

    # 3. 最重要的化学键
    top_bonds = edge_df.head(3)
    bond_types = ', '.join(top_bonds['bond_type'].tolist())
    insights.append(f"🔗 最重要的化学键: {bond_types}")

    # 4. 配位环境
    coord_df = pd.DataFrame(coord_analysis)
    avg_coord = coord_df['coordination_number'].mean()
    most_common_coord = coord_df['coordination_number'].mode()[0]
    insights.append(f"🔮 平均配位数: {avg_coord:.2f}，最常见配位数: {most_common_coord}")

    # 5. 重要子结构
    if substructures:
        top_motif = '-'.join(substructures[0]['elements'])
        insights.append(f"🧩 最重要的子结构基序: {top_motif}（重要性 {substructures[0]['total_importance']:.3f}）")

    # 6. 跨模态注意力
    if attention_weights is not None:
        if 'graph_to_text' in attention_weights and attention_weights['graph_to_text'] is not None:
            g2t = attention_weights['graph_to_text'].mean().item()
            t2g = attention_weights['text_to_graph'].mean().item()

            if g2t > 0.7:
                insights.append(f"💡 模型强烈依赖文本信息（Graph→Text: {g2t:.3f}）")
            elif g2t < 0.3:
                insights.append(f"💡 模型主要依赖图结构信息（Graph→Text: {g2t:.3f}）")
            else:
                insights.append(f"💡 图和文本信息均衡使用（Graph→Text: {g2t:.3f}, Text→Graph: {t2g:.3f}）")

    # 7. 元素多样性
    element_counts = atom_df['Element'].value_counts()
    if len(element_counts) <= 2:
        insights.append(f"⚗️  简单组成（{len(element_counts)} 种元素）: {', '.join(element_counts.index.tolist())}")
    else:
        insights.append(f"⚗️  复杂组成（{len(element_counts)} 种元素），主要元素: {', '.join(element_counts.head(3).index.tolist())}")

    # 8. 结构复杂度
    edge_per_atom = len(edge_importance) / len(atom_importance)
    if edge_per_atom > 6:
        insights.append(f"🏗️  高连接度结构（每原子 {edge_per_atom:.1f} 条边）")
    elif edge_per_atom < 3:
        insights.append(f"🏗️  低连接度结构（每原子 {edge_per_atom:.1f} 条边）")

    return insights


import pandas as pd


def demo():
    """演示函数"""
    print("\n" + "="*80)
    print("🎯 完整可解释性分析演示")
    print("="*80)
    print("\n这是一个演示脚本。要使用此功能，请：")
    print("\n1. 加载训练好的模型")
    print("2. 准备测试数据")
    print("3. 调用 complete_interpretability_analysis() 函数")
    print("\n示例代码:")
    print("-" * 80)
    print("""
from demo_complete_analysis import complete_interpretability_analysis

# 加载模型
model = ALIGNN(config.model)
model.load_state_dict(checkpoint['model'])
model.eval()

# 获取样本
g, lg, text, label = next(iter(test_loader))
atoms = Atoms.from_dict(sample_data['atoms'])

# 执行完整分析
report = complete_interpretability_analysis(
    model, g, lg, text,
    atoms_object=atoms,
    true_value=label.item(),
    save_dir='./analysis_results',
    sample_id='JVASP-1234'
)

# 查看生成的文件
print(report['files_generated'])
    """)
    print("-" * 80)
    print("\n生成的文件包括:")
    print("  ✅ 跨模态注意力热图")
    print("  ✅ 多头注意力分析")
    print("  ✅ 原子重要性可视化")
    print("  ✅ 边（化学键）重要性可视化")
    print("  ✅ 配位环境分析")
    print("  ✅ 重要子结构识别")
    print("  ✅ 综合分析文本报告")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    demo()
