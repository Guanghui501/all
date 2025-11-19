#!/usr/bin/env python
"""
预处理脚本 - 删除描述文本中的停用词和局部信息句子

这个脚本会：
1. 读取原始的description.csv文件
2. 加载停用词
3. 删除描述中的停用词
4. 过滤包含局部信息（键长、角度等）的句子，只保留全局/半全局信息
5. 保存处理后的文件

用法:
    python preprocess_remove_stopwords.py \
        --input_file ../dataset/jarvis/formation_energy_peratom/description.csv \
        --output_file ../dataset/jarvis/formation_energy_peratom/description_no_stopwords.csv \
        --stopwords_dir ./stopwords/en/ \
        --filter_local_info
"""

import os
import re
import csv
import argparse
from tqdm import tqdm

# 局部信息关键词列表 - 包含这些关键词的句子将被过滤掉
LOCAL_INFO_KEYWORDS = [
    # 键长相关
    'bond length', 'bond lengths', 'bond-length', 'bond-lengths',
    'bond distance', 'bond distances',
    'interatomic distance', 'interatomic distances',
    'atomic distance', 'atomic distances',

    # 角度相关
    'bond angle', 'bond angles', 'bond-angle', 'bond-angles',
    'dihedral angle', 'dihedral angles', 'dihedral-angle',
    'torsion angle', 'torsion angles',
    'angle between', 'angles between',

    # 配位相关
    'coordination number', 'coordination numbers',
    'coordination environment', 'coordination environments',
    'nearest neighbor', 'nearest neighbors', 'nearest-neighbor',
    'first neighbor', 'first neighbors',
    'second neighbor', 'second neighbors',

    # 局部几何
    'local geometry', 'local geometries',
    'local structure', 'local structures',
    'local environment', 'local environments',
    'atomic environment', 'atomic environments',

    # 特定原子位置
    'positioned at', 'located at', 'situated at',
    'at position', 'at site',

    # 具体数值描述（通常是局部信息）
    'angstrom', 'angstroms', 'Å',
    'degree', 'degrees', '°',
]


def split_into_sentences(text):
    """将文本分割成句子

    Args:
        text: 原始文本

    Returns:
        list: 句子列表
    """
    # 使用正则表达式按句号、问号、感叹号分割
    # 保留分隔符
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def contains_local_info(sentence, local_keywords):
    """检查句子是否包含局部信息关键词

    Args:
        sentence: 句子文本
        local_keywords: 局部信息关键词列表

    Returns:
        bool: 如果包含局部信息返回True
    """
    sentence_lower = sentence.lower()
    for keyword in local_keywords:
        if keyword.lower() in sentence_lower:
            return True
    return False


def filter_local_info_sentences(text, local_keywords=None):
    """过滤包含局部信息的句子，只保留全局/半全局信息

    Args:
        text: 原始文本
        local_keywords: 局部信息关键词列表（默认使用LOCAL_INFO_KEYWORDS）

    Returns:
        str: 过滤后的文本
        int: 被过滤的句子数量
    """
    if local_keywords is None:
        local_keywords = LOCAL_INFO_KEYWORDS

    sentences = split_into_sentences(text)
    filtered_sentences = []
    removed_count = 0

    for sentence in sentences:
        if not contains_local_info(sentence, local_keywords):
            filtered_sentences.append(sentence)
        else:
            removed_count += 1

    return ' '.join(filtered_sentences), removed_count


def load_stopwords(stopwords_dir):
    """从目录加载停用词

    Args:
        stopwords_dir: 停用词目录路径

    Returns:
        set: 停用词集合
    """
    stopwords = set()

    if not os.path.exists(stopwords_dir):
        print(f"警告: 停用词目录不存在: {stopwords_dir}")
        return stopwords

    # 遍历目录中的所有txt文件
    for filename in os.listdir(stopwords_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(stopwords_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            stopwords.add(word)
            except Exception as e:
                print(f"警告: 读取文件 {filepath} 失败: {e}")

    return stopwords


def remove_stopwords_from_text(text, stopwords):
    """从文本中删除停用词

    Args:
        text: 原始文本
        stopwords: 停用词集合

    Returns:
        str: 删除停用词后的文本
    """
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stopwords]
    return ' '.join(filtered_words)


def process_csv(input_file, output_file, stopwords, filter_local_info=False):
    """处理CSV文件，删除描述中的停用词和/或局部信息句子

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        stopwords: 停用词集合
        filter_local_info: 是否过滤局部信息句子
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("错误: CSV文件为空")
        return

    # 获取表头
    header = rows[0]
    print(f"CSV表头: {header}")

    # 查找描述列（通常是最后一列或名为'description'的列）
    desc_col = -1  # 默认最后一列
    for i, col_name in enumerate(header):
        if 'description' in col_name.lower() or 'text' in col_name.lower():
            desc_col = i
            break

    print(f"描述列索引: {desc_col} (列名: {header[desc_col] if desc_col < len(header) else '最后一列'})")

    # 处理每一行
    processed_rows = [header]

    total_words_before = 0
    total_words_after = 0
    total_sentences_removed = 0

    for row in tqdm(rows[1:], desc="处理描述"):
        if len(row) > abs(desc_col):
            original_text = row[desc_col]
            total_words_before += len(original_text.split())

            # 第一步：过滤局部信息句子（如果启用）
            if filter_local_info:
                filtered_text, sentences_removed = filter_local_info_sentences(original_text)
                total_sentences_removed += sentences_removed
            else:
                filtered_text = original_text

            # 第二步：删除停用词
            if stopwords:
                filtered_text = remove_stopwords_from_text(filtered_text, stopwords)

            total_words_after += len(filtered_text.split())

            # 创建新行
            new_row = row.copy()
            new_row[desc_col] = filtered_text
            processed_rows.append(new_row)
        else:
            processed_rows.append(row)

    # 保存输出文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(processed_rows)

    # 统计信息
    reduction = (1 - total_words_after / total_words_before) * 100 if total_words_before > 0 else 0
    print(f"\n处理完成!")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  处理样本数: {len(rows) - 1}")
    print(f"  总词数 (处理前): {total_words_before}")
    print(f"  总词数 (处理后): {total_words_after}")
    print(f"  词数减少: {reduction:.1f}%")
    if filter_local_info:
        print(f"  过滤的局部信息句子数: {total_sentences_removed}")


def main():
    parser = argparse.ArgumentParser(
        description='预处理描述文本，删除停用词和局部信息句子',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_file', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出CSV文件路径 (默认: 在输入文件名后添加_processed)')
    parser.add_argument('--stopwords_dir', type=str, default='./stopwords/en/',
                        help='停用词目录路径')
    parser.add_argument('--filter_local_info', action='store_true',
                        help='过滤包含局部信息（键长、角度等）的句子')
    parser.add_argument('--no_stopwords', action='store_true',
                        help='不删除停用词（仅过滤局部信息）')
    parser.add_argument('--show_local_keywords', action='store_true',
                        help='显示局部信息关键词列表')

    args = parser.parse_args()

    # 显示局部信息关键词
    if args.show_local_keywords:
        print("局部信息关键词列表:")
        print("-" * 40)
        for keyword in LOCAL_INFO_KEYWORDS:
            print(f"  - {keyword}")
        print("-" * 40)
        print(f"共 {len(LOCAL_INFO_KEYWORDS)} 个关键词")
        return

    # 设置默认输出文件名
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        suffix = "_processed"
        if args.filter_local_info and not args.no_stopwords:
            suffix = "_global_info"
        elif args.filter_local_info:
            suffix = "_no_local"
        elif not args.no_stopwords:
            suffix = "_no_stopwords"
        args.output_file = f"{base}{suffix}{ext}"

    print("="*60)
    print("预处理描述文本")
    print("="*60)

    # 加载停用词
    stopwords = set()
    if not args.no_stopwords:
        print(f"\n加载停用词: {args.stopwords_dir}")
        stopwords = load_stopwords(args.stopwords_dir)
        print(f"停用词数量: {len(stopwords)}")
    else:
        print("\n跳过停用词删除")

    # 显示过滤设置
    if args.filter_local_info:
        print(f"\n启用局部信息过滤 (关键词数: {len(LOCAL_INFO_KEYWORDS)})")
        print("将过滤包含以下类型信息的句子:")
        print("  - 键长、原子间距离")
        print("  - 键角、二面角、扭转角")
        print("  - 配位数、配位环境")
        print("  - 局部几何、局部结构")
        print("  - 具体位置描述")

    # 处理文件
    print(f"\n处理文件...")
    process_csv(args.input_file, args.output_file, stopwords, args.filter_local_info)

    print("\n" + "="*60)
    print("预处理完成! 使用处理后的文件进行训练:")
    print(f"  cp {args.output_file} {args.input_file}")
    print("或修改训练配置使用新文件")
    print("="*60)


if __name__ == '__main__':
    main()
