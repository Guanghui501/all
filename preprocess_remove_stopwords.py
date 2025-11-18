#!/usr/bin/env python
"""
预处理脚本 - 删除描述文本中的停用词

这个脚本会：
1. 读取原始的description.csv文件
2. 加载停用词
3. 删除描述中的停用词
4. 保存处理后的文件

用法:
    python preprocess_remove_stopwords.py \
        --input_file ../dataset/jarvis/formation_energy_peratom/description.csv \
        --output_file ../dataset/jarvis/formation_energy_peratom/description_no_stopwords.csv \
        --stopwords_dir ./stopwords/en/
"""

import os
import csv
import argparse
from tqdm import tqdm


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


def process_csv(input_file, output_file, stopwords):
    """处理CSV文件，删除描述中的停用词

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        stopwords: 停用词集合
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

    for row in tqdm(rows[1:], desc="处理描述"):
        if len(row) > abs(desc_col):
            original_text = row[desc_col]
            filtered_text = remove_stopwords_from_text(original_text, stopwords)

            total_words_before += len(original_text.split())
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


def main():
    parser = argparse.ArgumentParser(
        description='预处理描述文本，删除停用词',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input_file', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出CSV文件路径 (默认: 在输入文件名后添加_no_stopwords)')
    parser.add_argument('--stopwords_dir', type=str, default='./stopwords/en/',
                        help='停用词目录路径')

    args = parser.parse_args()

    # 设置默认输出文件名
    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_no_stopwords{ext}"

    print("="*60)
    print("预处理描述文本 - 删除停用词")
    print("="*60)

    # 加载停用词
    print(f"\n加载停用词: {args.stopwords_dir}")
    stopwords = load_stopwords(args.stopwords_dir)
    print(f"停用词数量: {len(stopwords)}")

    # 处理文件
    print(f"\n处理文件...")
    process_csv(args.input_file, args.output_file, stopwords)

    print("\n" + "="*60)
    print("预处理完成! 使用处理后的文件进行训练:")
    print(f"  cp {args.output_file} {args.input_file}")
    print("或修改训练配置使用新文件")
    print("="*60)


if __name__ == '__main__':
    main()
