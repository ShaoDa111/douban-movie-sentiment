import pandas as pd
import re
import jieba
import numpy as np
import os
import time

# 1. 读取数据（已验证GBK编码有效）
file_path = r"C:/Users/HP/Desktop/douban_comments_cleaned.csv"
comments = pd.read_csv(file_path, encoding='gbk', low_memory=False)
print(f"✅ 成功读取文件，总行数: {len(comments)}")
print(f"可用列名: {', '.join(comments.columns.tolist())}")


# 2. 智能识别真正的评论内容列（关键修复）
def find_real_comment_column(df):
    """精确识别真正的评论内容列（排除ID、评分等非文本列）"""
    # 定义候选列名（优先级从高到低）
    priority_columns = [
        # 英文列名
        'comment_text', 'comment_content', 'content', 'text', 'review', 'short_comment',
        # 中文列名
        '评论内容', '评论文本', '短评', '影评', '内容', '评论',
        # 其他可能
        'comment', 'comment_body'
    ]

    # 第1步：检查优先级列
    for col in priority_columns:
        if col in df.columns:
            # 验证是否为真正的文本内容
            sample_texts = df[col].head(10).dropna().astype(str).tolist()
            if any(len(text.strip()) > 10 and not re.match(r'^\d+$', text.strip()) for text in sample_texts):
                print(f"🔍 通过优先级匹配找到列: '{col}'")
                return col

    # 第2步：分析所有文本列
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # 采样10个非空值
            samples = df[col].dropna().head(10).astype(str).tolist()
            if not samples:
                continue

            # 计算文本特征
            avg_len = np.mean([len(s) for s in samples])
            has_chinese = any(re.search(r'[\u4e00-\u9fa5]', s) for s in samples)
            is_numeric_id = all(re.match(r'^\d+(_\d+)*$', s.strip()) for s in samples if s.strip())

            # 判断是否可能是评论内容
            if avg_len > 15 and has_chinese and not is_numeric_id:
                text_columns.append((col, avg_len))

    # 选择最长的文本列
    if text_columns:
        best_col = max(text_columns, key=lambda x: x[1])[0]
        print(f"🔍 通过内容分析找到列: '{best_col}' (平均长度: {max(text_columns, key=lambda x: x[1])[1]:.1f})")
        return best_col

    # 第3步：交互式选择（备用）
    print("\n" + "=" * 50)
    print("⚠️ 无法自动识别评论列，请从以下列中选择:")
    for i, col in enumerate(df.columns):
        # 显示列的样本数据
        sample = str(df[col].iloc[0])[:20] if not df[col].empty else "空"
        print(f"  {i + 1}. {col}: {sample}{'...' if len(sample) > 20 else ''}")

    while True:
        try:
            choice = int(input("\n请输入评论列的编号: ")) - 1
            if 0 <= choice < len(df.columns):
                selected_col = df.columns[choice]
                confirm = input(f"确认选择 '{selected_col}' 作为评论列? (y/n): ").lower()
                if confirm == 'y':
                    return selected_col
            print("无效选择，请重试")
        except (ValueError, IndexError):
            print("请输入有效数字")


# 3. 找到真正的评论列
comment_col = find_real_comment_column(comments)
print(f"\n✅ 确定评论列为: '{comment_col}'")

# 4. 验证列内容（显示真实示例）
print("\n" + "=" * 50)
print("评论列内容验证")
print("=" * 50)
valid_samples = comments[comment_col].dropna().head(5).tolist()
for i, sample in enumerate(valid_samples):
    print(f"样本 {i + 1}: {str(sample)[:100]}{'...' if len(str(sample)) > 100 else ''}")


# 5. 应用您的清洗函数（严格按要求）
def clean_text(text):
    """严格按要求的清洗函数：只保留中文，去HTML"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    try:
        # 1. 去HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 2. 只保留中文字符
        text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        # 3. 合并多个空格
        return ' '.join(text.split())
    except Exception as e:
        print(f"清洗错误: {e}")
        return ""


# 6. 应用清洗（带进度反馈）
print("\n🔄 正在清洗评论文本...")
start_time = time.time()

# 添加进度显示
total = len(comments)
for i in range(0, total, 10000):
    comments.loc[i:min(i + 10000, total), 'cleaned_comment'] = comments[comment_col].iloc[
        i:min(i + 10000, total)].apply(clean_text)
    print(f"  已处理 {min(i + 10000, total)}/{total} 条...", end='\r')

print(f"\n✅ 清洗完成！耗时: {time.time() - start_time:.1f}秒")


# 7. 添加分词功能
def chinese_tokenize(text):
    """中文分词（保留您要求的格式）"""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    # 使用jieba精确分词
    words = jieba.lcut(text)

    # 严格按您的要求：只保留中文词语（已由清洗步骤保证）
    return ' '.join(words)


# 8. 应用分词（带进度反馈）
print("\n🔄 正在进行中文分词...")
start_time = time.time()

for i in range(0, total, 10000):
    comments.loc[i:min(i + 10000, total), 'tokenized_comment'] = comments['cleaned_comment'].iloc[
        i:min(i + 10000, total)].apply(chinese_tokenize)
    print(f"  已处理 {min(i + 10000, total)}/{total} 条...", end='\r')

print(f"\n✅ 分词完成！耗时: {time.time() - start_time:.1f}秒")

# 9. 生成真实示例
print("\n" + "=" * 60)
print("清洗与分词结果示例")
print("=" * 60)

# 找一个有内容的评论
sample_idx = comments[comments['cleaned_comment'].str.len() > 5].index[0]

original = str(comments[comment_col].loc[sample_idx])
cleaned = comments['cleaned_comment'].loc[sample_idx]
tokenized = comments['tokenized_comment'].loc[sample_idx]

print(f"原始评论 (行 {sample_idx}):\n{original[:150]}{'...' if len(original) > 150 else ''}")
print(f"\n清洗后:\n{cleaned}")
print(f"\n分词后:\n{tokenized}")

# 10. 保存结果
output_path = r"C:/Users/HP/Desktop/douban_comments_processed.csv"
comments.to_csv(output_path, index=False, encoding='utf_8_sig')
print(f"\n✅ 处理完成！结果已保存至: {output_path}")

# 11. 详细统计
valid_comments = comments[comments['cleaned_comment'].str.len() > 0]
print(f"\n📊 最终数据统计:")
print(f"  总评论数: {len(comments):,}")
print(f"  有效评论数 (非空): {len(valid_comments):,} ({len(valid_comments) / len(comments) * 100:.1f}%)")
print(f"  平均清洗后长度: {valid_comments['cleaned_comment'].apply(len).mean():.1f} 字符")
print(
    f"  平均分词数量: {valid_comments['tokenized_comment'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0).mean():.1f} 词")

# 12. 验证电影ID存在
if 'movie_id' in comments.columns:
    print(f"  唯一电影ID数: {comments['movie_id'].nunique():,}")
else:
    print("⚠️ 未找到 'movie_id' 列，但数据处理已完成")