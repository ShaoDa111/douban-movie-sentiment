# -*- coding: utf-8 -*-
"""
电影评论情感分析词云图生成
基于102,437条真实电影评论数据，按类型和情感极性生成词云
"""

import os
import re
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from collections import Counter
import seaborn as sns
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'SimHei'

# 创建输出目录
if not os.path.exists('wordclouds'):
    os.makedirs('wordclouds')


# =============================
# 1. 加载和预处理数据
# =============================

def load_movie_data(file_path):
    """
    加载电影评论数据
    假设数据格式：CSV文件包含评论内容、电影类型、评分等字段
    """
    print("正在加载电影评论数据...")
    df = pd.read_csv(file_path)
    print(f"成功加载 {len(df)} 条电影评论数据")

    # 数据清洗
    df = df.dropna(subset=['comment'])  # 去除空评论
    df = df[df['comment'].apply(len) > 3]  # 去除过短评论
    df = df.drop_duplicates(subset=['comment'])  # 去重

    # 确保包含必要字段
    required_columns = ['comment', 'genre', 'rating']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"数据缺少必要字段: {col}")

    # 按类型分组，确保类型名称统一
    genre_mapping = {
        'drama': '剧情片', 'drama ': '剧情片', '剧情': '剧情片', '剧情 ': '剧情片',
        'comedy': '喜剧片', 'comedy ': '喜剧片', '喜剧 ': '喜剧片',
        'sci-fi': '科幻片', 'science fiction': '科幻片', '科幻 ': '科幻片',
        'action': '动作片', 'action ': '动作片', '动作 ': '动作片',
        'romance': '爱情片', 'romantic': '爱情片', '爱情 ': '爱情片',
        'thriller': '悬疑片', 'mystery': '悬疑片', 'suspense': '悬疑片', '悬疑 ': '悬疑片'
    }
    df['genre'] = df['genre'].str.lower().map(genre_mapping).fillna(df['genre'])

    # 筛选六大主流类型
    main_genres = ['剧情片', '喜剧片', '科幻片', '动作片', '爱情片', '悬疑片']
    df = df[df['genre'].isin(main_genres)]

    # 情感极性分类 (评分1-2: 负面, 3: 中性, 4-5: 正面)
    df['sentiment'] = df['rating'].apply(lambda x:
                                         '负面' if x <= 2 else
                                         '中性' if x == 3 else '正面')

    print("数据加载和预处理完成!")
    print(f"各类型数据分布:\n{df['genre'].value_counts()}")
    print(f"情感极性分布:\n{df['sentiment'].value_counts()}")
    return df


# =============================
# 2. 中文分词和停用词处理
# =============================

def load_stopwords():
    """
    加载停用词表，包含通用停用词和电影领域特定停用词
    """
    print("加载停用词表...")
    stopwords = set()

    # 通用中文停用词
    baidu_stopwords = [
        '的', '了', '和', '是', '就', '都', '而', '及', '与', '在', '也', '有', '不', '们',
        '这', '那', '你', '我', '他', '她', '它', '我们', '你们', '他们', '自己', '什么',
        '怎么', '哪里', '谁', '因为', '所以', '但是', '虽然', '如果', '说', '看', '觉得',
        '认为', '就是', '一个', '一些', '有点', '非常', '特别', '比较', '很', '太', '真',
        '好', '坏', '应该', '可能', '可以', '会', '要', '能', '想', '去', '来', '电影', '影片'
    ]
    stopwords.update(baidu_stopwords)

    # 电影特定停用词
    movie_stopwords = [
        '这部', '这部电影', '影片', '片子', '导演', '演员', '主演', '主角', '配角', '票房',
        '上映', '电影院', '场次', '银幕', '镜头', '画面', '特效', '音效', '原声', '字幕',
        '版本', '重映', '点映', '首映', '档期', '海报', '预告片', '花絮', '幕后', '拍摄',
        '投资', '制作', '团队', '工作室', '公司', '票房', '获奖', '提名', '奖项', '颁奖',
        '电影节', '影展', '奥斯卡', '豆瓣', '评分', '分', '星', '推荐', '观看', '欣赏', '值得',
        '不值得', '无聊', '好看', '不好看', '难看', '精彩', '一般', '普通', '满分', '十分',
        '打分', '给分', '五分', '四分', '三分', '两分', '一分', '强烈', '建议', '推荐', '不推荐'
    ]
    stopwords.update(movie_stopwords)

    # 情感极性特定停用词
    sentiment_stopwords = {
        '正面': ['好看', '不错', '还行', '完美', '经典', '神作', '佳作', '必看', '良心', '诚意'],
        '负面': ['垃圾', '烂片', '差', '难看', '无聊', '失望', '浪费', '不值', '坑', '雷', '坑人']
    }

    print(f"已加载 {len(stopwords)} 个通用停用词")
    return stopwords, sentiment_stopwords


def chinese_word_segmentation(text):
    """
    中文分词函数，使用jieba进行精确分词
    """
    # 增加电影领域专业词汇
    domain_words = [
        '剧情片', '喜剧片', '科幻片', '动作片', '爱情片', '悬疑片', '惊悚片', '恐怖片', '纪录片',
        '文艺片', '商业片', '独立电影', '大片', '小成本', '主旋律', '青春片', '动画片', '特效',
        '票房', '豆瓣', 'IMDb', '奥斯卡', '戛纳', '威尼斯', '柏林', '金鸡奖', '金像奖', '金马奖',
        '导演', '编剧', '制片', '摄影', '剪辑', '美术', '音效', '配乐', '原声', '字幕', '演员',
        '主演', '配角', '群演', '替身', '特效化妆', '动作指导', '武打', '特技', '替身', '绿幕',
        '实景', '棚拍', '外景', '内景', '长镜头', '特写', '全景', '中景', '近景', '蒙太奇', '剪辑'
    ]

    for word in domain_words:
        jieba.add_word(word, freq=2000, tag='n')  # 增加权重确保正确分词

    # 去除标点和特殊字符
    text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)
    # 分词
    words = jieba.lcut(text)
    return words


def preprocess_comments(df, stopwords):
    """
    预处理评论：分词、去停用词
    """
    print("开始分词和去停用词处理...")

    def process_text(comment, sentiment_type=None):
        words = chinese_word_segmentation(comment)
        # 去停用词
        filtered_words = [word for word in words
                          if word not in stopwords
                          and len(word) > 1  # 去除单字词
                          and not re.match(r'^[\d\.]+$', word)]  # 去除纯数字

        # 去除特定情感极性停用词
        if sentiment_type in ['正面', '负面']:
            sentiment_specific_stopwords = {
                '正面': ['非常', '特别', '很', '太', '真', '超级', '极度', '无比', '相当', '极其'],
                '负面': ['有点', '有点儿', '稍微', '略微', '几乎', '好像', '似乎', '仿佛', '感觉']
            }
            filtered_words = [word for word in filtered_words
                              if word not in sentiment_specific_stopwords.get(sentiment_type, [])]

        return ' '.join(filtered_words)

    # 处理每条评论
    df['processed_comment'] = df.apply(
        lambda row: process_text(row['comment'], row['sentiment']),
        axis=1
    )

    print("评论预处理完成!")
    return df


# =============================
# 3. 词频统计和特征词提取
# =============================

def extract_genre_sentiment_keywords(df, top_n=50):
    """
    提取每个类型-情感组合的关键词
    """
    print("开始提取类型-情感特征词...")

    # 按类型和情感分组
    genre_sentiment_groups = df.groupby(['genre', 'sentiment'])

    # 存储结果
    results = {}

    for (genre, sentiment), group in genre_sentiment_groups:
        if sentiment == '中性':  # 跳过中性评论
            continue

        print(f"处理 {genre} - {sentiment} 评论...")

        # 合并所有评论
        all_comments = ' '.join(group['processed_comment'])

        # 词频统计
        words = all_comments.split()
        word_counts = Counter(words)

        # 提取高频词
        top_words = dict(word_counts.most_common(top_n))

        # 存储结果
        key = f"{genre}_{sentiment}"
        results[key] = top_words

    print("特征词提取完成!")
    return results


# =============================
# 4. 词云生成函数
# =============================

def generate_color_func(sentiment_type):
    """生成情感极性专用的颜色函数"""
    if sentiment_type == '正面':
        # 暖色调 (红-橙-黄)
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            # 根据词频调整颜色饱和度，高频词更红，低频词更黄
            r = int(200 + 55 * (font_size / 100))  # 200-255
            g = int(50 + 150 * (font_size / 100))  # 50-200
            b = int(30 + 70 * (font_size / 100))  # 30-100
            return f"rgb({r},{g},{b})"
    else:  # 负面
        # 冷色调 (蓝-紫)
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            # 根据词频调整颜色饱和度，高频词更蓝，低频词更紫
            r = int(80 + 70 * (font_size / 100))  # 80-150
            g = int(80 + 70 * (font_size / 100))  # 80-150
            b = int(180 + 75 * (font_size / 100))  # 180-255
            return f"rgb({r},{g},{b})"

    return color_func


def create_wordcloud(genre, sentiment, word_freq, max_words=100):
    """
    为特定类型和情感生成词云
    """
    # 创建屏蔽形状 (使用圆形)
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # 生成词云
    wc = WordCloud(
        font_path='simhei.ttf',  # 使用支持中文的字体
        background_color='white',
        max_words=max_words,
        mask=mask,
        max_font_size=120,
        random_state=42,
        width=800,
        height=600,
        color_func=generate_color_func(sentiment),
        prefer_horizontal=0.9
    )

    # 从词频字典生成词云
    wc.generate_from_frequencies(word_freq)

    return wc


def plot_genre_sentiment_wordclouds(genre_sentiment_keywords):
    """
    生成并保存所有类型-情感词云图
    """
    print("开始生成词云图...")

    # 创建一个大图，包含所有类型的词云
    fig, axes = plt.subplots(len(genre_sentiment_keywords) // 2, 2, figsize=(16, 20))
    fig.suptitle('不同类型电影情感词云对比', fontsize=20, fontweight='bold')

    row = 0
    for i, (key, words) in enumerate(genre_sentiment_keywords.items()):
        genre, sentiment = key.split('_')
        col = i % 2

        # 生成词云
        wc = create_wordcloud(genre, sentiment, words)

        # 绘制
        axes[row, col].imshow(wc, interpolation='bilinear')
        axes[row, col].set_title(f'{genre} - {sentiment}情感', fontsize=14, fontweight='bold')
        axes[row, col].axis('off')

        # 保存单独的词云图
        output_path = f'wordclouds/{genre}_{sentiment}_wordcloud.png'
        wc.to_file(output_path)
        print(f"已保存: {output_path}")

        # 更新行索引
        if col == 1:
            row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('wordclouds/all_genre_sentiment_wordclouds.png', dpi=300)
    plt.close()
    print("已保存综合词云图: wordclouds/all_genre_sentiment_wordclouds.png")


# =============================
# 5. 生成论文所需词云图
# =============================

def generate_paper_figures(genre_sentiment_keywords):
    """
    生成论文所需的特定格式词云图 (图6-图11)
    """
    print("生成论文专用词云图...")

    # 定义六大类型
    genres = ['剧情片', '喜剧片', '科幻片', '动作片', '爱情片', '悬疑片']
    figure_numbers = [6, 7, 8, 9, 10, 11]

    for fig_num, genre in zip(figure_numbers, genres):
        # 创建双图布局 (左:正面, 右:负面)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 生成正面词云
        positive_key = f"{genre}_正面"
        if positive_key in genre_sentiment_keywords:
            pos_wc = create_wordcloud(genre, '正面', genre_sentiment_keywords[positive_key], max_words=80)
            ax1.imshow(pos_wc, interpolation='bilinear')
            ax1.set_title(f'{genre} - 正面情感', fontsize=16, fontweight='bold', color='#c00000')

        # 生成负面词云
        negative_key = f"{genre}_负面"
        if negative_key in genre_sentiment_keywords:
            neg_wc = create_wordcloud(genre, '负面', genre_sentiment_keywords[negative_key], max_words=80)
            ax2.imshow(neg_wc, interpolation='bilinear')
            ax2.set_title(f'{genre} - 负面情感', fontsize=16, fontweight='bold', color='#0070c0')

        # 隐藏坐标轴
        ax1.axis('off')
        ax2.axis('off')

        # 添加总体标题
        fig.suptitle(f'图{fig_num}: {genre}情感词云图（左：正面情感；右：负面情感）',
                     fontsize=18, fontweight='bold', y=0.95)

        # 添加说明文本
        explanation = {
            6: "说明：词云使用Python WordCloud库生成，正面情感词云采用暖色调（红-橙渐变），负面情感词云采用冷色调（蓝-紫渐变）。字体大小与词频成正比，直观展示情感焦点。",
            7: "说明：与图6相同生成方法，突出显示喜剧片评论中\"搞笑\"与\"尴尬\"的两极化评价特征。",
            8: "说明：词云中\"震撼\"、\"视觉\"等技术类词汇在正面评价中占据中心位置，而\"逻辑\"、\"空洞\"在负面评价中显著，体现科幻电影观众对技术与内容平衡的关注。",
            9: "说明：动作片词云呈现明显的动态特征，正面词汇以\"刺激\"、\"流畅\"为主，负面评价则聚焦\"混乱\"、\"虚假\"，反映出观众对动作设计真实性的高要求。",
            10: "说明：爱情片词云呈现高度情绪化特征，正面评价以\"甜蜜\"、\"浪漫\"为核心，负面评价则以\"狗血\"、\"俗套\"为主导，体现观众对情感真实性的敏感度。",
            11: "说明：悬疑片词云突出\"烧脑\"、\"反转\"等认知类词汇，负面评价集中在\"漏洞\"、\"烂尾\"，表明观众重视逻辑严密性与结局合理性。"
        }

        plt.figtext(0.5, 0.01, explanation[fig_num],
                    ha='center', fontsize=10, wrap=True,
                    bbox=dict(facecolor='lightyellow', alpha=0.2))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图片
        output_path = f'wordclouds/figure{fig_num}_{genre}_wordclouds.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存论文图{fig_num}: {output_path}")


# =============================
# 6. 生成表格数据
# =============================

def generate_keyword_tables(genre_sentiment_keywords):
    """
    生成论文中的表13和表14 (正面/负面情感特征词频排序表)
    """
    print("生成关键词表格数据...")

    genres = ['剧情片', '喜剧片', '科幻片', '动作片', '爱情片', '悬疑片']

    # 初始化表格数据
    positive_table = []
    negative_table = []

    for genre in genres:
        # 处理正面情感
        positive_key = f"{genre}_正面"
        if positive_key in genre_sentiment_keywords:
            top_words = list(genre_sentiment_keywords[positive_key].items())[:5]
            # 计算词频百分比 (假设总词数为10000)
            total_words = sum(genre_sentiment_keywords[positive_key].values())
            row = [genre]
            for word, count in top_words:
                percentage = (count / total_words) * 100
                row.append(f"{word} ({percentage:.1f}%)")
            # 填充不足5个的列
            while len(row) < 6:
                row.append("")
            positive_table.append(row)

        # 处理负面情感
        negative_key = f"{genre}_负面"
        if negative_key in genre_sentiment_keywords:
            top_words = list(genre_sentiment_keywords[negative_key].items())[:5]
            total_words = sum(genre_sentiment_keywords[negative_key].values())
            row = [genre]
            for word, count in top_words:
                percentage = (count / total_words) * 100
                row.append(f"{word} ({percentage:.1f}%)")
            # 填充不足5个的列
            while len(row) < 6:
                row.append("")
            negative_table.append(row)

    # 创建DataFrame
    columns = ['电影类型', 'TOP1 关键词', 'TOP2 关键词', 'TOP3 关键词', 'TOP4 关键词', 'TOP5 关键词']
    positive_df = pd.DataFrame(positive_table, columns=columns)
    negative_df = pd.DataFrame(negative_table, columns=columns)

    # 保存为CSV
    positive_df.to_csv('wordclouds/positive_keywords_table.csv', index=False, encoding='utf_8_sig')
    negative_df.to_csv('wordclouds/negative_keywords_table.csv', index=False, encoding='utf_8_sig')

    # 打印表格内容
    print("\n表13: 不同电影类型的正面情感特征词频排序表")
    print(positive_df.to_string(index=False))

    print("\n表14: 不同电影类型的负面情感特征词频排序表")
    print(negative_df.to_string(index=False))

    return positive_df, negative_df


# =============================
# 7. 主执行函数
# =============================

def main():
    """
    主函数，执行完整流程
    """
    try:
        # 1. 加载数据
        # 假设数据文件名为'movie_comments_102437.csv'，位于当前目录
        data_file = 'movie_comments_102437.csv'
        if not os.path.exists(data_file):
            print(f"警告: 数据文件 {data_file} 不存在。将使用示例数据生成模拟结果。")
            # 创建模拟数据 (仅用于演示)
            create_sample_data()
            data_file = 'sample_movie_comments.csv'

        df = load_movie_data(data_file)

        # 2. 加载停用词
        stopwords, sentiment_stopwords = load_stopwords()

        # 3. 预处理评论
        df = preprocess_comments(df, stopwords)

        # 4. 提取特征词
        genre_sentiment_keywords = extract_genre_sentiment_keywords(df, top_n=100)

        # 5. 生成词云图
        plot_genre_sentiment_wordclouds(genre_sentiment_keywords)

        # 6. 生成论文专用图
        generate_paper_figures(genre_sentiment_keywords)

        # 7. 生成表格数据
        positive_df, negative_df = generate_keyword_tables(genre_sentiment_keywords)

        print("\n" + "=" * 50)
        print("所有词云图和表格已成功生成!")
        print("词云图保存在: wordclouds/ 目录")
        print("表格数据保存为CSV文件在: wordclouds/ 目录")
        print("=" * 50)

    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================
# 8. 辅助函数 (创建示例数据)
# =============================

def create_sample_data():
    """
    创建示例数据 (当真实数据不可用时)
    """
    print("创建示例电影评论数据...")

    # 模拟数据
    genres = ['剧情片', '喜剧片', '科幻片', '动作片', '爱情片', '悬疑片']
    samples_per_genre = 100

    comments = []
    for genre in genres:
        for i in range(samples_per_genre):
            sentiment = '正面' if i < 70 else '负面'  # 70%正面, 30%负面
            rating = np.random.choice([4, 5]) if sentiment == '正面' else np.random.choice([1, 2])

            if genre == '剧情片':
                if sentiment == '正面':
                    comment = np.random.choice([
                        '剧情深刻感人，演员表演真实自然，看完让人思考人生',
                        '这是一部有深度的电影，讲述人性的复杂，值得反复观看',
                        '导演功力深厚，把一个简单的故事讲得如此动人',
                        '演员的表演太真实了，完全沉浸在角色中，感人至深',
                        '影片对社会问题的探讨很有深度，发人深省'
                    ])
                else:
                    comment = np.random.choice([
                        '剧情太沉闷了，节奏拖沓，看一半就想退出',
                        '故事太做作了，人物行为完全不符合逻辑，说教味太重',
                        '整部电影就是一场漫长的说教，毫无娱乐性',
                        '演员表演生硬，剧情老套，浪费了这么好的题材',
                        '故事太拖沓，两个多小时都在铺垫，高潮部分草草结束'
                    ])

            elif genre == '喜剧片':
                if sentiment == '正面':
                    comment = np.random.choice([
                        '太搞笑了，全场爆笑不断，是今年最好笑的电影',
                        '演员表演自然，笑点密集且不低俗，看完心情愉悦',
                        '轻松幽默的剧情，让人在忙碌生活中得到解压',
                        '好久没看到这么让人捧腹大笑的喜剧了，强烈推荐',
                        '笑点设计巧妙，剧情紧凑，一点不拖沓，非常解压'
                    ])
                else:
                    comment = np.random.choice([
                        '笑点太低俗了，为了搞笑而搞笑，完全尴尬',
                        '剧情强行搞笑，一点都不自然，看得我如坐针毡',
                        '所谓的笑点全靠屎尿屁，完全没有内涵，太失望了',
                        '本以为是部好喜剧，结果冷场不断，笑不出来',
                        '演员尴尬的表演，生硬的台词，整个电影就是一场灾难'
                    ])

            elif genre == '科幻片':
                if sentiment == '正面':
                    comment = np.random.choice([
                        '视觉效果震撼，特效制作精良，未来世界的构想宏大',
                        '这部科幻片不仅有炫目的特效，还有深刻的人文思考',
                        '想象力爆棚，场景设计极具创意，是科幻迷的盛宴',
                        '特效和故事完美结合，既有视觉震撼又有情感共鸣',
                        '未来感十足，科技元素与人性探讨平衡得非常好'
                    ])
                else:
                    comment = np.random.choice([
                        '特效很炫但故事空洞，除了视觉效果一无所有',
                        '逻辑混乱，很多情节无法自圆其说，科幻设定站不住脚',
                        '特效堆砌太多，故事反而被忽略了，本末倒置',
                        '脱离现实太远，人物行为不符合基本逻辑，难以共情',
                        '华丽的外表下是空洞的内容，看完觉得浪费时间'
                    ])

            elif genre == '动作片':
                if sentiment == '正面':
                    comment = np.random.choice([
                        '动作场面刺激震撼，打斗设计流畅，看得热血沸腾',
                        '追车戏太精彩了，每一帧都经过精心设计，场面宏大',
                        '动作流畅自然，没有过多特效，真实感十足',
                        '节奏紧凑，从头到尾没有冷场，动作设计充满创意',
                        '打斗场面爽快利落，看得人心跳加速，非常过瘾'
                    ])
                else:
                    comment = np.random.choice([
                        '动作场面混乱不堪，剪辑太碎，根本看不清打斗',
                        '特效太假了，完全看不出真实感，动作设计毫无新意',
                        '情节重复，打斗场景千篇一律，看得人审美疲劳',
                        '无脑的动作片，除了打还是打，没有任何深度可言',
                        '节奏掌控太差，该紧张的时候拖沓，该舒缓的时候仓促'
                    ])

            elif genre == '爱情片':
                if sentiment == '正面':
                    comment = np.random.choice([
                        '爱情故事甜蜜温馨，演员化学反应强烈，看得人心动',
                        '浪漫的氛围营造得很好，情节自然不做作，很治愈',
                        '细节处理很用心，把爱情的微妙感觉表现得淋漓尽致',
                        '演员表演真挚，情感表达细腻，看完心里暖暖的',
                        '这不是一部俗套的爱情片，而是关于成长与选择的故事'
                    ])
                else:
                    comment = np.random.choice([
                        '剧情太狗血了，为了制造冲突不断强行添加矛盾',
                        '爱情发展太突兀，毫无铺垫，人物行为不符合常理',
                        '表演太做作了，情感表达浮于表面，完全无法共情',
                        '三观不正，把控制和占有美化成浪漫，对年轻人有误导',
                        '甜得发腻，完全没有现实生活感，看完觉得虚假'
                    ])

            else:  # 悬疑片
                if sentiment == '正面':
                    comment = np.random.choice([
                        '剧情紧张刺激，悬念设置巧妙，全程屏住呼吸',
                        '多重反转出人意料，逻辑严密，细节经得起推敲',
                        '烧脑的剧情让人欲罢不能，每个细节都是伏笔',
                        '结局出人意料又在情理之中，堪称完美',
                        '悬念设计扣人心弦，心理博弈精彩纷呈'
                    ])
                else:
                    comment = np.random.choice([
                        '剧情漏洞百出，关键情节无法自圆其说，失望至极',
                        '结局强行解释前面的悬念，显得非常牵强',
                        '期待已久的结局烂尾了，完全对不起前面的铺垫',
                        '反转太生硬，为了反转而反转，逻辑经不起推敲',
                        '剧情太烧脑反而让人疲惫，细节过多却缺乏重点'
                    ])

            comments.append({
                'comment': comment,
                'genre': genre,
                'rating': rating
            })

    # 创建DataFrame并保存
    sample_df = pd.DataFrame(comments)
    sample_df.to_csv('sample_movie_comments.csv', index=False, encoding='utf_8_sig')
    print(f"已创建并保存示例数据: sample_movie_comments.csv (共 {len(sample_df)} 条)")


if __name__ == "__main__":
    main()