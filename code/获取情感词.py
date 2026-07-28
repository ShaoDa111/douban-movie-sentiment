# -*- coding: utf-8 -*-
"""
电影评论情感词库构建系统
版本: 1.0
适用场景: 学术研究/论文实证
"""

import os
import re
import json
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.metrics import cohen_kappa_score
import requests
from tqdm import tqdm
import time
from datetime import datetime

# 设置全局参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
sns.set(style="whitegrid", font="SimHei")

# 创建项目目录结构
project_dirs = [
    'data/raw',
    'data/processed',
    'data/dictionary',
    'results/statistics',
    'results/visualization',
    'results/paper_tables'
]

for dir_path in project_dirs:
    os.makedirs(dir_path, exist_ok=True)

print("✅ 项目目录结构创建完成")


def download_boson_nlp_dict(save_path='data/dictionary/boson_sentiment.txt'):
    """下载BosonNLP情感词典 (2023年更新版)"""
    url = "https://raw.githubusercontent.com/bosondata/BosonNLP-Sentiment-Analyzer/master/data/sentiment_score.txt"

    try:
        print("🔄 正在从GitHub下载BosonNLP情感词典 (2023更新版)...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # 检查请求是否成功

        # 保存词典
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"✅ BosonNLP情感词典下载成功，已保存至: {save_path}")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("🔄 尝试使用备用下载方式...")

        # 备用词典内容 (简化版，仅用于演示)
        backup_dict = """非常好 3.5
好 2.8
不错 2.5
一般 0.2
差 -2.1
非常差 -3.8
喜欢 3.2
讨厌 -3.6
精彩 3.4
无聊 -2.9
"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(backup_dict)

        print(f"✅ 已创建备用情感词典，仅包含示例词汇")
        return False


def load_sentiment_dictionary(dict_path='data/dictionary/boson_sentiment.txt'):
    """加载情感词典"""
    sentiment_dict = {}
    word_count = 0

    if not os.path.exists(dict_path):
        print(f"⚠️ 词典文件不存在，尝试下载: {dict_path}")
        download_boson_nlp_dict(dict_path)

    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    try:
                        score = float(parts[1])
                        sentiment_dict[word] = score
                        word_count += 1
                    except ValueError:
                        continue

        print(f"✅ 成功加载情感词典，共 {word_count} 个词汇，路径: {dict_path}")
        return sentiment_dict

    except Exception as e:
        print(f"❌ 加载词典失败: {str(e)}")
        return {}


# 下载并加载BosonNLP词典
boson_dict = load_sentiment_dictionary()
print(f"词典样例: {dict(list(boson_dict.items())[:5])}")