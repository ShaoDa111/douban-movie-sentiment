# 豆瓣影评中文情感分析（NLP 全链路）

> 对大规模中文影评做分词、情感标注与可视化，提炼不同电影类型观众的情感关注点。完整覆盖「数据清洗 → jieba 分词 → 情感词典打分 → 特征工程 → 机器学习分类 → 词云可视化」的端到端流程。

**数据规模**：原始约 12.4 万条豆瓣影评 → 清洗后约 10.6 万条，覆盖约 1068 部电影。
**适用场景**：中文 NLP 教学 / 论文实证 / 影评舆情分析。

---

## 项目结构

```
douban-movie-sentiment/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── 获取情感词.py     # 下载 BosonNLP 情感词典 + 构建分层情感词库
│   ├── 分词代码.py       # 自动识别评论列、jieba 分词、清洗
│   └── 对比词云图.py     # 按电影类型生成正/负词云与情感分布图
└── dictionary/
    ├── positive_words.txt # 正面情感词（词语\t权重）
    ├── negative_words.txt # 负面情感词（词语\t权重）
    └── README.txt        # 词库使用说明
```

---

## 运行步骤

```bash
pip install -r requirements.txt

# 1) 准备数据：将清洗后的影评 CSV 放到本地（见下方「数据集」）
# 2) 获取/构建情感词典（BosonNLP 会自动从 GitHub 下载）
python code/获取情感词.py

# 3) 分词与清洗（脚本会自动识别评论文本列、处理 GBK 编码与噪声）
python code/分词代码.py

# 4) 生成分类型正/负词云与情感分布图
python code/对比词云图.py
```

---

## 方法说明

1. **数据读取与清洗**：读取影评 CSV（GBK 编码），自动识别真正的评论内容列（排除 ID / 评分等），去重、去噪。
2. **分词**：`jieba` 中文分词。
3. **情感标注**：下载 **BosonNLP** 情感词典打分，结合自构建的正/负情感词库（`dictionary/`），并用 `cohen_kappa` 做一致性校验。
4. **特征工程**：构建 **TF-IDF** 特征矩阵与 **Word2Vec** 词向量。
5. **建模**：训练机器学习分类器对影评情感极性分类。
6. **可视化**：按爱情 / 悬疑 / 剧情 / 喜剧 / 科幻 / 动作等类型，分别生成正面 / 负面词云与情感分布图，对比各类型观众的情感关注点差异。

---

## 数据集（重要）

原始影评与中间产物（TF-IDF 特征约 2GB、Word2Vec 模型、训练特征矩阵约 3.2GB 等）**体积过大、未随仓库分发**，请自行准备并放在本地：

- `douban_comments_cleaned.csv`（清洗后影评，GBK 编码，含评论文本列）
- `douban_movies.csv`（电影元数据，用于按类型分组）

`dictionary/` 下的情感词库已随仓库提供，可直接用于特征工程或后处理校准。

---

## 依赖

`jieba` / `pandas` / `numpy` / `scikit-learn` / `gensim`(Word2Vec) / `matplotlib` / `seaborn` / `requests` / `tqdm`
