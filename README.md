# Online Shoppers Intention Analysis

基于 [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) 的数据分析与建模项目。

## 📐 项目概览

- Project : Online Shoppers Intention Analysis
- Type    : 数据分析 + 机器学习建模（Jupyter Notebook）
- Language: Python 3.10
- Dataset : UCI Online Shoppers Purchasing Intention Dataset（12330 条电商用户浏览记录）
- Goal    : 预测用户是否会产生购买行为（Revenue = True/False）
- 核心依赖：pandas、numpy、scikit-learn（分类/聚类/评估）、seaborn / matplotlib（可视化）、imbalanced-learn



## 📁 文件结构
```txt
Online-Shoppers-Intention-Analysis/
├── data/
│   └── online_shoppers_intention.csv   — UCI 数据集（12330×18）
├── Online Shoppers Intention Analysis.ipynb  — 唯一的代码文件（核心）
├── requirements.txt   — conda 导出的完整环境依赖
├── environment.yml    — conda 环境配置
├── README.md          — 项目说明
└── .gitignore
```
## 🔀 整体数据流
<p align="center">
  <img src="image\online_shoppers_ml_pipeline.svg" width="800" alt="整体数据流图">
</p>

## 快速开始

### 1. Python 版本
建议使用 Python 3.10 或 3.11

### 2. 环境配置
```pwsh
# 使用 pip requirements.txt
conda create -n shoppers python=3.10 -y
conda activate shoppers
pip install -r requirements.txt

# or 使用 conda environment.yml
conda env create -f environment.yml
conda activate shoppers
```

### 3. 启动 Notebook
```pwsh
jupyter notebook
```
运行笔记本:  `Online Shoppers Intention Analysis.ipynb`

## 代码结构（Jupyter Notebook，7 个 Section）

**Section 1 — 数据加载与探索性分析（EDA）**
- 加载数据，`.info()`, `.describe()`, 缺失值检查
- 目标变量分布（饼图/柱状图，展示类别不平衡）
- 数值特征的分布直方图、箱线图
- 类别特征的频率统计（Month, VisitorType, Weekend 等）
- 相关性热力图（数值特征之间）

**Section 2 — 数据预处理**
- 类别特征编码：`LabelEncoder` 或 `pd.get_dummies`（Month, VisitorType, Weekend, OperatingSystems, Browser, Region, TrafficType）
- 特征标准化：`StandardScaler`（对 SVM 和 K-Means 尤其重要）
- 处理类别不平衡：建议用 SMOTE 或简单的类权重调整（`class_weight='balanced'`）
- 训练集/测试集划分：80/20，`stratify=y`

**Section 3 — 无监督学习：K-Means 聚类**
- 用肘部法则（Elbow Method）确定最佳 K 值（画 SSE 曲线）
- 用轮廓系数（Silhouette Score）验证 K 值选择
- 对 K=3 或 K=4 进行聚类
- PCA 降维到 2D 做散点图可视化，颜色标记簇
- 分析每个簇的特征均值（群体画像）：哪些是"高意向用户"、"浏览型用户"、"跳出型用户"
- 将聚类标签作为新特征加入分类模型

**Section 4 — 监督学习：四个分类模型**
- 朴素贝叶斯（GaussianNB）
- 决策树（DecisionTreeClassifier）
- 随机森林（RandomForestClassifier）
- 支持向量机（SVC，使用 RBF 核）
- 每个模型用 5 折交叉验证评估
- 随机森林和 SVM 做 GridSearchCV 调参

**Section 5 — 模型评估与对比**
- 每个模型的 Accuracy, Precision, Recall, F1-Score
- 混淆矩阵热力图（4 个模型并排）
- ROC 曲线（4 条线在同一张图上）
- 汇总表格对比所有模型

**Section 6 — 特征重要性分析**
- 随机森林的 `feature_importances_` 柱状图
- 决策树可视化（前 3-4 层，用 `plot_tree`）

**Section 7 — 结论与总结**
- 打印最终推荐模型及其测试集表现
