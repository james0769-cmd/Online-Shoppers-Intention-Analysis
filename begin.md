## 一、数据集信息

**Online Shoppers Purchasing Intention Dataset**，来源于 UCI Machine Learning Repository。约 12,330 条用户浏览会话记录，18 个特征（数值型 + 类别型），目标变量 `Revenue`（是否产生购买，True/False）。数据集天然适合做二分类 + 聚类分析，且类别不平衡（约 85% 未购买），正好可以展示处理不平衡数据的能力。

本地数据文件已下载到：`data/online_shoppers_intention.csv`

---

## 二、代码结构（Jupyter Notebook，7 个 Section）

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
- **加分点**：将聚类标签作为新特征加入分类模型

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

---

## 三、分析报告大纲（≥4000 字）

| 章节 | 建议字数 | 核心内容 |
|------|---------|---------|
| 1. 引言 | ~400字 | 研究背景（电商转化率预测的商业价值）、研究目的、课程算法关联 |
| 2. 数据集介绍 | ~400字 | 数据来源、特征描述（18个特征逐一说明）、目标变量、数据规模 |
| 3. 数据预处理 | ~500字 | 缺失值处理、编码方式选择理由、标准化原因、不平衡处理策略 |
| 4. 探索性数据分析 | ~500字 | 配合图表描述数据分布规律、特征相关性发现 |
| 5. 聚类分析 | ~600字 | K值选择过程、聚类结果、群体画像解读、商业含义 |
| 6. 分类模型与实验设计 | ~600字 | 四个算法原理简述、为什么选这些算法、交叉验证设计、超参数搜索 |
| 7. 实验结果与讨论 | ~700字 | 各模型表现对比、哪个最好及为什么、偏差-方差分析、聚类特征是否提升效果 |
| 8. 结论与展望 | ~300字 | 总结发现、局限性、未来改进方向 |
