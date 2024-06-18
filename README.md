# 推荐   E-commerce search
大规模推荐系统
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/325a1634-793c-424d-b36e-cc74b653e7f3)
term mismatch between queries and items

## 数据循环
算法决定展示内容，展示内容影响用户行为，而用户行为反馈又会决定后续算法的学习，形成循环。在这种循环下，训练集和测试集与监督学习独立同分布的假设相去甚远，同时系统层面上缺乏有效探索机制的设计，可能导致模型更聚焦于局部最优。

在用户行为稀疏的场景下，数据循环问题尤其显著。问题的本质：有限的数据无法获得绝对置信的预估，探索和利用（Explore&Exploit）是突破数据循环的关键。


# cold start solutions
新用户冷启动  新上架的物品没有历史行为数据

提供非个性化的推荐(用户冷启动)
1）热门商品

2）人工指定策略

3）提供多样性的选择

利用用户的注册信息(用户冷启动、系统冷启动)

1）基于用户信息，比如年龄，性别，地域、学历、职业等做推荐

2）让用户选择兴趣点，让用户选择自己喜欢的分类标签（避免选项太多，操作复杂）

3）利用社交关系，将好友喜欢的商品推荐给你


## DropoutNet

## Contextual Bandit 
与传统方法的区别：

每个候选商品学习一个独立的模型，避免传统大一统模型的样本分布不平衡问题

传统方法采用贪心策略，尽最大可能利用已学到的知识，易因马太效应陷入信息茧房；Bandit算法有显式的exploration机制，曝光少的物品会获得更高的展现加权

是一种在线学习方案，模型实时更新；相较A/B测试方案，能更快地收敛到最优策略

# user profiling

#  query processing
NLP技术：从基础的分词、NER，到应用上的Query分析、基础相关

tokenization, spelling correction, query expansion and rewriting.
## query rewriting 
transforms the original query to
another similar query that might beer represent the search need 

learning query
rewrites  as an indirect approach to bridge vocabulary gap
between queries and documents. 


#  召回阶段  candidate retrieval
从全量信息集合中触发尽可能多的正确结果 

## 基于关键字的倒排索引
计算关键字在问题集中的 TF-IDF 以及 BM（Best Matching）25 得分，并建立倒排索引表

### 第三方库 Elasticsearch 
elasticsearch是面向文档（Document）存储的，可以是数据库中的一条商品数据，一个订单信息。文档数据会被序列化为json格式后存储在elasticsearch中


### 正向索引
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/320d2b75-e897-4b5c-9983-0436215b886a)
如果是根据id查询，那么直接走索引，查询速度非常快。

但如果是基于title做模糊查询，只能是逐行扫描数据
### 倒排索引 inverted indexes 

**Problem**:cannot retrieve dierent items according to the current
user’s characteristics

hand-crafed

indexing tags for items

 building separate indexes for dierent group of
users. 

## Semantic Retrieval  基于向量的语义召回
基于 BERT 向量的语义检索

对候选问题集合进行向量编码，得到 corpus 向量矩阵

当用户输入 query 时，同样进行编码得到 query 向量表示

然后进行语义检索（矩阵操作，KNN，FAISS）
### 分解user-item矩阵



![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/66a0a534-5d2a-4afd-b10f-dd170e21b91b)

推荐物品主要包含两部分。一个是搜索词，涉及到Query意图、核心词抽取、语义完整性等NLP技术，或者Query相关的信息流消费pv、索引网页的数量、信息流的分类、不同时间窗口的趋势，还有安全相关的质量控制。

另一个是搜索结果页的呈现，包括文章样式，搜索结果基础相关性、语义相关性、权威性、时效性，文章的意图、url点击分布。
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/7a729f34-5e38-4d58-9294-1356cee76924)
## 查询query
分词和查询Query的语义理解方面做到业务可用的效果，至少需要百万级有标注的商品和电商搜索关键词数据做训练
## query自动补全
更加关注topN满足率 


## two tower architecture
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/b125fecd-c93b-46a9-a401-46070f60aad2)

# Rank
历史上粗排模型经历了从最简单的统计反馈模型发展到了特征裁剪下的轻量级LR或FM模型以及当前双塔深度学习模型

双塔结构限定了用户侧与物品侧没法进行特征交叉 
## 粗排：
消费者行为个性化是指把消费者的浏览数据、购买数据使用到搜索排序中，当消费者用搜索时，可以快捷方便的找到这些商品。随后消费者性别模型、消费者购买力模型等数据也会被应用到搜索排序中，使排序多样化，满足不同消费者的不同搜索需求。
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/a137786c-fc1c-45d6-a7aa-0da848c6465e)


### 文本相关性
商品的文本描述信息和搜索关键词的是否相关或匹配。
### 类目
### 商品人气
商品销量

销售额

消费者评论
### 用户搜索反馈
反馈数据包括：某查询词结果中商品的点击量和下单量，消费者通过搜索进入商品单品页的平均时间，商品的搜索点击转化率。
## 精排
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/ff9f90fb-f702-4dcc-813f-2008e52878ca)
### CTR、CVR

将商家排序问题建模为点击率、转化率、客单价等指标的预测问题
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/e48876bd-6d11-416a-911f-e4c50cb625de)

## 
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/dfbcd1c7-5b9f-42dd-a952-a53605b5b00a)


##  离散排序 
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/2066920d-0cb7-443d-a8b3-fff5d3f3b923)

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/51bfa954-90b3-46ac-bc3f-f7e11ffbd74b)

每个特征从以上4个维度中各取一到两个进行组合，再从历史数据中统计该组合特征最终的特征值。

比如，商品（实体）最近1天（时间）的曝光（行为）量（统计指标）、商品所在店铺（实体）最近30天（时间）的销量（行为类型+统计维度）等等。

由以上方法产生的特征数量级，相当于4个维度的笛卡尔积。

## XGBoost
XGBoost 的特征重要性:

	1.	Gain：
	•	这是最常用的一种方法，计算每个特征在所有树节点中的平均增益。增益表示在该特征节点进行分裂所带来的目标函数的提升（通常是信息增益或者 Gini 指数的减少）。
	•	计算过程：对每棵树的每个节点，记录该节点的增益，然后对同一特征的所有增益进行累加，最后取平均值。
 
	2.	Cover：
	•	这个方法衡量的是使用特征的频率以及它所覆盖的样本数。每次使用一个特征进行分裂时，记录被该节点覆盖的样本数，然后对所有使用该特征的节点进行累加。
	•	计算过程：统计每个特征在所有树中的所有节点所覆盖的样本数，然后进行平均。
 
	3.	Frequency：
	•	也称为 “weight”，是特征在所有树中被使用的次数。
	•	计算过程：统计每个特征在所有树中被用作分裂节点的次数，然后进行累加


## FM（Factorization Machines）向量分解机
FM 模型的核心思想是将高维稀疏特征映射到低维的密集向量空间。具体来说，FM 模型通过隐向量来表示特征之间的交互作用。

引入矩阵分解技术来捕捉特征之间的交互作用 通过将所有的ID特征映射到同一个隐向量空间，使得这些向量之间可以进行直接比较和运算
	•	能够有效捕捉特征之间的二阶交互作用。
	•	在处理稀疏数据上表现良好。
	•	计算复杂度较低，适合大规模数据集
FM本质上是一个线性模型，不同项之间以线性组合的方式影响模型的输出
## FFM（Field-aware Factorization Machines ）
FM（Factorization Machines）和 FFM（Field-aware Factorization Machines）都是用于推荐系统和大规模稀疏数据集建模的机器学习算法

## GBDT


## Mixture-of-Experts
Mixture of Experts architectures enable large-scale models, even those comprising many billions of parameters, to greatly reduce computation costs during pre-training and achieve faster performance during inference time. Broadly speaking, it achieves this efficiency through selectively activating only the specific experts needed for a given task, rather than activating the entire neural network for every task.

## MMOE Multi-gate Mixture-of-Experts

DNN-based multi-task
learning models are sensitive to factors such as the data distribution
differences and relationships among tasks

The inherent
conicts from task dierences can actually harm the predictions of
at least some of the tasks, particularly when model parameters are
extensively shared among all tasks.

### Shared-Bottom multi-task DNN structure

## Transformer
Classic feed-forward neural networks (FFNs) process information by progressively passing input data from neurons in one layer to neurons in the following layer until it reaches an outer layer where final predictions occur. Some neural network architectures incorporate additional elements, like the self-attention mechanisms of transformer models, that capture additional patterns and dependencies in input data. 
#### AUC



