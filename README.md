# 推荐   E-commerce search
大规模推荐系统
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/325a1634-793c-424d-b36e-cc74b653e7f3)
term mismatch between queries and items

电子商务的兴起，用户并非一定是带着明确的购买意图去浏览，很多时候是去“逛”的

合理选取训练数据。选取的训练数据的时间窗口不宜过长，当然也不能过短。具体的窗口期数值需要经过多次的实验来确定。同时可以考虑引入时间衰减，因为近期的用户行为更能反映用户接下来的行为动作。

用户遇到信息过载才需要推荐系统，你确定你的产品真的需要推荐系统么？

## 负采样
用于处理推荐系统中数据稀疏和计算效率问题 

为了平衡正负样本比例，我们只使用一小部分样本进行训练

主要思想是将负样本的选择转化为随机采样的问题，

## NCE
为了解决Softmax 分母计算量过大的问题

在推荐系统中的召回模块里，许多优化例如"通过负采样打压热门物品"也是通过调节NCE等算法的参数实现的 

## NEG
NCE的简化版

## Sampled softmax


## 
### 流行度负采样

一种简单的负采样方法是根据物品的流行度进行采样，流行度较高的更有可能成为负样本，因为用户对他们已经有了较多的反馈信息 


## 截断策略
在生成推荐结果时，对结果列表进行截断或限制 
### qualityscore
基于 qualityscore 截断是一种 naive 的算法

### wand



## similarity
### 余弦相似度 业界常用
商品在表示成特征向量之后，两个特征向量之间的夹角越小，说明这两个向量越相似，也就是对应的两个商品越相似

## 优化和损失函数

第一类 PointWise，就是通过直接预估单个的物品的得分去做排序，在精排环节中最常用；

第二类叫PairWise，就是把排序问题看成是其中物品组成的任意pair，然后对比两两pair之间的顺序，所以样本就是这种物品对，这种在召回环节最常用；

第三类是ListWise算法，就是需要考虑待排序的物品中任意之间的顺序，把整个列表当作样本，一般在重排环节用的比较多。当然越后面的算法复杂度是越高，

## pointwise 
pointwise 把召回看成二元分类

pairwise 三元组

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/d9ef1c17-d716-47a3-948e-f063e5a38571)

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/a86bdf50-a53c-46ab-95d4-97254890fed9)


### 贝叶斯个性化（BPR）
训练样本就是一个UI矩阵，横轴表示所有的物品，纵轴表示所有的用户，然后矩阵里面填充的结果是用户对当前物品的打分值，可以是显式的隐式的，之前的思路是通过分解矩阵来填充里面空的格子，然后去预测用户的偏好。



### 多任务

多任务目标。比如视频推荐不仅需要预测用户是否会观看外，还希望去预测用户对于视频的评分，是否会关注该视频的上传者，否会分享到社交平台

单个用户行为并不能准确反映用户对Item的好恶，例如一个用户播放某个视频，但是最后给了一个低分。并且这些行为之间不是相互独立的，可能会结合在一起决定用户对视频的偏好。所以，我们要结合这些行为分数在一起来评价用户的对某个视频的偏好。
## 偏置信息
### 位置偏置
比如用户是否会点击和观看某个视频，并不一定是因为他喜欢，可能仅仅是因为它排在推荐页的最前面，这会导致训练数据产生位置偏置的问题。

比较常用的做法是把位置作为一个参数带入模型训练和预测过程。
###  Exposure Bias

### Popularity Bias
推荐系统数据存在长尾现象，少部分流行度高的物品占据了大多数的交互



## 数据循环
算法决定展示内容，展示内容影响用户行为，而用户行为反馈又会决定后续算法的学习，形成循环。在这种循环下，训练集和测试集与监督学习独立同分布的假设相去甚远，同时系统层面上缺乏有效探索机制的设计，可能导致模型更聚焦于局部最优。

在用户行为稀疏的场景下，数据循环问题尤其显著。问题的本质：有限的数据无法获得绝对置信的预估，探索和利用（Explore&Exploit）是突破数据循环的关键。


# 冷启动cold start solutions

cold start is equivalent to the missing data problem where preference information is missing

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
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/2feca7c5-3983-42bb-becc-7d7058f50a02)

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

为了提高多样性召回一般使用多路通道

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/8bd2b1b1-1c25-4ee1-a8d5-68599486178a)

## Co-occurrence Matrix

用于表示用户和物品之间的交互频次或相关性

用户与物品的交互往往非常稀疏，导致共同访问矩阵中大量的零值


## ItemCF i2i
i2i：通过计算item间的相似度，找到相似的item

两个物品的受众重合度越大，两个物品的相似度越大

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/ac6b8b49-b542-49d8-993d-f7157db9912f)


## Embedding召回
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/26eb301f-2bf2-4012-8592-8cdcef6c1207)
### 如何embedding


#### 基于Graph的Embedding
核心思想是根据用户行为，构造user、item的关系图，然后采用Graph embedding方法实现对节点（即user、item）的embedding向量。

## swing 召回通道
如果大量用户同时喜欢两个物品，且这些用户之间的重合度很低，那么这两个物品一定很相似。

Swing 算法与 ItemCF 算法的唯一区别在于计算物品相似度的公式不同。

## UserCF
u2i：通过计算user和item间的相似度，找到与user相似的item

## 矩阵补充
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/f8e1be1e-8058-4e36-9b2b-260ee2772ac3)

## 双塔模型 two tower architecture

![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/b125fecd-c93b-46a9-a401-46070f60aad2)

## 正负样本

## GNN召回

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
## 用户-物品矩阵的构建和处理方法
### 分解user-item矩阵



![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/66a0a534-5d2a-4afd-b10f-dd170e21b91b)

推荐物品主要包含两部分。一个是搜索词，涉及到Query意图、核心词抽取、语义完整性等NLP技术，或者Query相关的信息流消费pv、索引网页的数量、信息流的分类、不同时间窗口的趋势，还有安全相关的质量控制。

另一个是搜索结果页的呈现，包括文章样式，搜索结果基础相关性、语义相关性、权威性、时效性，文章的意图、url点击分布。
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/7a729f34-5e38-4d58-9294-1356cee76924)
## 隐语义模型 LFM(latent factor model)
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/ec5fde86-bdd1-4422-8486-0ec9b1c9bdb4)



## SVD

## biasSVD


## FM（Factorization Machines）向量分解机
FM 模型的核心思想是将高维稀疏特征映射到低维的密集向量空间。具体来说，FM 模型通过隐向量来表示特征之间的交互作用。

引入矩阵分解技术来捕捉特征之间的交互作用 通过将所有的ID特征映射到同一个隐向量空间，使得这些向量之间可以进行直接比较和运算
	•	能够有效捕捉特征之间的二阶交互作用。
	•	在处理稀疏数据上表现良好。
	•	计算复杂度较低，适合大规模数据集
FM本质上是一个线性模型，不同项之间以线性组合的方式影响模型的输出
## FFM（Field-aware Factorization Machines ）
FM（Factorization Machines）和 FFM（Field-aware Factorization Machines）都是用于推荐系统和大规模稀疏数据集建模的机器学习算法








## GeoHash 召回

## 作者召回

## 缓冲召回

## 查询query
分词和查询Query的语义理解方面做到业务可用的效果，至少需要百万级有标注的商品和电商搜索关键词数据做训练
## query自动补全
更加关注topN满足率 


## 曝光过滤
如果集合用线性表存储，查找的时间复杂度为 O(n)；

如果用平衡 BST（如 AVL树、红黑树）存储，时间复杂度为 O(logn)；

如果用哈希表存储，并用链地址法与平衡 BST 解决哈希冲突（参考 JDK8 的 HashMap 实现方法），时间复杂度也要有O[log(n/m)]，m 为哈希分桶数。
### Bloom Filter

Bloom Filter 是由一个长度为 m 的比特位数组 与 k 个哈希函数组成的数据结构

# Rank
对召回阶段输出的数百个内容，应用复杂的模型进行排序，选出其中最有可能被用户喜欢的

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



## DPP 行列式点过程的推荐多样性提升

矩阵的行列式的物理意义为矩阵中的各个向量张成的平行多面体体积的平方。这些向量彼此之间越不相似，向量间的夹角就会越大，张成的平行多面体的体积也就越大，矩阵的行列式也就越大，对应的商品集合的多样性也就越高。当这些向量彼此正交的时候，多样性达到最高




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


## LHUC（Learning Hidden Unit Contributions）
特征就是重要的先验知识。但是有时我们自认为加入了一个非常重要的特征，但是模型效果却没有提升。很有可能是你加的位置不对。

重要的特征加到DNN底部，层层向上传递，恐怕到达顶部时，也不剩多少了。另外，推荐系统中，DNN的底层往往是若干特征embedding的拼接，动辄几千维，这时你再新加入一个特征32维embedding，“泯然众人矣”，恐怕也不会太奇怪。
## PPNet 
由基础的DNN结构和Gate NN结构组成


## Wide & Deep
wide&deep框架来缓和选择偏见



![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/d3f2371e-05b0-4801-a716-8157cdc339fc)

### wide 
The wide component is a generalized linear model

The simple architecture makes the domain features directly influence the final prediction

加入wide侧的特征离最终目标也近，避免自dnn底部层层传递带来的信息损失，更有机会将我们的先验知识贯彻到“顶”。
### Deep
The deep component is a feed-forward neural network 

## DeepFM
通过将Wide & Deep模型中的Wide侧模型替换成FM模型，实现自动的交叉特征选择，从而实现无需人工参与就可以通过模型进行端到端的学习，自动学习到各种层级的交叉特征
## GBDT


## 重排




## Mixture-of-Experts
Mixture of Experts architectures enable large-scale models, even those comprising many billions of parameters, to greatly reduce computation costs during pre-training and achieve faster performance during inference time. Broadly speaking, it achieves this efficiency through selectively activating only the specific experts needed for a given task, rather than activating the entire neural network for every task.

## MMOE Multi-gate Mixture-of-Experts
优化多目标排序

DNN-based multi-task learning models are sensitive to factors such as the data distribution differences and relationships among tasks

The inherent
conticts from task dierences can actually harm the predictions of
at least some of the tasks, particularly when model parameters are extensively shared among all tasks.

### Shared-Bottom multi-task DNN structure

## NGCF
用户和物品embedding的内积作为模型的预测结果
## LightGCN
图卷积神经网络


## 自监督学习

利用无标签数据，通过自监督方法预训练模型，提高对稀疏数据的处理能力。
## Transformer
Classic feed-forward neural networks (FFNs) process information by progressively passing input data from neurons in one layer to neurons in the following layer until it reaches an outer layer where final predictions occur.

Some neural network architectures incorporate additional elements, like the self-attention mechanisms of transformer models, that capture additional patterns and dependencies in input data. 
#### AUC

### A|B 测试

随机分桶

#### 分层实验


