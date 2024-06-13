# 推荐   E-commerce search
大规模推荐系统
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/325a1634-793c-424d-b36e-cc74b653e7f3)
term mismatch between queries and items
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
## GBDT




 


