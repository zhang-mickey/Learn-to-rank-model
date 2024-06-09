# 推荐
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/66a0a534-5d2a-4afd-b10f-dd170e21b91b)

推荐物品主要包含两部分。一个是搜索词，涉及到Query意图、核心词抽取、语义完整性等NLP技术，或者Query相关的信息流消费pv、索引网页的数量、信息流的分类、不同时间窗口的趋势，还有安全相关的质量控制。

另一个是搜索结果页的呈现，包括文章样式，搜索结果基础相关性、语义相关性、权威性、时效性，文章的意图、url点击分布。
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/7a729f34-5e38-4d58-9294-1356cee76924)
## 查询query
分词和查询Query的语义理解方面做到业务可用的效果，至少需要百万级有标注的商品和电商搜索关键词数据做训练
## query自动补全
更加关注topN满足率 

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
# Learn to rank
将商家排序问题建模为点击率、转化率、客单价等指标的预测问题
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/e48876bd-6d11-416a-911f-e4c50cb625de)

## 
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/dfbcd1c7-5b9f-42dd-a952-a53605b5b00a)

## Competition
The competition for this assignment is held on Kaggle platform. Click [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment) for detail.
Besides, the original competition can be found on Kaggle as well by clicking [this](https://www.kaggle.com/c/expedia-personalized-sort).
### Dataset 
The dataset is available on [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment/data).
### Metric
The evaluation metric for this competition is Normalized Discounted Cumulative Gain.

## Exploring the dataset
[//]: # (In EDA, )
### 1. Overview of dataset
View the overall situation of the data set by functions like `head()`, `tail()`, `describe()` and `info()`.
Those information consists of mean value, data type, data size, non-null count and so on.

### 2. Missing data and anomalies
Features are classified into date, category, numerical, and text features. 
Check the missing rate, number of categories and outliers of each dimension feature.

### Converting the raw data to an aggregated data format

### Handking Noise and Missing Values

#### outlier removal

##### probability distribution
Chauvenet's criterion

### 3. Normalize
group by prop_id

#### 
众数 平均数来填充
### 4. Correlation analysis and feature selection
![image](https://github.com/montpelllier/VU-DMT-A2/assets/145342600/014b2e93-aa79-45e3-b8ae-c5f46e3a1693)

## GBDT

### 特征工程
• 按照小时数hour划分
用户行为数据，取前后
n个小时数据进行衰减
后线性加权

• 按照星期数weekday
划分用户行为数据，按
照不同的相似度关系衰
减后线性加权


 
### Elasticsearch  全文检索

#### 倒排索引

