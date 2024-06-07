# 推荐
推荐物品主要包含两部分。一个是搜索词，涉及到Query意图、核心词抽取、语义完整性等NLP技术，或者Query相关的信息流消费pv、索引网页的数量、信息流的分类、不同时间窗口的趋势，还有安全相关的质量控制。

另一个是搜索结果页的呈现，包括文章样式，搜索结果基础相关性、语义相关性、权威性、时效性，文章的意图、url点击分布。
![image](https://github.com/zhang-mickey/Learn-to-rank-model/assets/145342600/7a729f34-5e38-4d58-9294-1356cee76924)

## query自动补全
更加关注topN满足率 
# Learn to rank

## Competition
The competition for this assignment is held on Kaggle platform. Click [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment) for detail.
Besides, the original competition can be found on Kaggle as well by clicking [this](https://www.kaggle.com/c/expedia-personalized-sort).
### Dataset 
The dataset is available on [here](https://www.kaggle.com/competitions/dmt-2024-2nd-assignment/data).
### Metric
The evaluation metric for this competition is Normalized Discounted Cumulative Gain.
See https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG for more details.

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

## Learn to rank

### RankNet
特征工程不足: 如果特征工程不足，即未能提取出对排序任务有意义的特征，那么简单的线性模型可能更容易理解和适应数据

# temporal data
##
Kalman Filter: identifies outliers and also replaces these with new values
