# 图模型

在这个项目中，我们会实现比如基于jaccard距离的召回模型, 基于euclidean距离的召回模型，下面简称为Jaccard和Euclidean。我们在MovieLens数据集上进行对比。我们会持续丰富我们算法包和对比实验结果。

## 模型列表


|    模型     |                训练脚本                 |                        算法实现                         | 论文                                                                                                                                     |
|:---------:|:-----------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
|  Jaccard  |      [jaccard.py](jaccard/jaccard.py)    |   [jaccard_retrieval.py](../../python/algos/graph/jaccard/jaccard_retrieval.py)         | `-` |
| Euclidean | [euclidean.py](euclidean/euclidean.py)   |   [euclidean_retrieval.py](../../python/algos/graph/euclidean/euclidean_retrieval.py)   | `-` |
我们正在不断的添加新模型。

## 测试结果

|  Model  |           Dataset         | Precision@20 | Recall@20 |  MAP@20  | 
|:-- ----:|:--------------------------|:------------:|:---------:|:--------:|
| Jaccard | MovieLens-1M NegSample-10 |  0.00429635  | 0.08592715|0.02708586|
|Euclidean| MovieLens-1M              |  0.00019867  | 0.00397350|0.00228660|

对于基于物品相似度的召回模型，我们给出了它们各自实验结果。实验结果仅供参考。

## 如何运行
### 1. 数据准备
对于MovieLens数据集, 我们目前仅用了`user_id`，`movie_id`作为模型的特征。

### 2. 初始化训练配置文件
通过替换对应`YAML`模板中的变量，初始化我们需要的模型配置文件，举例来说:
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < jaccard/conf/jaccard.yaml > jaccard/conf/jaccard.yaml.dev
```

### 3. 运行模型测试脚本
我们现在可以运行训练脚本了。举例来说，用MovieLens-1M数据集训练并测试Jaccard模型，需要执行这样的命令:
```shell
cd jaccard
python jaccard.py --conf conf/jaccard.yaml.dev > log/jaccard_ml_1m.log 2>&1 &
```