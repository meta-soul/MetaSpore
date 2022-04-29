# 双塔模型

在推荐系统中，一般说来我们服务器的资源永远是有限的，同时用户对系统响应时长的耐心也是有限的。召回阶段会从 `百万`～`亿` 数量级的物料集合中，快速筛选出那些与用户需求并不匹配的物料，而让剩余相关的物料进入到系统的之后处理流程中。近些年随着深度网络的推陈出新，使用神经网路做召回的工作也不断涌现出来，尤其是双塔模型，结构简单，效果好，已经成为召回阶段的标配算法。 同时，一些经典的 [协同过滤](https://en.wikipedia.org/wiki/Collaborative_filtering) 算法，虽然已经出现了很长时间了，但是一些 [新的研究](https://arxiv.org/abs/1907.06902) 表明，这些方法依然是非常强的baseline，具有实际的应用价值。 

<p align="center">
 <img width="600" alt="image" src="https://user-images.githubusercontent.com/7464971/165916173-49d26410-91cd-408d-bbb3-18ca43d877b6.png">
</p>

在这个项目中，我们会对双塔结构的算法，比如 [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf) 、[SimpleX](https://arxiv.org/abs/2109.12613) 等经典双塔结构，以及[Swing](https://arxiv.org/abs/2010.05525) 、[Item CF](https://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf) ，在MovieLens数据集上进行对比。需要说明对是，有些经典的算法还没实现，我们也并没有对模型进行充分调参，我们会持续丰富我们算法包和对比实验结果。

## 模型列表


|    模型     |                训练脚本                 |                        算法实现                         | 论文                                                                                                                                     |
|:---------:|:-----------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| Global Hot | [global_hot.py](baseline/global_hot.py) | - | -                                     |
| Item CF I2I  |    [item_cf.py](baseline/item_cf.py)    |   [item_cf_retrieval.py](../../python/algos/item_cf_retrieval.py)   | [WWW 2010] [Item-Based Collaborative Filtering Recommendation Algorithms](https://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf)  |
| Swing I2I  |   [swing.py](baseline/swing.py)    |  [swing_retrieval.py](../../python/metaspore/swing_retrieval.py)   | [arxiv 2020] [Large Scale Product Graph Construction for Recommendation in E-commerce](https://arxiv.org/abs/2109.12613)  | 
| ALS MF  |   [spark_als.py](baseline/spark_als.py)    |   [Spark Mllib ALS](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.recommendation.ALS.html)   | [ICDM 2008] [Collaborative Filtering for Implicit Feedback Datasets](http://www.yifanhu.net/PUB/cf.pdf)  | 
| DSSM  |   [dssm.py](dssm/dssm.py)    |   [dssm_net.py](../../python/algos/dssm_net.py)   | [CIKM 2013] [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)  |
| SimpleX  |   -    |   [simplex_net.py](../../python/algos/simplex/simplex_net.py)   | [CIKM 2021] [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)  |
我们正在不断的添加新模型。

## 测试结果

| 模型 | 数据集 | Precision@20 | Recall@20 | MAP@20 | NDCG@20 | 
|:--------------:|:--------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| Global Hot | MovieLens-1M | 0.002477| 0.049533 | 0.008923 | 0.017346|
| Spark ALS | MovieLens-1M | 0.002472 | 0.049444 | 0.015736 | 0.017743 |
| Swing I2I | MovieLens-1M | 0.006334 | 0.126674 | 0.029579 | 0.050461 |
| Item CF I2I | MovieLens-1M | 0.009383 | 0.187667 | 0.050912 | 0.080504 |
| DSSM BCE | MovieLens-1M NegSample-10 | 0.010776 | 0.215533 | 0.043305 | 0.080013 |
| DSSM BCE | MovieLens-1M NegSample-100 | 0.011313 | 0.226264 | 0.047736 | 0.085856 |
| SimpleX CCL | MovieLens-1M NegSample-100 | - | - | - | - |

对于基于神经网络双塔模型，我们基于不同的损失函数、不同负采样数量，分别给出了各自实验结果。实验结果并未进行充分调参，后续相关实验结果会进行更新。

## 如何运行
### 1. 数据预处理和特征生成
对于MovieLens数据集, 我们目前仅用了`user_id`，`movie_id`，`recent_movie_id`，`last_movie`作为模型的特征。特征的生成过程可以参考我们的数据处理和准备的[说明](../dataset/README-CN.md) 。

### 2. 特征描述文件上传
对于神经网络模型而言，在MetaSpore中需要对数据对特征列、特征交叉的情况进行描述，并上[schema](dssm/schema)文件上传到S3存储中。比如以`DSSM`模型为例，假设我们在这个项目的根目录中，
我们需要执行以下命令：

```shell
aws s3 cp --recursive dssm/schema/ml_1m/.* ${MY_S3_BUCKET}/movielens/1m/schema/dssm/
```

### 3. 初始化训练配置文件
通过替换对应`YAML`模板中的变量，初始化我们需要的模型配置文件，举例来说:
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < dssm/conf/dssm_bce_neg10_ml_1m.yaml > dssm/conf/dssm_bce_neg10_ml_1m.yaml.dev
```

### 4. 运行模型测试脚本
我们现在可以运行训练脚本了。举例来说，用MovieLens-1M数据集训练并测试一个DSSM模型，需要执行这样的命令:
```shell
cd dssm
python dssm.py --conf conf/dssm_bce_neg10_ml_1m.yaml.dev > log/dssm_bce_neg10_ml_1m.log 2>&1 &
```






