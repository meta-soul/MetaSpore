# 时序模型

In this project, we will implement and benchmark the algorithms of sequential model, such as [HRM](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.827.9692&rep=rep1&type=pdf), [GRU4Rec](https://arxiv.org/abs/2109.12613). It should be noted that some algorithms have not yet been implemented, and we have not sufficiently tuned the parameters of the model. We will continue to enrich our algorithm package and provide experimental results.
在这个项目中，我们会对时序算法，比如[HRM](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.827.9692&rep=rep1&type=pdf), [GRU4Rec](https://arxiv.org/abs/2109.12613)等经典时序模型，在MovieLens数据集上进行对比。需要说明对是，有些经典的算法还没实现，我们也并没有对模型进行充分调参，我们会持续丰富我们算法包和对比实验结果。

## 模型列表


|    模型     |                训练脚本                 |                        算法实现                         | 论文                                                                                                                                     |
|:---------:|:-----------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
|     HRM    |            [hrm.py](hrm/hrm.py)         |   [hrm_net.py](../../python/algos/sequential/hrm/hrm_net.py)           | [Learning Hierarchical Representation Model for Next Basket Recommendation](https://arxiv.org/pdf/1511.06939)     |
|   GRU4Rec  |    [item_cf.py](gru4rec/gru4rec.py)     |   [gru4rec_net.py](../../python/algos/sequential/hrm/gru4rec_net.py)   | [SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1511.06939)     |
我们正在不断的添加新模型。

## 测试结果

| Model | Dataset | Precision@20 | Recall@20 | MAP@20 | NDCG@20 | 
|:--------------:|:--------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| HRM | MovieLens-1M | 0.007319 | 0.146393 | 0.026403 | 0.051770 |
| GRU4Rec | MovieLens-1M | 0.007686 | 0.153733 | 0.033994 | 0.059516 |

对于基于神经网络的时序模型，我们给出了它们各自实验结果。实验结果并未进行充分调参，后续相关实验结果会进行更新。

## 如何运行
### 1. 数据预处理和特征生成
对于MovieLens数据集, 我们目前仅用了`user_id`，`movie_id`，`recent_movie_id`，`last_movie`作为模型的特征。特征的生成过程可以参考我们的数据处理和准备的[说明](../dataset/README-CN.md) 。

### 2. 特征描述文件上传
对于神经网络模型而言，在MetaSpore中需要对数据对特征列、特征交叉的情况进行描述，并将[schema](dssm/schema)文件上传到S3存储中。比如以`HRM`模型为例，假设我们在这个项目的根目录中，
我们需要执行以下命令：

```shell
aws s3 cp --recursive hrm/schema/.* ${MY_S3_BUCKET}/movielens/1m/schema/hrm/
```

### 3. 初始化训练配置文件
通过替换对应`YAML`模板中的变量，初始化我们需要的模型配置文件，举例来说:
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < hrm/conf/hrm_bce_neg10_ml_1m.yaml > hrm/conf/hrm_bce_neg10_ml_1m.yaml.dev
```

### 4. 运行模型测试脚本
我们现在可以运行训练脚本了。举例来说，用MovieLens-1M数据集训练并测试一个HRM模型，需要执行这样的命令:
```shell
cd hrm
python hrm.py --conf conf/hrm_bce_neg10_ml_1m.yaml.dev > log/hrm_bce_neg10_ml_1m.log 2>&1 &
```