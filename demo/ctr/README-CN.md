# 点击率模型的演示

点击率(Click-through rate)是物料(Item)等被点击次数，与物料被显示次数的比率。是一种衡量物料热门程度的指标。
在这个Demo中，我们实现了近年来那些在业界中效果最好的CTR模型，并给出这些模型在MovieLens及Criteo数据集上的基准(Benchmark)。

## 模型列表
我们正在不断的添加新模型:

|    模型     |                 训练脚本                 |                        模型网络实现                         | 论文                                                                                                                |
|:---------:|:------------------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------|
| Wide&Deep | [widedeep.py](widedeep/widedeep.py)  | [widedeep_net.py](../../python/algos/widedeep_net.py) | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                |
|  DeepFM   |    [deepfm.py](deepfm/deepfm.py)     |   [deepfm_net.py](../../python/algos/deepfm_net.py)   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)   |
|    DCN    |         [dcn.py](dcn/dcn.py)         |      [dcn_net.py](../../python/algos/dcn_net.py)      | [DeepAndCross: Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) |


## 基准
|    模型     |                                       criteo-d5 |                                           ml-1m |                                          ml-25m |
|:---------:|------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
| Wide&Deep | Train AUC:  `0.7394` <br /> Test AUC:  `0.7294` | Train AUC:  `0.8937` <br /> Test AUC:  `0.8682` | Train AUC:  `0.8898` <br /> Test AUC:  `0.8343` |
|  DeepFM   | Train AUC:  `0.7531` <br /> Test AUC:  `0.7271` | Train AUC:  `0.8891` <br /> Test AUC:  `0.8658` | Train AUC:  `0.8908` <br /> Test AUC:  `0.8359` |
|    DCN    | Train AUC:  `0.7413` <br /> Test AUC:  `0.7304` | Train AUC:  `0.9021` <br /> Test AUC:  `0.8746` | Train AUC:  `0.8972` <br /> Test AUC:  `0.8430` |


## 如何运行

### 数据预处理和特征生成
对于MovieLens数据集, 我们目前仅用了`user_id`和`movie_id`作为模型的特征.

对于Criteo数据集，我们标准化所有数值型特征`z`，具体方法是用`log(z)`代替所有大于2的`z`。此方法由Criteo竞赛的获胜者提出。
```python
import numpy as np
def transform_number(x):
    value = -1
    try:
        if x is not None:
            value = float(x)
    except ValueError:
        pass
    return int(np.floor(np.log(value) ** 2)) if value>2.0 else int(value)
```
此外，我们用比赛提供的训练集中的头部五天训练数据，作为我们的训练集。用比赛提供的测试集中的头部一天测试数据，作为我们的测试集。

我们会尽快更新这部分代码。

### 初始化模型配置文件
首先, 上传 [schema](schema) 到你的3s云存储.
然后通过替换对应`YAML`模板中的变量，初始化我们需要的模型配置文件，举例来说:
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < template.yaml > output.yaml 
```

### 训练模型
我们现在可以运行训练脚本了. 举例来说，用MovieLens 25M数据集训练一个Wide&Deep模型，需要执行这样的命令:
```shell
cd MetaSpore/demo/ctr/widedeep
python widedeep.py --conf conf/widedeep_ml_25m.yaml
```