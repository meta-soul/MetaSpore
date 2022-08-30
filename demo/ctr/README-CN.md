# 点击率模型的演示

点击率(Click-through rate)是物料(Item)等被点击次数，与物料被显示次数的比率，是一种衡量物料排序的指标。
在这个Demo中，我们实现了近年来那些在业界中常用的CTR模型，并给出这些模型在MovieLens及Criteo数据集上的评测结果。

## 模型列表
我们正在不断的添加新模型:

|    模型     |                训练脚本                 |                        模型网络实现                         | 论文                                                                                                                                     |
|:---------:|:-----------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| Wide&Deep | [widedeep.py](widedeep/widedeep.py) | [widedeep_net.py](../../python/algos/widedeep_net.py) | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                                     |
|  DeepFM   |    [deepfm.py](deepfm/deepfm.py)    |   [deepfm_net.py](../../python/algos/deepfm_net.py)   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                        |
|    DCN    |        [dcn.py](dcn/dcn.py)         |      [dcn_net.py](../../python/algos/dcn_net.py)      | [DeepAndCross: Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)                      |
|  DCN V2   |    [dcn_v2.py](dcn_v2/dcn_v2.py)    |   [dcn_v2_net.py](../../python/algos/dcn_v2_net.py)   | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) |
|    PNN    |        [pnn.py](pnn/pnn.py)         |      [pnn_net.py](../../python/algos/pnn_net.py)      | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                     |
|  AutoInt  |  [autoint.py](autoint/autoint.py)   |  [autoint_net.py](../../python/algos/autoint_net.py)  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                 |
|  xDeepFM  |  [xdeepfm.py](xdeepfm/xdeepfm.py)   |  [xdeepfm_net.py](../../python/algos/xdeepfm_net.py)  | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)          |
|   FwFM    |       [fwfm.py](fwfm/fwfm.py)       |     [fwfm_net.py](../../python/algos/fwfm_net.py)     | [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf) |
|   FFM     |     [ffm.py](ffm/ffm.py)|  [ffm_net.py](../../python/algos/ffm_net.py)  | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)


## 基准
|    模型     |                                       criteo-d5 |                                           ml-1m |                                          ml-25m |
|:---------:|------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
| Wide&Deep | Train AUC:  `0.7394` <br /> Test AUC:  `0.7294` | Train AUC:  `0.8937` <br /> Test AUC:  `0.8682` | Train AUC:  `0.8898` <br /> Test AUC:  `0.8343` |
|  DeepFM   | Train AUC:  `0.7531` <br /> Test AUC:  `0.7271` | Train AUC:  `0.8891` <br /> Test AUC:  `0.8658` | Train AUC:  `0.8908` <br /> Test AUC:  `0.8359` |
|    DCN    | Train AUC:  `0.7413` <br /> Test AUC:  `0.7304` | Train AUC:  `0.9021` <br /> Test AUC:  `0.8746` | Train AUC:  `0.8972` <br /> Test AUC:  `0.8430` |
|  DCN V2   | Train AUC:  `0.7487` <br /> Test AUC:  `0.7290` | Train AUC:  `0.8901` <br /> Test AUC:  `0.8611` | Train AUC:  `0.8888` <br /> Test AUC:  `0.8323` |
|   iPNN    | Train AUC:  `0.7544` <br /> Test AUC:  `0.7292` | Train AUC:  `0.8914` <br /> Test AUC:  `0.8649` | Train AUC:  `0.8916` <br /> Test AUC:  `0.8362` |
|   oPNN    | Train AUC:  `0.7533` <br /> Test AUC:  `0.7287` | Train AUC:  `0.8896` <br /> Test AUC:  `0.8633` | Train AUC:  `0.8905` <br /> Test AUC:  `0.8353` |
|  AutoInt  | Train AUC:  `0.7558` <br /> Test AUC:  `0.7361` | Train AUC:  `0.9028` <br /> Test AUC:  `0.8741` | Train AUC:  `0.8968` <br /> Test AUC:  `0.8421` |
|  xDeepFM  | Train AUC:  `0.7541` <br /> Test AUC:  `0.7300` | Train AUC:  `0.8892` <br /> Test AUC:  `0.8641` | Train AUC:  `0.8911` <br /> Test AUC:  `0.8367` |
|   FwFM    | Train AUC:  `0.7517` <br /> Test AUC:  `0.7298` | Train AUC:  `0.8911` <br /> Test AUC:  `0.8660` | Train AUC:  `0.8918` <br /> Test AUC:  `0.8376` |
|   FFM     | Train AUC:  `0.7518` <br /> Test AUC:  `0.7280` | Train AUC:  `0.8919` <br /> Test AUC:  `0.8672` | Train AUC:  `0.8921` <br /> Test AUC:  `0.8379` |

## 如何运行

### 数据预处理和特征生成
对于MovieLens数据集, 我们目前仅用了`user_id`和`movie_id`作为模型的特征。可以参考我们的数据处理和准备的[说明](../dataset/README.md) 。

对于Criteo数据集，我们标准化所有数值型特征`z`，具体方法是用`floor(log(z)^2)`代替所有大于2的`z`。此方法由Criteo竞赛的获胜者提出。
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
此外，我们用比赛提供的训练集中的头部五天训练数据，作为我们的训练集。用比赛提供的测试集中的头部一天测试数据，作为我们的测试集。可以参考我们的数据处理和准备的[说明](../dataset/README.md) 。

### 初始化模型配置文件
首先, 上传 [schema](schema) 到你的S3云存储。
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