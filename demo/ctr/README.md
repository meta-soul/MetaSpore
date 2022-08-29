## [中文介绍](README-CN.md)

# CTR Demo

Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page,
email, or advertisement.
In this demo, we will implement the most effective CTR models, and give the benchmarks of MovieLens and Criteo dataset.

## Model list
We are continuously adding models:

|   Model   |            Train script             |                     Model implementation                     | Paper                                                                                                                                  |
|:---------:|:-----------------------------------:|:-----------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| Wide&Deep | [widedeep.py](widedeep/widedeep.py) | [widedeep_net.py](../../python/algos/widedeep_net.py) | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                                     |
|  DeepFM   |    [deepfm.py](deepfm/deepfm.py)    |   [deepfm_net.py](../../python/algos/deepfm_net.py)   | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)                        |
|    DCN    |        [dcn.py](dcn/dcn.py)         |      [dcn_net.py](../../python/algos/dcn_net.py)      | [DeepAndCross: Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754)                      |
|  DCN V2   |    [dcn_v2.py](dcn_v2/dcn_v2.py)    |   [dcn_v2_net.py](../../python/algos/dcn_v2_net.py)   | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) |
|    PNN    |        [pnn.py](pnn/pnn.py)         |      [pnn_net.py](../../python/algos/pnn_net.py)      | [Product-based Neural Networks for User Response Prediction](https://arxiv.org/pdf/1611.00144.pdf)                                     |
|  AutoInt  |  [autoint.py](autoint/autoint.py)   |  [autoint_net.py](../../python/algos/autoint_net.py)  | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                 |
|  xDeepFM  |  [xdeepfm.py](xdeepfm/xdeepfm.py)   |  [xdeepfm_net.py](../../python/algos/xdeepfm_net.py)  | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)          |
| FFM            | [deepffm.p](deepffm/deepffm.py)|  [deepffm_net.py](../../python/algos/deepffm_net.py)  | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

## Benchmarks
|   Model   |                                       criteo-d5 |                                           ml-1m |                                          ml-25m |
|:---------:|------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
| Wide&Deep | Train AUC:  `0.7394` <br /> Test AUC:  `0.7294` | Train AUC:  `0.8937` <br /> Test AUC:  `0.8682` | Train AUC:  `0.8898` <br /> Test AUC:  `0.8343` |
|  DeepFM   | Train AUC:  `0.7531` <br /> Test AUC:  `0.7271` | Train AUC:  `0.8891` <br /> Test AUC:  `0.8658` | Train AUC:  `0.8908` <br /> Test AUC:  `0.8359` |
|    DCN    | Train AUC:  `0.7413` <br /> Test AUC:  `0.7304` | Train AUC:  `0.9021` <br /> Test AUC:  `0.8746` | Train AUC:  `0.8972` <br /> Test AUC:  `0.8430` |
|  DCN V2   | Train AUC:  `0.7487` <br /> Test AUC:  `0.7290` | Train AUC:  `0.8901` <br /> Test AUC:  `0.8611` | Train AUC:  `0.8888` <br /> Test AUC:  `0.8323` |
|   iPNN    | Train AUC:  `0.7544` <br /> Test AUC:  `0.7292` | Train AUC:  `0.8914` <br /> Test AUC:  `0.8649` | Train AUC:  `0.8916` <br /> Test AUC:  `0.8362` |
|   oPNN    | Train AUC:  `0.7533` <br /> Test AUC:  `0.7287` | Train AUC:  `0.8896` <br /> Test AUC:  `0.8633` | Train AUC:  `0.8905` <br /> Test AUC:  `0.8353` |
|  AutoInt  | Train AUC:  `0.7558` <br /> Test AUC:  `0.7361` | Train AUC:  `0.9028` <br /> Test AUC:  `0.8741` | Train AUC:  `0.8968` <br /> Test AUC:  `0.8421` |
|  xDeepFM  | Train AUC:  `0.7541` <br /> Test AUC:  `0.7300` | Train AUC:  `0.8892` <br /> Test AUC:  `0.8641` | Train AUC:  `0.8911` <br /> Test AUC:  `0.8367` |
|  DeepFFM  | Train AUC:  `0.7518` <br /> Test AUC:  `0.7280` | Train AUC:  `0.8919` <br /> Test AUC:  `0.8672` | Train AUC:  `0.8921` <br /> Test AUC:  `0.8379` |

## How to run

### Data preprocessing and feature generation
For MovieLens dataset, we just use `user_id` and `movie_id` of as model's features for now. Please refer to this [guide](../dataset/README.md) to processing and prepare the dataset.

For Criteo dataset, we normalize numerical values by transforming from a value `z` to `floor(log(z)^2)` if `z > 2`, which is proposed by the winner of Criteo Competition. 
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
Moreover, we use the training data of the first 5 days provided by the competition as the training set, and the test data provided by the first day as the test set. Please refer to this [guide](../dataset/README.md) to processing and prepare the dataset.

### Initialize the configuration files for models
First, upload the [schema](schema) to you S3 storage.
Then initialize the config files from their `YAML` template for substituting some variables. For example:
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < template.yaml > output.yaml 
```

### Train model
We could run the training script now. For example, to train a Wide & Deep model using MovieLens 25M dataset:
```shell
cd MetaSpore/demo/ctr/widedeep
python widedeep.py --conf conf/widedeep_ml_25m.yaml
```