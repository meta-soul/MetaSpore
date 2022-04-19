# CTR Demo

Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page,
email, or advertisement.
In this demo, we will implement the most effective CTR models, and give the benchmarks of MovieLens and Criteo dataset.

## Model list
We are continuously adding models:

|   Model   |            Train script             |                     Net implement                      | Paper                                                                                                             |
|:---------:|:-----------------------------------:|:------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------|
| Wide&Deep | [widedeep.py](widedeep/widedeep.py) | [widedeep_net.py](../../python/algos/widedeep_net.py)  | [Wide & Deep Learning for Recommender Systems](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454)                |
|  DeepFM   |    [deepfm.py](deepfm/deepfm.py)    |   [deepfm_net.py](../../python/algos/deepfm_net.py)    | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)   |
|    DCN    |        [dcn.py](dcn/dcn.py)         |      [dcn_net.py](../../python/algos/dcn_net.py)       | [DeepAndCross: Deep & Cross Network for Ad Click Predictions](https://dl.acm.org/doi/pdf/10.1145/3124749.3124754) |


## Benchmarks
|   Model   |                                       criteo-d5 |                                           ml-1m |                                          ml-25m |
|:---------:|------------------------------------------------:|------------------------------------------------:|------------------------------------------------:|
| Wide&Deep | Train AUC:  `0.7394` <br /> Test AUC:  `0.7294` | Train AUC:  `0.8937` <br /> Test AUC:  `0.8682` | Train AUC:  `0.8898` <br /> Test AUC:  `0.8343` |
|  DeepFM   | Train AUC:  `0.7531` <br /> Test AUC:  `0.7271` | Train AUC:  `0.8891` <br /> Test AUC:  `0.8658` | Train AUC:  `0.8908` <br /> Test AUC:  `0.8359` |
|    DCN    | Train AUC:  `0.7413` <br /> Test AUC:  `0.7304` | Train AUC:  `0.9021` <br /> Test AUC:  `0.8746` | Train AUC:  `0.8972` <br /> Test AUC:  `0.8430` |


## How to run

### Data preprocessing and feature generation
For MovieLens dataset, we just use user_id and movie_id of as model's features for now.

For Criteo dataset, we normalize numerical values by transforming from a value z to log(z) if z > 2, which is proposed by the winner of Criteo Competition.
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
Moreover, we use the training data of the first 5 days provided by the competition as the training set, and the test data provided by the first day as the test set.

We will update the related code as soon as possible.

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