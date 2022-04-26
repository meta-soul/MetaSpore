## [中文介绍](README-CN.md)

# Data Processing and Preparation
In this project, data processing is unified for MovieLens-1M, MovieLens-25M, Criteo-5d and other datasets, including feature generation,  matching dataset generation, ranking dataset generation, negative sampling, etc.  If you are Chinese developer, you may like to visit our [CN Doc](README-CN.md).

## Initialize the Configuration Files
First of all, we should initialize the config files from their YAML template for substituting some variables. For example
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < fg.yaml > fg.yaml.dev 
```
For the latter stages, we assume that we have done this to generate the available configurations before running the python scripts. If you have not install `envsubst`, you can run `sudo apt-get install gettext-base` assumming you are running this scripts on Debian based Linux systems.

## MovieLens-1M
In this section, we use [MoiveLens-1M](https://grouplens.org/datasets/movielens/1m/) to demonstrate. You can download these datasets from the provided urls and store these files onto your own cloud storage. 

### 1. Feature Generation

Assuming we are in root directory of this project, we can execute the following commands to get the result of feature generation processing.

 ```shell
 cd ml_1m
 python fg.py --conf fg.yaml.dev --verbose
 ```

### 2. Matching Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `Collaborative Filtering` algorithms.

```shell
cd ml_1m
python match_dataset_cf.py --conf match_dataset.yaml.dev --verbose
```

After that we can get the train and test dataset of `TwoTowers` models through following command:

```shell
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose
```

### 3. Ranking Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of [CTR models](../ctr/README.md):

```shell
python rank_dataset.py --conf rank.yaml.dev --verbose
```

## MovieLens-25M
In this section, we use [MoiveLens-25M](https://grouplens.org/datasets/movielens/25m/) to demonstrate. You can download these datasets from the provided urls and store these files onto your own cloud storage.

### 1. Feature Generation

Assuming we are in root directory of this project, we can execute the following commands to get the result of feature generation processing.

 ```shell
 cd ml_25m
 python fg.py --conf fg.yaml.dev --verbose
 ```

### 2. Matching Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of `TwoTowers` models.

```shell
cd ml_25m
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose
```

### 3. Ranking Dataset
Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of [CTR models](../ctr/README.md):

```shell
cd ml_25m
python rank_dataset.py --conf rank.yaml.dev --verbose
```

## Criteo-5D
In this section, we use the publicly available dataset [Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/) published by CriteoLabs as our demo dataset. 
```python
import metaspore
metaspore.demo.download_dataset()
```

If the downloading fails, please refer to [MetaSpore Demo Dataset](https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/demo/criteo/index.html) and download the dataset manually. We normalize numerical values by transforming from a value `z` to `floor(log(z)^2)` if `z > 2`, which is proposed by the winner of [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) in [3 Idiots' Approach](https://github.com/ycjuan/kaggle-2014-criteo). 

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

Assuming we are in root directory of this project, we can execute the following commands to get the train and test dataset of [CTR models](../ctr/README.md):

```shell
cd criteo
python fg.py --conf fg_5d.yaml.dev --verbose
```
