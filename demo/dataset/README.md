## [中文介绍](README-CN.md)

# Data Processing and Preparation
In this project, data processing is unified for MovieLens-1M, MovieLens-25M, Criteo-5d and other datasets, including feature generation,  matching dataset generation, ranking dataset generation, negative sampling, etc. If you are Chinese developer, you may like to visit our [CN Doc](README-CN.md).


Here is the overview of the datasets:

| Datasets                        | How to use in MetaSpore                            | References                                                                                                                             |
|:--------------------------------|:---------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| [MovieLens-1M](#MovieLens-1M)   | [Movie Recommendation End2End Demo](../movielens/) | [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)                                                                   |
| [MovieLens-25M](#MovieLens-25M) | [CTR Demo](../ctr/)                                | [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/1m/)                                                                  |
| [Criteo-5D](#Criteo-5D)         | [CTR Demo](../ctr/)                                | [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)                                                 |
| [Census](#Census)               | [MMoE Demo](../multitask/mmoe/)                    | [Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid](http://robotics.stanford.edu/~ronnyk/nbtree.pdf)          |
| [Ali-CCP](#Ali-CCP)             | [MMoE Demo](../multitask/esmm/)                    | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf) |

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

## Census
In this section, we use the publicly available dataset [Census](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz) as our demo dataset. 
### Data preprocessing
```shell
cd census
sh data_process.sh
```
We extract 'marital_stat' and 'income_50k' as two labels of multitask model. And transform continuous features using:
```python
import numpy as np
def fun3(x):
    return np.log(x+1).astype(int)
```
Moreover, we don't need to transform categorical features to one-hot embeddings because MetaSpore can handle embedding layer automatically.

## Ali-CCP
In this section, we will introduce how to process [Ali-CCP](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408) dataset. Original dataset is very large, we will use two subset of this data provided by [PaddleRec](https://github.com/PaddlePaddle/PaddleRec): 
* **[Small subset](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/ali-ccp)**: a dataset contains 10,000 training and test samples approximately.
* **[Large subset](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/ali-cpp_aitm)**: a dataset contains 38,000,000 training and 43,000,000 test samples approximately.

### Download Data
Assuming we are in root directory of this project, we can execute the following commands to download these two versions of Ali-CCP dataset and upload them into your S3 bucket.

 ```shell
cd aliccp
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < data_processing.sh > data_processing_dev.sh
data_processing_dev.sh
 ```

### Feature Generation
After the download is completed, we can run our provided Python scripts to generate features and labels that are able to used in MetaSpore.

```python
# small dataset
python fg_small_dataset.py --conf fg_small_dataset.yaml.dev
# large dataset
python fg_large_dataset.py --conf fg_large_dataset.yaml.dev
```
