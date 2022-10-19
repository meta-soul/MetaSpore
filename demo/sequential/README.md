## [中文介绍](README-CN.md)

In this project, we will implement and benchmark the algorithms of sequential models, such as [HRM](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.827.9692&rep=rep1&type=pdf), [GRU4Rec](https://arxiv.org/abs/2109.12613). We will continue to enrich our algorithm package and provide new experimental results.

## Model List
 
|    Model   |                Training                 |                              Implementation                            |            Paper              |
|:----------:|:---------------------------------------:|:----------------------------------------------------------------------:|:------------------------------|
|     HRM    |            [hrm.py](hrm/hrm.py)         |   [hrm_net.py](../../python/algos/sequential/hrm/hrm_net.py)           | [Learning Hierarchical Representation Model for Next Basket Recommendation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.827.9692&rep=rep1&type=pdf)     |
|   GRU4Rec  |    [gru4rec.py](gru4rec/gru4rec.py)     |   [gru4rec_net.py](../../python/algos/sequential/hrm/gru4rec_net.py)   | [SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS](https://arxiv.org/pdf/1511.06939)     |

We are continuously adding new models.

## Benchmark Result

|  Model  |           Dataset         | Precision@20 | Recall@20 |  MAP@20  |  NDCG@20 | 
|:-------:|:--------------------------|:------------:|:---------:|:--------:|:--------:|
| HRM     | MovieLens-1M NegSample-10 |   0.010479   | 0.209581  | 0.041875 | 0.077521 |
| GRU4Rec | MovieLens-1M              |   0.008594   | 0.171882  | 0.034879 | 0.063995 |

For the sequential model based on neural network, we give the respective experimental results based on different loss functions and different negative sampling ratio. As our models and experiments are tuned, this experiment results may be updated in the future.

## How to Run
### 1. Data processing and feature generation
For the MovieLens dataset, we currently only use `user_id`, `movie_id`, `recent_movie_id`, `last_movie` as model features. The feature generation process can refer to our data processing and preparation [description](../dataset/README.md).

### 2. Upload the schema file of features
For the neural network model, in MetaSpore, it is necessary to describe the feature column and feature combination column in [schema](schema) file, and upload these files to S3 storage. For example, taking the `HRM` model as an example, suppose we are in the root directory of this project,
we need to execute the following commands:

```shell
aws s3 cp --recursive hrm/schema/.* ${MY_S3_BUCKET}/movielens/1m/schema/hrm/
```

### 3. Initialize the configuration file 
Before running models, we should initialize the config files from their `YAML` template for substituting some variables. 

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < hrm/conf/hrm_bce_neg10_ml_1m.yaml > hrm/conf/hrm_bce_neg10_ml_1m.yaml.dev
```

### 4. Running Models
Now, We can run the training scripts. For example, to train and test a `HRM` model with the MovieLens-1M dataset, we can execute a command like this:

```shell
cd hrm
python hrm.py --conf conf/hrm_bce_neg10_ml_1m.yaml.dev > log/hrm_bce_neg10_ml_1m.log 2>&1 &
```

