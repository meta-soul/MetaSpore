## [中文介绍](README-CN.md)
In the recommendation system, generally speaking, the resources of serving are always limited, and the user's patience with the system response time is also limited. The recall algorithms are used to quickly filter out those items that do not match the user's needs from the item collection in the order of `million` ~ `billion`, and feed the remaining related items to the ranking algorithms. In recent years, with the development of deep networks, the use of neural networks for recall has also emerged, especially the two-towers model, which has simple structure and good effect, and has become the standard algorithm in the recall phase. Meanwhile, some classic [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) algorithms have been around for a long time, but some [new research](https://arxiv.org/abs/1907.06902) shows that these methods are still very strong baselines and have practical application value.

In this project, we will implement and benchmark the algorithms of two-tower model, such as [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf), [SimpleX](https://arxiv.org/abs/2109.12613) and other classic collaborative filtering algorithms, as well as [Swing](https://arxiv.org/abs/2010.05525), [Item CF](https://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf) for comparison on the MovieLens datasets. It should be noted that some algorithms have not yet been implemented, and we have not sufficiently tuned the parameters of the model. We will continue to enrich our algorithm package and provide experimental results.

## Model List

| Model |  Training Script | Implementation | Paper |
|:---------:|:---------:|:---------:|:-------------------------------------|
| Global Hot | [global_hot.py](baseline/global_hot.py) | - | -                                     |
| Item CF I2I  |    [item_cf.py](baseline/item_cf.py)    |   [item_cf_retrieval.py](../../python/algos/item_cf_retrieval.py)   | [WWW 2010] [Item-Based Collaborative Filtering Recommendation Algorithms](https://www.ra.ethz.ch/cdstore/www10/papers/pdf/p519.pdf)  |
| Swing I2I  |   [swing.py](baseline/swing.py)    |  [swing_retrieval.py](../../python/metaspore/swing_retrieval.py)   | [arxiv 2020] [Large Scale Product Graph Construction for Recommendation in E-commerce](https://arxiv.org/abs/2109.12613)  | 
| ALS MF  |   [spark_als.py](baseline/spark_als.py)    |   [Spark Mllib ALS](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.recommendation.ALS.html)   | [ICDM 2008] [Collaborative Filtering for Implicit Feedback Datasets](http://www.yifanhu.net/PUB/cf.pdf)  | 
| DSSM  |   [dssm.py](dssm/dssm.py)    |   [dssm_net.py](../../python/algos/dssm_net.py)   | [CIKM 2013] [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)  |
| SimpleX  |   -    |   [simplex_net.py](../../python/algos/simplex/simplex_net.py)   | [CIKM 2021] [SimpleX: A Simple and Strong Baseline for Collaborative Filtering](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)  |

We are continuously adding new models.

## Benchmark Result

| Model | Dataset | Precision@20 | Recall@20 | MAP@20 | NDCG@20 | 
|:--------------:|:--------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|
| Global Hot | MovieLens-1M | 0.002477| 0.049533 | 0.008923 | 0.017346|
| Spark ALS | MovieLens-1M | 0.002472 | 0.049444 | 0.015736 | 0.017743 |
| Swing I2I | MovieLens-1M | 0.006334 | 0.126674 | 0.029579 | 0.050461 |
| Item CF I2I | MovieLens-1M | 0.009383 | 0.187667 | 0.050912 | 0.080504 |
| DSSM BCE | MovieLens-1M NegSample-10 | 0.010776 | 0.215533 | 0.043305 | 0.080013 |
| DSSM BCE | MovieLens-1M NegSample-100 | 0.011313 | 0.226264 | 0.047736 | 0.085856 |
| SimpleX CCL | MovieLens-1M NegSample-100 | - | - | - | - |

For the two-tower model based on neural network, we give the respective experimental results based on different loss functions and different negative sampling ratio. As our models and experiments are tuned, this experiment results may be updated in the future.

## How to Run
### 1. Data processing and feature generation
For the MovieLens dataset, we currently only use `user_id`, `movie_id`, `recent_movie_id`, `last_movie` as model features. The feature generation process can refer to our data processing and preparation [description](../dataset/README.md).

### 2. Upload the schema file of features
For the neural network model, in MetaSpore, it is necessary to describe the feature column and feature combination column in [schema](dssm/schema) file, and upload these files to S3 storage. For example, taking the `DSSM` model as an example, suppose we are in the root directory of this project,
we need to execute the following commands:

```shell
aws s3 cp --recursive dssm/schema/ml_1m/.* ${MY_S3_BUCKET}/movielens/1m/schema/dssm/
```

### 3. Initialize the configuration file 
Before running models, we should initialize the config files from their `YAML` template for substituting some variables. 

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < dssm/conf/dssm_bce_neg10_ml_1m.yaml > dssm/conf/dssm_bce_neg10_ml_1m.yaml.dev
```

### 4. Running Models
Now, We can run the training scripts. For example, to train and test a `DSSM` model with the MovieLens-1M dataset, we can execute a command like this:

```shell
cd dssm
python dssm.py --conf conf/dssm_bce_neg10_ml_1m.yaml.dev > log/dssm_bce_neg10_ml_1m.log 2>&1 &
```

