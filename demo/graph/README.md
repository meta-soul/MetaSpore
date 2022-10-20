## [中文介绍](README-CN.md)

# Graph model

In this project, we will implement recall models based on jaccard distance, euclidean distance and node to vector, hereinafter referred to as Jaccard, Euclidean and node2vec. We compare them on the MovieLens and/or Pokec dataset. We will continue to enrich our algorithm package and compare experimental results.

## Model List

|   Model   |            Training Script             |                         Algorithm Implementation                          | Paper                                                                                    |
|:---------:|:--------------------------------------:|:-------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------|
|  Jaccard  |    [jaccard.py](jaccard/jaccard.py)    |   [jaccard_retrieval.py](../../python/algos/graph/jaccard_retrieval.py)   |                                                                                          |
| Euclidean | [euclidean.py](euclidean/euclidean.py) | [euclidean_retrieval.py](../../python/algos/graph/euclidean_retrieval.py) |                                                                                          |
| node2vec  |  [node2vec.py](node2vec/node2vec.py)   |  [node2vec_retrieval.py](../../python/algos/graph/node2vec_retrieval.py)  | [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) |

We will constantly add new models.

## How to run

### 1. Data preparation
For MovieLens dataset, we only use `user_id`，`movie_id` is the feature of the model.

### 2. Initialize training configuration file

Initialize the model configuration file we need by replacing the variables in the corresponding 'YAML' template. For example:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < jaccard/conf/jaccard.yaml > jaccard/conf/jaccard.yaml.dev
```
### 3. Run the model test script

We can now run the training script. For example, to train and test the Jaccard model with MovieLens-1M dataset, you need to execute the following command:
```shell
cd jaccard
python jaccard.py --conf conf/jaccard.yaml.dev > log/jaccard_ml_1m.log 2>&1 &
```