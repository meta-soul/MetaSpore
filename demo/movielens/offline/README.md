# Offline Models for MovieLens Recommender

As we known, for one typical personalized recommender system, as depicted in the figure below, the offline work is mainly composed by data preprocessing, recall model developing, ranking model developing, etc. For example, in the recall stage, collaborative filtering or graph theory-based methods, or even neural network-based methods, may be used to match between users and candidate items. In the ranking and reranking stage, the final business indicators are generally modeled and ranking model directly. Lots of optimization work is focused on offline model iterations. Here we introduce how to develop the basic data preprocessing script, recall model and ranking model on our `MetaSpore` platform. If you are Chinese developer, you may like to visit our [CN Doc](README-CN.md).

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/7464971/160760862-48b81b21-b729-4b34-b4fe-c83985474664.png">
</p>

In this demo project, we use [MoiveLens-1M](https://grouplens.org/datasets/movielens/1m/) to demonstrate our system. You can download this dataset from this their website and store these files onto your own cloud strorage.

## 1. Initialize the Configuration Files for Models
Before we continue to dive into the offline models, we should firstly initialize the config files from their `YAML` template for substituting some variables. For example
```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < template.yaml > output.yaml 
```
For the latter stages, we assume that we have done this to generate the `training configurations` before running the python scripts.

## 2. Data Preprocessing and Feature Generation
At this stage, we mainly do four things, including: splitting the training set and test set, organizing the training samples, generating discrete features and continuous features, save some user and item features into `MongoDB`. 

```shell
python fg_movielens.py --conf fg.yaml 
```
 
Moreover, we should dump the user and item features into `MongoDB` using `Spark`:
```shell
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin items --dest item --queryid movie_id

spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin users --dest user --queryid user_id
    
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin item_feature --dest item_feature --queryid movie_id
```

You can change the input and output paths to adapt to your scenarios.
 
 
## 3. Retrieval algorithm 

In this stage, we mainly introduce the use of three offline recall algorithms, including: `Item CF`, `Swing`, `Two-Twoer`. These algorithms usage will be introduced in detail as below.


### 3.1 Item CF
Firstly, We can run the `Item CF` trainer script:
```python
python item_cf.py --conf item_cf.yaml 
``` 

After that, the similarity matrix of I2I is saved into `item_cf_out_path`, which is configured in this `item_cf.yaml` file. Now, We can dump the result of this model into `MongoDB` as we did in our previous stage: 
```shell
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin itemcf --dest itemcf --queryid key 
```

### 3.2 Swing
Firstly, We can run the `Swing` trainer script:

```python
python swing.py --conf swing.yaml 
``` 

After that, the similarity matrix of I2I is saved into `swing_out_path`, which is configured in this `swing.yaml` file. Now, We can dump the result of this model into `MongoDB` as we did in our previous stage: 
```shell
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin swing --dest swing --queryid key
```

### 3.3 Two-Tower  
`SimpleX` algorithm is a simple but robust implementation of the two-tower model. Firstly, We can run the `SimpleX` trainer script:
```python
python simplex.py --conf simplex.yaml 
``` 
After the training script is executed, the embedding vector of the movies has been stored into database of `Milvus`, which is configured in this `simplex.yaml` file.

## 4. Ranking model

In the ranking stage, we model the personalized ranking as a `CTR prediction` problem, which is widely used in recommendation scenarios. Here we introduce two methods, one is to use the classical tree ensemble model that developed by using `LightGBM`, the other is to use the neural network model that developed by `MetaSpore`.

### 4.1 Tree Ensemble Model
As we described previously, we can use `LightGBM` to solve the ranking problem. Now we can run the trainer script:
```shell
python lgbm_model_train.py --conf lgbm.yaml
```
A special note is needed here. After training the model, we using code below to transform the model into ONNX format for `MetaSpore Serving`:
```python
def convert_model(lgbm_model: LGBMClassifier or Booster, input_size: int) -> bytes:
    initial_types = [("input", FloatTensorType([-1, input_size]))]
    onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset = 9)
    return onnx_model
```

### 4.2 Neural Network Model
In this section, we use `Wide & Deep` model to demonstrate the ability to train the neural network in `MetaSpore` in distributed clusters. We can run the trainer script as below:
```shell
python widedeep.py --conf widedeep.yaml
```
After the execution of this script, the ONNX format of this nn model is automatically exported into s3 path `model_export_path`, which is configured in this `widedeep.yaml` file.

## 5. Tuner
Moreover, for the model, such as `Item CF`, `Swing`, `Wide & Deep`, which developed in `MetaSpore`, we deliver an tuner lib to search a combination of hyperparamters of the model. Here, we can use the [Wide & Deep tuner configuration](https://github.com/meta-soul/MetaSpore/blob/sunkai/20220328_movielens_demo_offline/demo/movielens/offline/tuner/widedeep_tuner.yaml) as a demo to illustrate:
```YAML
app_name: Wide&Deep CTR Model Tuner
num_experiment: 3
model_name: widedeep
result_path: ${MY_S3_DIR}/tuner/model/movielens/widedeep/

dataset:
    train: ${MY_S3_DIR}/movielens/rank/train.parquet
    test: ${MY_S3_DIR}/movielens/rank/test.parquet

common_param:
    local: False
    column_name_path: ${MY_S3_DIR}/movielens/schema/widedeep/column_schema
    combine_schema_path: ${MY_S3_DIR}/movielens/schema/widedeep/combine_column_schema
    ...

hyper_param:
    use_wide: [True, False]
    embedding_size: [10, 20]
    deep_hidden_units: [[1024, 512, 256, 128, 1], [1024, 512, 1]]
    adam_learning_rate: [0.00001, 0.0001, 0.001]
    ...
```

1. We should define the basic configuration in this experiment, such as `app_name`, `num_expriment` etc.
2. We should define the `dataset` configuration where training data is saved.
3. We should define the `common_param` that we would not like to search in this experiment.
4. We should define the `hyper_param` that we indeed would like search over these parameter spaces in this experiment.

After this `YAML` file is configure `properly`, we can run:
```shell
python widedeep_tuner.py --conf widedeep_tuner.yaml
```
When this script is executed, the searching result will be stored in `result_path` that defined in the `YAML` file.
