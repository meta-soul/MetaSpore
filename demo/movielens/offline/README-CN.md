# MovieLens Demo-离线模型训练

一般来说，一个典型的推荐系统正如下午所描述的那样，主要由数据预处理、召回模型开发和优化、排序模型开发和优化等等。举例来说，在召回阶段，一般会使用协同过滤，基于图的方法，甚至是基于神经网络的方法来做用户和商品的匹配；而在排序和重排阶段，一些业务指标，比如CTR、CVR、Novelty等，会被机器学习模型来直接建模。对于一个算法算法工程师来说，日常优化工作的重点基本都集中在离线数据和模型上面。在这里，我们会使用 `MetaSpore` 来介绍基本的数据预处理、召回、排序等模块，来展示利用我们的系统如何快速构建一个工业级推荐系统。

<p align="center">
<img width="800" alt="image" src="https://user-images.githubusercontent.com/7464971/160760862-48b81b21-b729-4b34-b4fe-c83985474664.png">
</p>

在这个Demo的项目里，我们使用 [MoiveLens-1M](https://grouplens.org/datasets/movielens/1m/) 这个数据集来演示，这份数据集在推荐领域非常著名，可以从链接给出的官网中下载，并存储在您的云端 S3 存储上。


## 1. 初始化模型配置文件
在我们深入到具体的数据和模型开发工作之前，我们需要通过给出的 `YAML` 配置模版对不同阶段的配置文件进行初始化，主要是替换模版中一些需要定制的变量。举例来说，我们需要替换自己具体的 S3 路径 `MY_S3_BUCKET`:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < template.yaml > output.yaml 
```

在后面我们运行 python 脚本之前，我们假设大家已经完成了初始化模型配置文件的工作。


## 2. 数据预处理和特征生成
在这个阶段，我们主要做 4 件事情，分别是：训练集和测试集的划分，组织训练样本，生成模型训练需要的离散特征和连续特征，初始化用户特征、电影特征以便于存储与 `MongoDB` 中。

```shell
python fg_movielens.py --conf fg.yaml 
```
 
当我们执行完以上脚本之后，我们需要使用 `Spark` 用户和电影特征灌到 `MongoDB` 中：
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

当然，我们可以根据自己场景的需要，来更改这些输入、输出路径。
 
 
## 3. 召回算法 
在这个阶段，我们主要介绍 3 中召回算法，包括 `Item CF`, `Swing`, `Two-Twoer`，这些算法的执行过程会在接下来进行详细的说明。


### 3.1 Item CF
首先，我们运行 `Item CF` 的训练脚本：
```python
python item_cf.py --conf item_cf.yaml 
``` 

这条命令执行结束之后，我们算法中计算的相似度 I2I 矩阵会被保存到 `item_cf_out_path` 这个路径下，其中 `item_cf_out_path` 是 `item_cf.yaml` 中到配置项。现在我们可以像在之前的步骤那样，将我们计算的结果灌到 `MongoDB` 中。

```shell
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin itemcf --dest itemcf --queryid key 
```

### 3.2 Swing
首先，我们可以运行Swing的训练脚本：

```python
python swing.py --conf swing.yaml 
``` 

在这条命令执行结束之后，我们算法中计算的相似度 I2I 矩阵会被保存到 `item_cf_out_path` 这个路径下，其中 `item_cf_out_path` 是 `item_cf.yaml` 中到配置项。现在我们可以像在之前的步骤那样，将我们计算的结果灌到 `MongoDB` 中。

```shell
spark-submit \
    --master local \
    --name write_mongo \
    --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    dump/write_mongo.py --origin swing --dest swing --queryid key
```

### 3.3 Two-Tower  
`SimpleX` 算法是一种简单并鲁棒的双塔模型的实现，我们可以运行 `SimpleX` 训练脚本：

```python
python simplex.py --conf simplex.yaml 
``` 

在执行完这条命令之后，我们电影的 embedding 向量会被自动输出到 `Milvus` 到数据库中，其中有关 Milvus 服务的配置，比如域名、端口等详细的配置在 `simplex.yaml` 文件中。

## 4. 排序算法

在排序阶段，我们把个性化排序问题建模成 `CTR 预估` 问题，这种建模方法已经在业界被广泛采用。这里我们展示两种模型，如何在 `MetaSpore` 中使用：第一种是经典的树模型，我们以`LightGBM` 为例，第二种是是神经网络模型，我们以 `Wide & Deep` 为例。

### 4.1 树模型
如前文所述，我们这里使用 `LightGBM` 模型来解决排序问题，我们可以使用一下训练脚本：
```shell
python lgbm_model_train.py --conf lgbm.yaml
```

这里需要注意的是，当我们树模型训练完成之后，我们使用一下代码转化成 ONNX 格式，以便于 `MetaSpore Serving` 加载并进行线上预测。
```python
def convert_model(lgbm_model: LGBMClassifier or Booster, input_size: int) -> bytes:
    initial_types = [("input", FloatTensorType([-1, input_size]))]
    onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset = 9)
    return onnx_model
```

### 4.2 神经网络模型
如前文所述，我们这里使用 `Wide & Deep` 模型 来演示在 `MetaSpore` 平台进行分布式训练的能力，我们可以运行一下训练脚本：
```shell
python widedeep.py --conf widedeep.yaml
```
在上面脚本执行结束之后，ONNX 格式的模型文件已经被自动导出到 S3 的路径： `model_export_path`，而这个路径变量在 `widedeep.yaml` 文件中配置。


## 5. Tuner
最后，对于 `MetaSpore` 中开发的模型，如 `Item CF`，`Swing`，`Wide & Deep` 等，我们实现了一个轻量级的超参数搜索工具，我们这里以 [Wide & Deep tuner 配置](tuner/widedeep_tuner.yaml)来说明如何使用：

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
1. 我们在配置文件中需要对一些基本的配置项进行定义，包括 `app_name`, `num_expriment`等；
2. 我们在配置文件中需要对 `dataset` 的路径进行定义；
3. 我们在配置文件中需要对 `common_param` 来配置哪些模型参数**是不希望**在本次实验中进行搜索；
4. 我们在配置文件中需要对 `hyper_param` 来配置哪写模型参数**是希望**在在本次实验中进行搜索的。

在以上 `YAML` 文件被完善的配置之后，我们可以运行：
```shell
python widedeep_tuner.py --conf widedeep_tuner.yaml
```
当这条命令运行完成之后，超参数寻优的结果被保存在 `YAML` 文件中定义的 `result_path` 路径中。




