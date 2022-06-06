# 贷款违约率预估 Demo
我们使用天池社区提供的 [贷款违约率比赛](https://tianchi.aliyun.com/competition/entrance/531830/information) 的数据来训练我们的违约率预估模型。在这个Demo中，我们使用 [Spark LightGBM](https://microsoft.github.io/SynapseML/) 训练一个二分类模型，训练中用到的训练样本的预处理在 [fg.ipynb](../../dataset/tianchi_loan/fg.ipynb) 中。 LightGBM模型的超参数使用 `HyperOpt` 进行搜索，可以参考  [overdue_estimation_spark_lgbm.ipynb](./notebooks/overdue_estimation_spark_lgbm.ipynb) 中的代码。

## 测试结果

|    Dataset    | Train AUC | Test AUC |
|:-------------:|:----------:|:--------:|
| Tianchi |  `0.7580`  | `0.7336` |

## 如何运行
### 初始化模型所需的配置文件
通过替换对应YAML模板中的变量，初始化我们需要的模型配置文件。假设我们在项目的根目录，举例来说:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < conf/overdue_estimation_spark_lgbm.yaml > conf/overdue_estimation_spark_lgbm.yaml.dev
```

### 训练模型
假设我们在项目的根目录，我们现在可以运行以下的训练脚本：
```shell
python overdue_estimation_spark_lgbm.py --conf conf/overdue_estimation_spark_lgbm.yaml.dev
```