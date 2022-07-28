# 欺诈检测 Demo
我们知道，对于信用卡公司来说，准确、快速地识别交易中的欺诈行为非常重要，这样客户就不会为他们没有购买的商品付费。我们使用 ULB 提供的 [信用卡欺诈检测数据集](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 来训练我们的欺诈检测模型。在这个Demo中，我们使用 [Spark LighGBM](https://microsoft.github.io/SynapseML/docs/next/features/lightgbm/LightGBM%20-%20Overview/) 训练一个二分类模型来进行预测。同时，在[ notebooks 目录](./notebooks/)，我们会展示在这个业务场景中，如何应用 [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) 和 [Isolation Forest](https://mmlspark.blob.core.windows.net/docs/0.9.1/pyspark/synapse.ml.isolationforest.html) 算法。

## 测试结果

|    Dataset    | Model      |  Train AUC | Test AUC |
|:-------------:|:----------:|:----------:|:--------:|
| ULB Credit Card |  LightGBM  | `0.9982`  | `0.9813` |
| ULB Credit Card |  Isolation Forest  | `0.9490`  | `0.9772` |

## 如何运行
### 初始化模型所需的配置文件
通过替换对应YAML模板中的变量，初始化我们需要的模型配置文件。假设我们在项目的根目录，举例来说:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < conf/spark_lgbm.yaml > conf/spark_lgbm_dev.yaml
```

### 训练模型
假设我们在项目的根目录，我们现在可以运行以下的训练脚本：
```shell
python spark_lgbm.py --conf conf/spark_lgbm_dev.yaml
```