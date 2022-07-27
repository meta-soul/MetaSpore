## [中文介绍](README-CN.md)

# Credit Card Fraud Detection Demo
As we known, it is very important to recognize fraudulent credit card transactions accurately and quickly for credit card companies so that customers are not charged for items that they did not purchase. In this demo, we will use the dataset of [ULB Credit Card Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to train our fraud detection model based on [Spark LighGBM](https://microsoft.github.io/SynapseML/docs/next/features/lightgbm/LightGBM%20-%20Overview/). Moreover, in [notebooks directory](./notebooks/), [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) and [Isolation Forest](https://mmlspark.blob.core.windows.net/docs/0.9.1/pyspark/synapse.ml.isolationforest.html) will be shown how to applied in this scenario.


## Benchmarks

|    Dataset    | Model      |  Train AUC | Test AUC |
|:-------------:|:----------:|:----------:|:--------:|
| ULB Credit Card |  LightGBM  | `0.9985`  | `0.9715` |
| ULB Credit Card |  Isolation Forest  | `0.9490`  | `0.9772` |

## How to run
### Initialize the configuration files for models
First we should initialize the config files from their `YAML` template for substituting some variables. Suppose we are in root directory of this project, for example:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < conf/spark_lgbm.yaml > conf/spark_lgbm_dev.yaml
```

### Train model
Suppose we are in root directory of this project, we could run the training script now:
```shell
python spark_lgbm.py --conf conf/spark_lgbm_dev.yaml
```
