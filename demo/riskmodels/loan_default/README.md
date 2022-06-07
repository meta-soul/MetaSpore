## [中文介绍](README-CN.md)

# Loan Overdue Rate Estimation Demo
We use the dataset of [Tianchi Competetion](https://tianchi.aliyun.com/competition/entrance/531830/information) to train our loan overdue rate estimation. In this demo, we use [Spark LightGBM](https://microsoft.github.io/SynapseML/) to train a binary classifier based on the preprocessed dataset in [fg.ipynb](../../dataset/tianchi_loan/fg.ipynb). Hyper parameters of this LightGBM model is optimized using `HyperOpt`, which is refered to [overdue_estimation_spark_lgbm.ipynb](./notebooks/overdue_estimation_spark_lgbm.ipynb).


## Benchmarks

|    Dataset    | Train AUC | Test AUC |
|:-------------:|:----------:|:--------:|
| Tianchi-Loan |  `0.7580`  | `0.7336` |

## How to run
### Initialize the configuration files for models
First we should initialize the config files from their `YAML` template for substituting some variables. Suppose we are in root directory of this project, for example:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < conf/overdue_estimation_spark_lgbm.yaml > conf/overdue_estimation_spark_lgbm_dev.yaml
```

### Train model
Suppose we are in root directory of this project, we could run the training script now:
```shell
python overdue_estimation_spark_lgbm.py --conf conf/overdue_estimation_spark_lgbm_dev.yaml
```
