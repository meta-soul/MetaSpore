# 数据预处理与准备
在这个项目中，我们会统一所有Demo项目的数据处理过程，包括对MovieLens-1M，MovieLens-25M，Criteo-5D等，以及其他等数据集。处理过程包括：特征生成、召回样本生成、排序样本生成、负采样等具体等工作。

## 初始化模型配置文件
首先，我们需要初始化配置文件，我们需要通过给出的 YAML 配置模版对不同阶段的配置文件进行初始化，主要是替换模版中一些需要定制的变量。举例来说，我们需要替换自己具体的 S3 路径 `MY_S3_BUCKET`:

```shell
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < fg.yaml > fg.yaml.dev 
```

在后面我们运行 python 脚本之前，我们假设大家已经完成了初始化模型配置文件的工作。如果还没有安装 `envsubst` 命令，可以进行安装，比如在基于 `Debian` 的 Linux 系统上，可以使用 `sudo apt-get install gettext-base` 命令来完成安装。

## MovieLens-1M
在这一节，我们使用 [MoiveLens-1M](https://grouplens.org/datasets/movielens/1m/) 来进行演示。您可以从上面提供的网址中下载项目所需要的数据，并存储在您的云端 S3 存储上。

### 1. 特征生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令来完成特征生成的工作：

 ```shell
 cd ml_1m
 python fg.py --conf fg.yaml.dev --verbose
 ```

### 2. 召回样本生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令准备好 `Collaborative Filtering` 算法训练所使用的样本数据：

```shell
cd ml_1m
python match_dataset_cf.py --conf match_dataset.yaml.dev --verbose
```

之后，我们通过执行以下命令来获取 `TwoTowers` 算法训练所使用的样本数据：

```shell
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose
```

### 3. 排序样本生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令准备好 [CTR 模型](../ctr/README-CN.md) 训练所使用的样本数据：

```shell
python rank_dataset.py --conf rank.yaml.dev --verbose
```

## MovieLens-25M
在这一节，我们使用 [MoiveLens-25M](https://grouplens.org/datasets/movielens/25m/) 来进行演示。您可以从上面提供的网址中下载项目所需要的数据，并存储在您的云端 S3 存储上。

### 1. 特征生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令来完成特征生成的工作：

 ```shell
 cd ml_25m
 python fg.py --conf fg.yaml.dev --verbose
 ```

### 2. 召回样本生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令准备好 `TwoTowers` 算法训练所使用的样本数据：

```shell
python match_dataset_negsample.py --conf match_dataset_negsample_10.yaml.dev --verbose
```

### 3. 排序样本生成
假设我们位于dataset项目的根目录，我们可以通过执行以下命令准备好 [CTR 模型](../ctr/README-CN.md) 训练所使用的样本数据：

```shell
python rank_dataset.py --conf rank.yaml.dev --verbose
```

## Criteo-5D
在这一节，我们使用由 CriteoLabs 公开的广告曝光点击日志 [Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs-2/) 来进行演示。

```python
import metaspore
metaspore.demo.download_dataset()
```

数据集比较大，如果您下载失败，可以重试或者通过 [MetaSpore Demo Dataset](https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/demo/criteo/index.html) 来进行手动下载。 根据2014年 [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) 优胜团队提供的方法 [3 Idiots' Approach](https://github.com/ycjuan/kaggle-2014-criteo) ，我们对里面的数值特征进行离散处理：

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

这里选取了原始训练集前 5 天的日志作为训练数据，测试集第 1 天的日志作为测试数据，进行验证。假设我们位于dataset项目的根目录，我们可以通过执行以下命令准备好 [CTR 模型](../ctr/README-CN.md) 训练所使用的样本数据：

```shell
cd criteo
python fg.py --conf fg_5d.yaml.dev --verbose
```

