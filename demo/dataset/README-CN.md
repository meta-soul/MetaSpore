# 数据预处理与准备
在这个项目中，我们会统一所有Demo项目的数据处理过程，包括对MovieLens-1M，MovieLens-25M，Criteo-5D等，以及其他等数据集。处理过程包括：特征生成、召回样本生成、排序样本生成、负采样等具体等工作。

以下是数据集的概述：

| 数据集                             | 如何在MetaSpore中使用                                    | 引用链接                                                                                                                                   |
|:--------------------------------|:---------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| [MovieLens-1M](#MovieLens-1M)   | [Movie Recommendation End2End Demo](../movielens/) | [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)                                                                   |
| [MovieLens-25M](#MovieLens-25M) | [CTR Demo](../ctr/)                                | [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/1m/)                                                                  |
| [Criteo-5D](#Criteo-5D)         | [CTR Demo](../ctr/)                                | [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)                                                 |
| [Census](#Census)               | [MMoE Demo](../multitask/mmoe/)                    | [Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid](http://robotics.stanford.edu/~ronnyk/nbtree.pdf)          |
| [Ali-CCP](#Ali-CCP)             | [ESMM Demo](../multitask/esmm/)                    | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/pdf/1804.07931.pdf) |
| [Tianchi Loan](#Tianchi Loan)             | [Loan Overdue Demo](../riskmodels/loan_overdue/)                    | [Tianchi Competetion](https://tianchi.aliyun.com/competition/entrance/531830/information) |

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

## Census
在这一节, 我们使用人口统计公开集 [Census](https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census.tar.gz) 作为我们Demo的数据集. 
### 数据预处理
```shell
cd census
sh data_process.sh
```
我们提取出 'marital_stat' 以及 'income_50k' 作为多任务模型的两个标签。并且对连续型特征做如下变换:
```python
import numpy as np
def fun3(x):
    return np.log(x+1).astype(int)
```
此外, 我们无需对离散型特征取one-hot编码，因为MetaSpore会自动处理模型的Embedding层。

## Ali-CCP
在这一节里，我们将介绍如何处理 [Ali-CCP](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408) 这个数据集。原始的数据集比较大，我们这里只使用了 [PaddleRec](https://github.com/PaddlePaddle/PaddleRec) 项目中使用的两个子集：

* **[小版本子数据集](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/ali-ccp)**：包含了大概10万的训练样本和测试样本。
* **[大版本子数据集](https://github.com/PaddlePaddle/PaddleRec/tree/master/datasets/ali-cpp_aitm)**：包含了大概3800万训练忘本和4300万测试样本。

### 下载数据
假设我们位于dataset项目的根目录，我们可以通过执行以下命令下载这两个版本的数据并上传到S3云存储上：

```shell
cd aliccp
export MY_S3_BUCKET='your S3 bucket directory'
envsubst < data_processing.sh > data_processing_dev.sh
data_processing_dev.sh
```

### Feature Generation
在数据下载完成之后，我们可以通过以下的 Python 脚本来生成MetaSpore可以使用特征和 label 的列：

```python
# small dataset
python fg_small_dataset.py --conf fg_small_dataset.yaml.dev
# large dataset
python fg_large_dataset.py --conf fg_large_dataset.yaml.dev
```

## Tianchi Loan
在这一节中，我们将介绍如何处理天池社区提供的 [贷款违约率比赛](https://tianchi.aliyun.com/competition/entrance/531830/information) 的数据。

### Download Data
首先我们应该从 [贷款违约率比赛](https://tianchi.aliyun.com/competition/entrance/531830/information) 的网址手动下载这个数据集。

### Feature Generation
下载完成后，我们可以使用 [fg.ipynb](./tianchi_loan/fg.ipynb) 来生成我们模型中使用的数值特征。

