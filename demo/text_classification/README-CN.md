## 预测目标

一般来说大型的电商网站都有自己的类目体系，电商的类目规划在运营工作中起着举足轻重的作用，
我们这个Demo的目标是使用预训练的BERT模型去完成电商网站的物品分类任务，通过输入物品的title去判断该物品是否属于Fashion（服饰）行业。

## 模型结构

模型方面我们选取的是 [HuggingFace Bert Base Cased 预训练 checkpoint](https://huggingface.co/bert-base-cased) 。
模型有 12 层的 Transformer Encoder，隐藏层维度是 768，共 1.1 亿的参数量。
在 Bert 层之上，是 Dropout 层，Linear 层和 Sigmoid 层，整体结构如下：
![title_to_fishion 架构](/MetaSpore/docs/images/title_to_fishion.PNG)

## 准备数据

我们使用 [Amazon Review Data(2018)](https://nijianmo.github.io/amazon/) 数据集，
我们选取`Clothing, Shoes & Jewelry`这个行业作为我们的正样本，其他行业作为我们的负样本。
原始数据集很大，全部训练的时间较长，我们随机选取了 50 万条数据，
这些数据已经可以满足我们的训练目标，其中 train 数据集占比 90%，dev 数据集占比 5%， test 数据集占比 5%，正样本占比20%，负样本占比80%。
可运行以下脚本生成数据。
```
cd dataset
python -u gen_data.py
```

## 模型训练
训练模型的脚本在[train.sh](text_classifier/train.sh)，你可以根据你的环境进行参数配置。
```
cd text_classifier
bash train.sh
```

## 模型效果

模型训练一个 epoch 已经收敛的比较好了，下面是 Accuracy、AUC、Precision、Recall、F1等几个指标，如果我们取 Threshold = 0.5 的时候，那么


| Accuracy | AUC | Precision | Recall   |    F1    |
|:--------:|:---:|:---------:|:--------:|:--------:|
|0.9757|0.9943|0.9355|0.9389|0.9372|
