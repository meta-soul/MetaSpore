## 预测目标

一般来说大型的电商网站都有自己的类目体系，电商的类目规划在运营工作中起着举足轻重的作用，
我们这个Demo的目标是使用预训练的BERT模型去完成电商网站的物品分类任务，通过输入物品的title去判断该物品是否属于Fashion（服饰）行业。

## 模型结构

模型方面我们选取的是 [HuggingFace Bert Base Cased 预训练 checkpoint](https://huggingface.co/bert-base-cased) 。
模型有 12 层的 Transformer Encoder，隐藏层维度是 768，共 1.1 亿的参数量。
在 Bert 层之上，是 Dropout 层，Linear 层和 Sigmoid 层，整体结构如下：

<div align=center>
<img src="../../docs/images/title_to_fishion.PNG" alt="title_to_fish architecture" width="500">
</div>

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
在gen_data.py中，注意到原始数据为json格式，我们用load_file函数加载原始数据，使用gen_dataset函数处理原始数据，使用save_dataset函数将处理好的数据存到base_path文件夹中，用作之后模型的训练。

## 模型训练

在得到模型训练需要的数据后，就可以运行训练模型的脚本[train.sh](text_classifier/train.sh)来训练模型了。
```
cd text_classifier
bash train.sh
```
以下为train.sh的内容：
```
python -u train.py --name title_to_fashion \
    --model bert-base-cased --num-labels 1 \
    --train-file /your/working/path/title_to_fashion_500k/train.tsv \
    --eval-file /your/working/path/title_to_fashion_500k/val.tsv \
    --eval-steps 1000 \
    --num-epochs 1 --lr 2e-5 --train-batch-size 32 --eval-batch-size 32 --gpu 0 \
    --output-path ./output
```
在该脚本中，你可以自定义参数配置，下面简单说明每个参数作用：
+ name: 指定模型的名字
+ model: 指定使用的预训练模型
+ num-labels: 分类的类别个数
+ train-file: 数据准备阶段生成的训练数据
+ eval-file: 数据准备阶段生成的验证数据
+ eval-steps: 模型训练过程中进行推理的步数间隔
+ num-epochs: 模型训练的轮数
+ lr: 模型训练的学习率
+ train-batch-size: 训练阶段的批量大小
+ eval-batch-size: 测试阶段的批量大小
+ gpu: 使用指定gpu进行训练
+ output-path: 模型保存的路径

## 模型效果

在我们的实验中，模型训练一个 epoch 已经收敛的比较好了，下面是Threshold = 0.5时Accuracy、AUC、Precision、Recall、F1等几个指标的值：

| Accuracy | AUC | Precision | Recall   |    F1    |
|:--------:|:---:|:---------:|:--------:|:--------:|
|0.9757|0.9943|0.9355|0.9389|0.9372|
