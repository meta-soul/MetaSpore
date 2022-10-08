# 文本分类
我们这个Demo的目标是使用预训练的BERT模型去完成电商网站的物品分类任务。
通过输入物品的title去判断该物品是否属于Fashion行业。

## 1. 准备数据
我们使用 [Amazon Review Data(2018)](https://nijianmo.github.io/amazon/) 数据集，使用 [gen_data.py](dataset/gen_data.py)脚本采样500k条数据去训练我们的模型。
```
cd dataset
python -u gen_data.py
```

## 2. 训练
训练阶段的脚本在[train.sh](text_classifier/train.sh)。你可以根据你的环境进行参数配置。
```
cd text_classifier
bash train.sh
```

## 3. 预测
预测阶段的脚本也在[train.sh](text_classifier/train.sh)中。
```
cd text_classifier
bash train.sh
```

## 4. 线上取得推荐结果
您可以运行在线服务入口点（MovielensRecommendApplication.java）并立即测试它。
例如：`curl http://localhost:8080/user/10`为userId等于10的用户获取推荐的电影。
