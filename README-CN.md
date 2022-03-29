# MetaSpore 一站式机器学习开发平台

MetaSpore 是一个一站式端到端的机器学习开发平台，提供从数据预处理、模型训练、离线实验、在线预测到在线实验分桶 ABTest 的全流程框架和开发接口。

## 核心功能
MetaSpore 具有如下几个特点：

1. 一站式端到端开发，从离线模型训练到在线预测和分桶实验，全链路统一的开发体验；
2. 深度学习训练框架，兼容 PyTorch 接口，支持分布式大规模稀疏特征学习，并与 PySpark 打通，无缝读取数据湖和数仓上的训练数据；
3. 高性能在线预测服务，支持神经网络、决策树、Spark ML、SKLearn 等多种模型；支持 GPU、NPU 加速；
4. 在离线统一特征抽取框架，自动生成线上特征读取逻辑，统一特征抽取逻辑；
5. 在线算法应用框架，提供模型预测、实验分桶切流、参数动态热加载和丰富的 Debug 功能；
6. 丰富的行业算法示例和端到端完整链路解决方案。

## 文档和示例

* [离线训练入门教程](tutorials/metaspore-getting-started.ipynb)

* [一个 MovieLens 端到端推荐系统](demo/movielens/online)

## 安装包下载
我们提供了一个预编译的离线训练安装包：[下载链接]()。 该安装包需要 Python 3.8.

下载后在 Python 3.8 环境下，通过命令行执行安装：
```bash
pip install pyspark
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip install metaspore-1.0.0-cp38-cp38-linux_x86_64.whl
```

## 编译代码

* [离线训练框架编译](docs/build-offline.md)

## 问题反馈

关于使用上的问题，可以在 [GitHub Discussion](https://github.com/meta-soul/MetaSpore/discussions) 中发帖提问，也可以通过 [GitHub Issue](https://github.com/meta-soul/MetaSpore/issues?q=) 反馈。

### 微信公众号
欢迎关注元灵数智公众号，我们会定期推送关于 MetaSpore 的架构代码解读、端到端算法业务落地案例分享等文章：

![元灵数智公众号](docs/images/%E5%85%83%E7%81%B5%E6%95%B0%E6%99%BA%E5%85%AC%E4%BC%97%E5%8F%B7.jpg)

### MetaSpore 开发者社区微信群
欢迎扫码加入 MetaSpore 开发者社区微信群，随时交流 MetaSpore 开发相关的各类问题：



## 开源项目
MetaSpore 是一个完全开源的项目，以 Apache License 2.0 协议发布。欢迎参与使用、反馈和贡献代码。