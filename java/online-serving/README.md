# Metaspore Online Serving

Metaspore Online Serving 是由 [DMetaSoul](https://www.dmetasoul.com/) 研发的一套支撑在线算法业务服务、低代码模式的在线应用框架。包含 serving 、experiment-pipeline、feature-extract 三个子模块。

- serving：模型计算服务，提供Sparse DNN模型、双塔模型、Spark ML模型、SKLearn模型、XGBoost模型预测服务以及 Milvus 向量召回服务。

- experiment-pipiline：实验工作流 SDK，实时动态更新、低代码模式的算法实验流框架。支持动态更新实验、多层正交、实验切流等线上常用功能。

- feature-extract：特征自动生成框架，根据 Yaml 文件自动生成特征的 JPA 代码的框架。支持多种 DB 源，便捷提供特征接口。

  

## Demo 示例

完整示例：[Movielens Demo 示例链接](https://github.com/meta-soul/MetaSpore/blob/main/demo/movielens/online/README.md)



## 说明文档

| 说明文档            |                                           |
| :------------------ | :---------------------------------------: |
| serving             |                                           |
| experiment-pipeline | [中文说明](experiment-pipeline/README.md) |
| feature-extract     |   [中文说明](feature-extract/README.md)   |
|                     |                                           |

