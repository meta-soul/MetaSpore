# 多模态检索 Demo

深度学习在计算机视觉、自然语言处理等领域不断取得创新性突破，不同模态数据之间联合建模的多模态理解技术也越发成熟。前沿技术进展使得*打通多模态数据之间的语义鸿沟*成为可能，但繁复的离线模型优化、数据处理、高昂的线上推理和实验成本等因素，阻碍了多模态技术的落地和普惠。本项目 Demo 将围绕**以文搜文**、**以文搜图**等多模态检索场景，来向大家演示，如何基于 MetaSpore 技术体系低成本接入 [HuggingFace](https://huggingface.co/) 社区多模态预训练模型。

本项目 Demo 将提供从离线数据处理到在线检索服务一整套解决方案

- **在线服务**，一整套多模态检索线上体系，支撑以文搜文、以文搜图等多场景语义检索，涵盖前端检索 UI、数据预处理、检索召回/排序算法等服务
- **离线处理**，涵盖 demo 各个语义检索场景的离线部分，主要有离线模型训练和导出、检索数据建库以及推送线上服务组件等

## 1. 在线服务

多模态 Demo 线上服务部分由以下几部分构成：

1. [multimodal_web](online/multimodal_web)，多模态示例的前端服务，提供 web UI 界面供用户体验多模态检索能力
2. [multimodal_serving](online/multimodal_serving)，多模态示例的检索算法服务，含有实验配置、预处理、召回、排序等整个算法处理链路
3. [multimodal_preprocess](online/multimodal_preprocess)，对多模态大模型预处理逻辑（含文本/图像等）的封装，以 gRPC 接口提供服务

**注：**以上几个服务从前往后依次形成依赖关系，所以要把多模态 Demo 搭建起来，就需要**从后往前**依次把各个服务先跑起来，详见线上服务具体[操作指南](online/README-CN.md)。当然做这些之前，要记得先把[离线](./offline)的模型导出、上线和建库先搞定哈！

## 2. 离线处理

离线处理部分，主要包含了各个应用场景 demo 的离线模型导出、数据建库推送等内容。由于以文搜文和以文搜图场景离线部分较为类似，这里重点介绍基于**百科问答**数据的[以文搜文离线部分](offline/QA/README-CN.md)。

### 2.1 数据建库

我们基于开放的百万级百科问答[数据](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa)来构建检索库，整个建库过程具体说明如下：

首先，需要切换到问答数据的建库代码目录：

```bash
cd offline/QA/index_and_export/
```

**预处理**，把原始数据转换为较为通用的 jsonline 格式以供建库使用

```bash
sh preprocess.sh
```

**构建索引**，输入预处理后的数据产出向量+文档关键字段等索引数据（每行一个文档对象）

```bash
sh build.sh
```

**推送倒排和正排**，把倒排（向量）和正排（文档字段）数据分别推送到各个组件服务端

```bash
# 倒排数据推送到 Milvus 集群
sh push_milvus.sh
# 测试推送是否成功
# sh pull_milvus.sh

# 正排数据推送到 MongoDB 集群
sh push_mongo.sh
# 测试推送是否成功
# sh pull_mongo.sh
```

**注**：请在 `env.sh` 文件中给出自己 Milvus 和 MongoDB 集群配置信息。

### 2.2 模型导出

我们这里的问答检索使用了 query-doc 对称双塔模型，因为一定要保证离线 doc 建库模型和在线 query 推理模型的**一致性**，继续进入问答数据的建库代码目录，然后运行如下命令来导出模型：

```bash
sh export.sh
```

模型将被导出到 `./export` 目录，导出内容主要有线上推理使用的 ONNX 模型和预处理模型 Tokenizer。

**注**：导出模型也会被拷贝到 S3，需要在 `env.sh` 中配置自己的 S3 根路径。

### 2.3 模型优化

百科问答检索能力的基础 **Sentence Embedding** 模型，模型优化结果将决定检索效果，这里使用的语义检索模型正是我们在多个开源数据集上调优过的，这些模型已经开放到 [HuggingFace Hub](https://huggingface.co/DMetaSoul)，用户可以直接下载使用。同时我们也提供了一套模型离线优化的 [pipeline](offline/QA/training)，只需要提供简单的配置就可以训练调优用户自己的模型。