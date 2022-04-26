# 以文搜文 - QA 离线部分

QA 演示样例离线部分有：

- `index_and_export`，百科问答数据建库以及离线模型导出等
- `training`，基于 Huggingface 生态的语义召回模型训练流程

## 操作指南

可以按照下面的说明一步一步来完成离线处理。

### 1. 数据建库

我们基于开放的百万级百科问答[数据](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa)来构建检索库，需要下载此数据并放在 `index_and_export/data/baike` 目录中，整个建库过程具体说明如下：

首先，需要切换到问答数据的建库代码目录：

```bash
cd index_and_export/
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

### 2. 模型导出

我们这里的问答检索使用了 query-doc 对称双塔模型，因为一定要保证离线 doc 建库模型和在线 query 推理模型的**一致性**，继续进入问答数据的建库代码目录，然后运行如下命令来导出模型：

```bash
cd index_and_export/

sh export.sh
```

模型将被导出到 `./export` 目录，导出内容主要有线上推理使用的 ONNX 模型和预处理模型 Tokenizer。

**注**：导出模型也会被拷贝到 S3，需要在 `env.sh` 中配置自己的 S3 根路径。

### 3. 模型优化

百科问答检索能力的基础 **Sentence Embedding** 模型，模型优化结果将决定检索效果，这里使用的语义检索模型正是我们在多个开源数据集上调优过的，这些模型已经开放到 [HuggingFace Hub](https://huggingface.co/DMetaSoul)，用户可以直接下载使用。同时我们也提供了一套模型离线优化的 [pipeline](training/README-CN.md)，只需要提供简单的配置就可以训练调优用户自己的模型。