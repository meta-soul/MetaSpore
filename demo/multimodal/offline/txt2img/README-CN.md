# 以文搜图 - 图库搜索离线部分

图库搜索演示样例离线部分有：

- `index_and_export`，图库数据建库以及离线模型导出等

## 操作指南

可以按照下面的说明一步一步来完成离线处理。

### 0. 准备

这里 demo 使用了 [Unsplash Lite](https://unsplash.com/data) 图库数据，首先需要前往下载该数据，假设下载完成后图片保存在 `~/unsplash` 目录。

接下来为了演示需要，我们用 Python 内置模块搭建一个简单的图片服务器：

```
python3 -m http.server --directory ~/unsplash 8081
```

然后就可以通过 `http://127.0.0.1:8081` 来请求图片资源啦，要注意这里的图片目录和服务地址后面会用到。

### 1. 数据建库

首先，需要切换到建库代码目录：

```bash
cd index_and_export/
```

**预处理**，给定一个图片目录，生成一个较为通用的 jsonline 文件供建库使用

```bash
sh preprocess.sh ~/unsplash http://127.0.0.1:8081
```

**构建索引**，使用 [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) 预训练模型对图库进行索引，输出索引数据每行一个文档对象

```bash
sh build.sh
```

**推送倒排和正排**，把倒排（向量）和正排（文档字段）数据分别推送到各个组件服务端

```bash
# 倒排数据推送到 Milvus 集群
sh push_milvus.sh

# 正排数据推送到 MongoDB 集群
sh push_mongo.sh
```

**注**：请在 `env.sh` 文件中给出自己 Milvus 和 MongoDB 集群配置信息。

检索入库数据样例：

```
# Milvus
{"id": 0, "image_emb": [-0.058228425681591034, -0.006109456066042185, -0.005825215484946966,...,-0.04344896227121353, 0.004351312294602394]}

# MongoDB
{"name" : "zZzKLzKP24o.jpg", "url" : "http://127.0.0.1:8081/zZzKLzKP24o.jpg", "queryid" : "0" }
```

### 2. 模型导出

以文搜图采用以 [CLIP](https://arxiv.org/abs/2103.00020) 预训练模型为基础的双塔检索架构：

![](https://github.com/openai/CLIP/raw/main/CLIP.png)

> ref: https://github.com/openai/CLIP/raw/main/CLIP.png

**离线**建库时需要用到双塔的**图像侧**模型，**线上**检索时需要用到双塔的**文本侧**模型，这里需要把文本侧模型导出以供线上 [MetaSpore Serving](https://github.com/meta-soul/MetaSpore) 推理。

由于我们检索基于中文场景，所以这里文本侧模型采用了支持中文理解的[多语言版](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)预训练模型。

这里演示如何把离线模型导出并推送到线上：

```bash
cd index_and_export/

sh export.sh
```

模型将被导出到 `./export` 目录，导出内容主要有线上推理使用的 ONNX 模型和预处理模型 Tokenizer。

**注**：导出模型也会被拷贝到 S3，需要在 `env.sh` 中配置自己的 S3 根路径。