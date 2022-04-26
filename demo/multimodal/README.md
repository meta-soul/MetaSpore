## [中文介绍](README-CN.md)

# Multimodal Retrieval Demo

Deep learning has made innovative breakthroughs in computer vision (CV), natural language processing (NLP) and other fields, and the multimodal understanding technology of joint modeling between different modal data has become more and more mature. The SOTA technologies have made it possible to bridge the semantic gap between multimodal data, but factors such as complicated offline model optimization, data processing, high online inference and experimental costs have hindered the landing and popularization of multimodal technology. The demo of this project will focus on **search text by text** (T2T), **search image by text** (T2I) and other multimodal retrieval scenarios, to demonstrate to you how to easily access [HuggingFace](https:/ /huggingface.co/) multimodal pretrained models with our **MetaSpore** technology ecology.

The demo will provide a whole of solution from offline data processing to online retrieval services:

- **Online Service**, a multimodal retrieval online system, supporting multi-scenario semantic retrieval such as text search, image search, etc., covering retrieval web UI, query preprocessing, retrieval matching/ranking algorithm and other services.
- **Offline Processing**, covering the offline part of each semantic retrieval scenario of the demo, mainly including offline model training and export, retrieval database indexing and pushing.

## 1. Online Service

The Multimodal Demo online service consists of the following parts:

1. [multimodal_web](online/multimodal_web), a front-end service for multimodal demo, providing a web UI interface for users to experience multimodal retrieval service.
2. [multimodal_serving](online/multimodal_serving), retrieval online service for multimodal demo, including the entire algorithm pipeline such as a/b experimental configuration, query preprocessing, matching, ranking, summary, etc.
3. [multimodal_preprocess](online/multimodal_preprocess), wrap the multimodal pretrained model preprocessing methods (including text/image, etc.), and provides service through the gRPC api.

**Note:** The above services form dependencies in sequence from front to back, so to build a multimodal demo, you need to run each service in sequence from back to front, see more in the online [guide](online/README.md). Of course, before doing this, remember to take done the [offline](./offline) part.

## 2. Offline Processing

The offline processing mainly includes the offline model export, data indexing and pushing and so on. Since the offline part of the T2T and the T2I demo is similar, here we focus on the [offline part of the T2T](offline/QA/README.md) based on the **Encyclopedia Q&A** [database](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa).

### 2.1 Database Indexing

First, you need download the [raw data](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa) and put them into `offline/QA/data/baike` directory, then change to the indexing directory:

```bash
cd offline/QA/index_and_export/
```

**preprocessing**

```
sh preprocess.sh
```

**indexing**

```bash
sh build.sh
```

**pushing**

```bash
# push embeddings into Milvus
sh push_milvus.sh

# push doc fields into MongoDB
sh push_mongo.sh
```

**Note**: Please config your Milvus and MongoDB params in the `env.sh` file.

### 2.2 Model Export

We use a symmetrical two-tower model for semantic retrieval, so we must ensure consistency of online and offline models. We export the above offline indexing model:

```bash
sh export.sh
```

**Note**: Please config your S3 path in the `env.sh` file.

### 2.3 Model Training

We open our sentence embedding models in the [HuggingFace Hub](https://huggingface.co/DMetaSoul). Everyone can use those pre-trained model to build their own text-to-text retrieval system. At the same time, we provide a model training pipeline for community to train their own models on the custom tasks.