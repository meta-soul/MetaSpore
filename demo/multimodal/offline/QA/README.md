## [中文介绍](README-CN.md)

# Text-To-Text - QA offline

QA text-to text demo offline including:

- `index_and_export` for model export and database indexing
- `training` for offline model training pipeline

## User Guide

You can follow the below instructions step by step to complete the offline processing.

### 1. Database Indexing

First, you need download the [raw data](https://github.com/brightmart/nlp_chinese_corpus#3%E7%99%BE%E7%A7%91%E7%B1%BB%E9%97%AE%E7%AD%94json%E7%89%88baike2018qa) and put them into `index_and_export/data/baike` directory, then change to the indexing directory:

```bash
cd index_and_export/
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

### 2. Model Export

We use a symmetrical two-tower model for semantic retrieval, so we must ensure consistency of online and offline models. We export the above offline indexing model:

```bash
cd index_and_export/

sh export.sh
```

**Note**: Please config your S3 path in the `env.sh` file.

### 3. Model Training

We open our sentence embedding models in the [HuggingFace Hub](https://huggingface.co/DMetaSoul). Everyone can use those pre-trained model to build their own text-to-text retrieval system. At the same time, we provide a model [training pipeline](training) for community to train their own models on the custom tasks.