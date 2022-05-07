## [中文介绍](README-CN.md)

# Text-To-Image - search image library by text offline

Text-to-image demo offline including:

- `index_and_export` for model export and database indexing

## User Guide

You can follow the below instructions step by step to complete the offline processing.

### 0. Preparation

This demo uses the [Unsplash Lite](https://unsplash.com/data) image dataset. First, you need to download the dataset. It is assumed that the image is saved in the `~/unsplash` directory after the download is done.

Next, we should setup a simple image server based on Python's built-in `http.server`:

```
python3 -m http.server --directory ~/unsplash 8081
```

Now you can request image throught `http://127.0.0.1:8081`

**Note**: the image saved directory and server url will be used in the next step.

### 1. Database Indexing

First, change to the indexing directory:

```bash
cd index_and_export/
```

**preprocessing**: input a local image directory, a like-jsonline file will be generated for building

```
sh preprocess.sh ~/unsplash http://127.0.0.1:8081
```

**indexing**: we'll use [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) pre-trained model to encode the images

```bash
sh build.sh
```

**pushing**: the indexing docs will be pushed into Milvus/MongoDB services

```bash
# push embeddings into Milvus
sh push_milvus.sh

# push doc fields into MongoDB
sh push_mongo.sh
```

**Note**: Please config your Milvus and MongoDB params in the `env.sh` file.

The indexing docs examples:

```
# Milvus
{"id": 0, "image_emb": [-0.058228425681591034, -0.006109456066042185, -0.005825215484946966,...,-0.04344896227121353, 0.004351312294602394]}

# MongoDB
{"name" : "zZzKLzKP24o.jpg", "url" : "http://127.0.0.1:8081/zZzKLzKP24o.jpg", "queryid" : "0" }
```

### 2. Model Export

We take the [CLIP](https://arxiv.org/abs/2103.00020) pre-trained model as our two-tower retrieval system's backbone:

![](https://github.com/openai/CLIP/raw/main/CLIP.png)

> ref: https://github.com/openai/CLIP/raw/main/CLIP.png

The image side of CLIP will be used to index database. The text side of CLIP will be exported and used to online model [inference serving](https://github.com/meta-soul/MetaSpore).

Because we focus chinese retrieval scene, we use a [pre-trained model](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) that supports chinese query understanding.

Export/push onnx and tokenizer model:

```bash
cd index_and_export/

sh export.sh
```

**Note**: Please config your S3 path in the `env.sh` file.