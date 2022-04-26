# 基线模型

本项目基线模型主要有以下几种：

- BERT + 均值池化
- BERT + 白化后处理
- SimBERT + 均值池化
- fastText 词向量 + 均值池化
- Tencent 词向量 + 均值池化

运行基线模型：

```
sh run.sh lcqmc ../../datasets/processed/lcqmc/dev.tsv > output/lcqmc.log 2> /dev/null &
```

*其中第一个参数指定基线模型唯一键名，第二个参数指定用来评估句对相似度的评测文件*

# 依赖工具

1. [fastText 0.9.2](https://github.com/facebookresearch/fastText) 从源代码进行安装，放在目录 `~/tools/fastText-0.9.2`：

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ mkdir build && cd build && cmake ..
$ make && make install
```

2. fastText 对中文处理依赖切词工具 [stanford-segmenter-4.2.0](https://nlp.stanford.edu/software/segmenter.shtml)，下载后放在目录 `~/tools/stanford-segmenter-2020-11-17`

3. 项目用到了[腾讯词向量文件](https://ai.tencent.com/ailab/nlp/en/download.html)（d200, small），下载后放在 `~/tools/TencentEmbedding/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt`
