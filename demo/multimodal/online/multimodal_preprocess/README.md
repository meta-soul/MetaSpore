## [中文介绍](README-CN.md)

# Introduction

This project provides a NLP/CV preprocess service based on Python gRPC framework.

The proto definition as follows:

```proto
# call tokenizer
message HfTokenizerRequest {
  string model_name = 1;
  map<string, string>   parameters = 3;
  map<string, bytes>    payload    = 5;
}

message HfTokenizerResponse {
  map<string, bytes>    payload    = 1;
  map<string, string>   extras     = 3;
}

# push tokenizer
message HfTokenizerPushRequest {
  string model_name = 1;
  string model_url = 2;
}

message HfTokenizerPushResponse {
  int32 status = 1;
  string msg = 2;
}
```

**Note**: You have to make sure that the proto of preprocess service is the same as [multimodal serving proto](../multimodal_serving/src/main/protos/hf_preprocessor.proto).

# Usage

Start service:

```shell
sh server.sh
```

Push preprocess model:

```shell
MY_S3_PATH='your S3 bucket'
aws s3 cp ${MY_S3_PATH}/demo/nlp-algos-transformer/models/sbert-chinese-qmc-domain-v1/sbert-chinese-qmc-domain-v1.tar.gz ./
python client.py push bert-qmc-v1 ./sbert-chinese-qmc-domain-v1.tar.gz
```

Call/Test the pushed model:

```shell
python client.py tokenize bert-qmc-v1 "预处理服务——基于 Python gRPC 框架"
```

------

For our Multimodal Retrieval Demo the following models should be pushed into your preprocess service:

```bash
MY_S3_PATH='your S3 bucket'

# search text by text --- QA demo
aws s3 cp ${MY_S3_PATH}/demo/nlp-algos-transformer/models/sbert-chinese-qmc-domain-v1/sbert-chinese-qmc-domain-v1.tar.gz ./
python client.py push sbert-chinese-qmc-domain-v1 ./sbert-chinese-qmc-domain-v1.tar.gz

# search image by text --- TxtToImg
aws s3 cp ${MY_S3_PATH}/demo/nlp-algos-transformer/models/clip-text-encoder-v1/clip-text-encoder-v1.tar.gz ./
python client.py push clip-text-encoder-v1 ./clip-text-encoder-v1.tar.gz
```