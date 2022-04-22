# 简介

NLP/CV 预处理服务——基于 Python gRPC 框架封装了 NLP/CV 相关的预处理逻辑。

预处理逻辑的具体实现样例见[代码](./hf_preprocessor/hf_tokenizer.py)，每个预处理逻辑会封装成类似这样的类型，支持通过 `load()`, `predict()` 方法来进行预处理模型加载和预测。

目前支持 NLP tokenizer 预处理逻辑的封装，对应服务请求&响应 proto 定义如下：

```proto
# 模型调用
message HfTokenizerRequest {
  string model_name = 1;
  map<string, string>   parameters = 3;
  map<string, bytes>    payload    = 5;
}

message HfTokenizerResponse {
  map<string, bytes>    payload    = 1;
  map<string, string>   extras     = 3;
}

# 模型推送服务
message HfTokenizerPushRequest {
  string model_name = 1;
  string model_url = 2;
}

message HfTokenizerPushResponse {
  int32 status = 1;
  string msg = 2;
}
```

注：该预处理 Python gRPC 服务所用 proto 文件（server 端），必须跟 multimodal serving 的 [proto 文件](../multimodal_serving/src/main/protos/hf_preprocessor.proto)（client 端）严格一致。


# 用法

下面演示 gRPC 服务启动以及 NLP tokenizer 预处理服务的调用：

1. 启动服务

    ```shell
    sh server.sh
    ```

2. 预处理模型推送并完成调用测试

    ```shell
    sh client.sh

    # push model
    aws s3 cp s3://dmetasoul-bucket/demo/nlp-algos-transformer/models/sbert-chinese-qmc-domain-v1/sbert-chinese-qmc-domain-v1.tar.gz ./
    python client.py push bert-qmc-v1 ./sbert-chinese-qmc-domain-v1.tar.gz

    # call service
    python client.py tokenize bert-qmc-v1 "预处理服务——基于 Python gRPC 框架"
    ```
