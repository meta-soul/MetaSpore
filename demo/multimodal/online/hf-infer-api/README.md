# Huggingface推理调用 — 线上服务

Huggingface推理调用线上服务，同时支撑**文本翻译**、**文本续写**、**文本生成图像**、**文本到文本**等任务。

## 0. prepare

准备工作主要有：

- **上传模型**

本服务支持使用Huggingface hub上已保存的模型，若使用自己本地训练的模型，需先上传至Huggingface hub

```bash
    # 安装插件

    pip install huggingface_hub

    # 登录（需要hugging face账号对应的token）

    huggingface-cli login

    # 在hugging face网页端创建仓库

    # 克隆远程仓库到本地

    git clone https://huggingface.co/<your-username>/<your-model-name>

    # 安装git文件系统以上传大文件

    git lfs install

    # 使用git add/commit/push上传模型及相关文件即可
```

- **设置token**

服务启用前需要在[Huggingface](https://huggingface.co/settings/tokens)网站获取token




```bash
    # 在目录下创建.env文件
    touch  .env

    # 在.env文件中添加获取的token

    HF_API_TOKEN=
```


## 1. 服务启动和调用

### 1.1 启动服务

服务启动：

```bash
    sh start.sh
```

### 1.2 服务调用

调用翻译模型，并保存结果
```bash
    curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-translation?model_type=zh2en -o out.translation.json
```


## 2. 服务接口支持的参数

本服务是在线模型的推理调用，同时支撑**文本翻译**、**文本续写**、**文本生成图像**、**文本到文本**等任务。

### 2.1 通用参数
**url参数**

| 参数名称 | 参数用途 | 参数类型 | 是否必填 | 默认值或可选项 |
| ----- | ----- | ----- | ----- | :----: |
| task_type | 任务类型 | str | 是 | 必须为text-completion，text-to-text，text-to-image，text-translation之一 |
| model_name | 模型名称 | str |  否| - |
| model_type | 翻译任务语言 | str | 否 | zh2en |


**请求体参数**

| 参数名称 | 参数用途 | 参数类型 | 是否必填 | 默认值或可选项 |
| ----- | ----- | ----- | ----- | :----: |
| inputs | 模型输入 | str | 是 | - |
| num_sentences | 返回结果数量 | int |  否| 1 |

### 2.2任务差异参数
· **文本续写任务**

请求示例
```bash
    curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-completion -o out.completion.json
```

| 参数名称 | 参数用途 | 参数类型 | 是否必填 | 默认值或可选项 |
| ----- | ----- | ----- | ----- | :----: |
| do_sample | 是否使用采样策略 | bool | 否 | True |
| top_k | 采样参数top_k | int |  否| 30 |
| top_p | 采样参数top_p | float |  否| 1(0-1之间) |
| max_new_tokens | 续写文本最大长度 | int |  否| 128 |
| return_full_text | 返回文本是否包含输入部分 | bool |  否| False |


· **文本到文本任务**

请求示例
```bash
    curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-to-text -o out.t2t.json
```

| 参数名称 | 参数用途 | 参数类型 | 是否必填 | 默认值或可选项 |
| ----- | ----- | ----- | ----- | :----: |
| do_sample | 是否使用采样策略 | bool | 否 | True |
| top_k | 采样参数top_k | int |  否| 30 |
| top_p | 采样参数top_p | float |  否| 1(0-1之间) |
| max_new_tokens | 返回文本最大长度 | int |  否| 128 |

· **文本翻译任务**

请求示例
```bash
    curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙","args":"{\"num_sentences\":3}"}' http://127.0.0.1:8098/api/infer/text-translation?model_type=zh2en -o out.translation.json
```
| 参数名称 | 参数用途 | 参数类型 | 是否必填 | 默认值或可选项 |
| ----- | ----- | ----- | ----- | :----: |
| do_sample | 是否使用采样策略 | bool | 否 | True |
| top_k | 采样参数top_k | int |  否| 30 |
| top_p | 采样参数top_p | float |  否| 1(0-1之间) |

· **文本生成图像任务**

请求示例
```bash
    curl -H "Content-Type: application/json" -X POST -d '{"inputs":"连衣裙"}' http://127.0.0.1:8098/api/infer/text-to-image -o out.t2i.json
```

