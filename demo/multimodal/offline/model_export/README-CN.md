# 简介

我们在这里提供了一套 [HuggingFace](https://huggingface.co/models) 预训练模型的**通用上线机制**，只需要按照我们提供的规范来导出模型，就可以被载入 [MetaSpore Serving](https://github.com/meta-soul/MetaSpore) 进行在线模型推理。

具体来说，目前 NLP/CV 等领域的预训练模型从模块结构上划分来看的话，主要由**预处理**和**模型张量计算**两部分构成，因此模型的导出规范同样是针对这两部分设计的，包括模块如何以通用格式来规范化导出、各模块输入和输出编解码如何处理以及各个模块如何串联运行等细节内容。

我们将在本文详细介绍模型*导出上线规范*，并给出 NLP/CV 领域若干较为主流模型的导出*演示样例*供大家参考。

-----

*更新：*

- *2022.06.09* 新增**文本分类**模型的导出支持
- *2022.06.01* 新增**文本向量表示**和**图片向量表示**模型的导出支持

# 规范

[MetaSpore Serving](https://github.com/meta-soul/MetaSpore) 载入 NLP/CV 模型进行推理时，需要提供以下规范内容：

```
export
├── main
│   └── model.onnx
└── preprocess
    ├── config.json
    ├── export_config.json
    ├── preprocessor.py
    ├── requirements.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

从逻辑和目录结构上划分的话，主要分为两部分 `export/main` 和 `export/preprocess`，前者用于模型的张量计算，后者用于预处理计算。这个就是模型上线需要遵循的整体规范，也即在离线模型导出时需要准备好这些内容。

具体来看，对于模型张量计算部分，其实仅包含 `main/model.onnx` 文件，此文件是 ONNX 格式的通用模型表示，用户可以利用 Pytorch 内置的 `torch.onnx.export` 接口或其它方式将模型张量计算部分导出。因为 ONNX 本身就是规范程度较高的模型表示格式，因此这里我们并没有引入额外规范要求。

对于预处理部分，情况就要稍微复杂些，这主要在于不同预训练模型可能会有不同的预处理逻辑，为了统一规范化上线，我们对预处理逻辑引入了若干规范要求，具体有：

1. 所有预处理有关的脚本以及配置需要放在 `preprocess` 子目录中，服务载入时从这里进行加载
2. 预处理逻辑脚本必须命名为 `preprocessor.py`，并且内部定义的预处理类需要实现指定接口，后面会具体开展介绍
3. 预处理逻辑依赖第三方库需要写入到 `requirements.txt` 文件中，服务加载时会安装依赖，至少需要包括 `grpcio`, `grpcio-tools`, `pyarrow`, `protobuf` 这几个依赖

前面给出的上线规范 `preprocess` 子目录中，可以看到有较多 `*.json`, `*.txt` 等配置文件，这些是都是预处理的相关依赖，应该由模型导出时提供。

至于预处理逻辑的具体实现（也即 `preprocessor.py`）往往跟特定的模型有关，我们规范约定在此脚本文件中必须实现一个以 `Preprocessor` 为后缀的类型，并且该类型至少需要实现以下接口：

```python
class ExamplePreprocessor(object):
    def __init__(self, config_dir):
        # load preprocessor from `config_dir`, this is the `export/preprocess`
        pass
        # define preprocessor input&output
        self._input_names = 'texts',
        self._output_names = 'input_ids', 'token_type_ids', 'attention_mask'

    # define input names
    @property
    def input_names(self):
        return self._input_names

    # define output names
    @property
    def output_names(self):
        return self._output_names

    # implement the preprocessor's predict method
    def predict(self, inputs):
        raise NotImplementedError
```

还有一些地方需要注意，在预处理输入和输出数据格式都是字典，其中键是字符串类型，值是字节类型。也就是说预处理内部需要负责外部上游输入数据的反序列化，以及输出数据序列化后传递给下游的模型张量计算。

这就是我们模型上线规范的全部内容，下面会通过几个演示样例来向大家说明具体如何实现这套规范。

# 演示用法

## NLP 模型

这里我们针对**文本向量表示任务**提供了导出相关的脚本逻辑，导出模型适用于向量召回、特征服务等使用场景，目前能够支持大多数 NLP 预训练模型的导出。下面将介绍导出脚本的使用以及具体的实现细节，大家也可以通过扩展这里的脚本来支持自定义模型的导出。

目前我们支持导出的 NLP 模型/任务汇总如下：

| 任务         | Exporter                     | Preprocessor                |
| ------------ | ---------------------------- | --------------------------- |
| 文本向量表示 | `text_transformer_encoder`   | `hf_tokenizer_preprocessor` |
| 双塔检索模型 | `text_transformer_encoder`   | `hf_tokenizer_preprocessor` |
| 文本分类模型 | `seq_transformer_classifier` | `hf_tokenizer_preprocessor` |

### 导出

导出命令：

```bash
model_name=sentence-transformers/clip-ViT-B-32-multilingual-v1
model_key=clip-text-encoder-v1

python src/modeling_export.py --exporter text_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow

s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
```

以上这段代码完成了 HuggingFace 预训练模型 [sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1) 的导出（含有预处理和模型计算模块以及配置文件等），并将导出后的模型以及数据文件推送到了云存储（`${MY_S3_PATH}`）。

导出命令有关参数说明如下：

| Name                 | Value                                                 | Desc                                       |
| -------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `--exporter`         | `text_transformer_encoder`                            | 导出器：跟模型以及任务相关联的导出逻辑实现 |
| `--export-path`      | `./export`                                            | 导出目录地址                               |
| `--model-name`       | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | HuggingFace 的预训练模型名                 |
| `--model-key`        | `clip-text-encoder-v1`                                | 预训练模型导出后的唯一标识串               |
| `--raw-inputs`       | `texts`                                               | 预处理的输入字段                           |
| `--raw-preprocessor` | `hf_tokenizer_preprocessor`                           | 预处理的逻辑定义                           |
| `--raw-decoding`     | `json`                                                | 预处理的输入反序列化/解码方式              |
| `--raw-encoding`     | `arrow`                                               | 预处理的输出序列化/编码方式                |

可以看到导出命令是通过字符串键值这种方式来组装导出逻辑的，而这个背后其实是我们按照规范编写了一组脚本来实现的，下面将介绍具体的实现细节。

### 实现细节

NLP 预训练模型的导出拆分为三部分来实现的：

- **模型计算的封装定义**   对模型计算部分的输入和输出、前向计算以及 dummy 输入样例构造等规范的实现，具体参见 `TextTransformerEncoder` 类。
- **预处理逻辑的定义**   对预处理逻辑的实现，预处理逻辑按照规范实现了相应的接口，具体参见 `HfTokenizerPreprocessor` 类。
- **通用导出适配上线规范**   这个是模型/任务无关的导出定义，只要按照前面规范将模型计算和预处理定义实现好，就可以借助该通用导出逻辑来制作符合上线规范的配置以及模型等数据文件。

```
src/
├── modeling_export.py                                       # 通用导出，适配上线规范
├── modeling.py                                              # 模型计算封装
└── preprocessors
    ├── hf_tokenizer_preprocessor.py                         # 预处理逻辑实现
    └── hf_tokenizer_preprocessor.requirements.txt           # 预处理依赖定义
```

## CV 模型

这里我们针对**图片向量表示任务**提供了导出相关的脚本逻辑，导出模型适用于向量召回、特征服务等使用场景，目前支持 Vision Transformer 系列模型导出（包括 ViT/DeiT/BEiT/DINO/MAE），下面将介绍导出脚本的使用以及具体的实现细节，大家也可以通过扩展这里的脚本来支持自定义模型的导出。

目前我们支持导出的 CV 模型/任务汇总如下：

| 任务         | Exporter                    | Preprocessor                |
| ------------ | --------------------------- | --------------------------- |
| 图片向量表示 | `image_transformer_encoder` | `hf_extractor_preprocessor` |

导出命令：

```bash
model_name=google/vit-base-patch16-224-in21k
model_key=vit-base-patch16-224-in21k

python src/modeling_export.py --exporter image_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs images --raw-preprocessor hf_extractor_preprocessor --raw-decoding bytes --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}

if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
```

以上代码对 [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) 预训练模型进行导出，使用方式跟上述 NLP 导出类似，唯一不同之处在于参数的定义：

| Name                 | Value                               | Desc                          |
| -------------------- | ----------------------------------- | ----------------------------- |
| `--exporter`         | `image_transformer_encoder`         | 跟图片向量表示对应的导出器    |
| `--export-path`      | `./export`                          | 导出目录地址                  |
| `--model-name`       | `google/vit-base-patch16-224-in21k` | HuggingFace 的预训练模型名    |
| `--model-key`        | `vit-base-patch16-224-in21k`        | 预训练模型导出后的唯一标识串  |
| `--raw-inputs`       | `images`                            | 预处理的输入字段              |
| `--raw-preprocessor` | `hf_extractor_preprocessor`         | 预处理的逻辑定义              |
| `--raw-decoding`     | `bytes`                             | 预处理的输入反序列化/解码方式 |
| `--raw-encoding`     | `arrow`                             | 预处理的输出序列化/编码方式   |

可以看到图片向量表示的模型导出使用了不同的预处理逻辑 `hf_extractor_preprocessor` 并且预处理的输入字段名和格式也不一样了。

因为 CV 和 NLP 是遵循同一套规范来实现导出逻辑的，所以 CV 导出的具体实现细节跟 NLP 模型类似，这里不做赘述，想了解更多可以参考 `modeling.ImageTransformerEncoder` 和 `src/preprocessors/hf_extractor_preprocessor.HfExtractorPreprocessor` 具体实现。

