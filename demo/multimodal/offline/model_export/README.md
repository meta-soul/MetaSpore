## [中文文档](./README-CN.md)

# Introduction

Here we provide a **general export specification** for [HuggingFace](https://huggingface.co/models) pre-trained model online. Just export the model according to the specifications we provide, and it can be loaded into [MetaSpore Serving](https://github.com/meta-soul/MetaSpore) for online model inference.

The current pre-trained NLP/CV models are mainly composed of two modules: **preprocessing** and **model tensor forward**. Our export specification is designed for these two parts, including the details of how modules can be exported in a general format, how each module's input and output encoding are processed, and how each module runs in pipeline.

In this article, we will introduce the model *export specification* in detail, and give the export *examples* of some classic NLP/CV models for your reference.

# Specification

When [MetaSpore Serving](https://github.com/meta-soul/MetaSpore) loading NLP/CV models for online inference, the following content should be provided:

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

It is mainly divided into two parts `export/main` and `export/preprocess`, the former is used for model tensor forward, and the latter is used for raw data preprocessing. This is the overall specification that the model needs to follow when it goes online.

Specifically, for the model tensor forward part, it actually only contains the `main/model.onnx` file. This file is a general model representation in ONNX format. Users can use the built-in `torch.onnx.export` interface of Pytorch or other methods to export model. Because ONNX itself is a highly standardized model representation format, we do not introduce additional specification requirements here.

For the preprocessing part, the situation is slightly more complicated. This is mainly because different pre-trained models may have different preprocessing methods. In order to standardize, we have introduced several specification requirements for the preprocessing, which are as follows:

1. All preprocessing related scripts and configurations need to be placed in the `preprocess` subdirectory, and the service is loaded from here.
2. The preprocessing script must be named `preprocessor.py`, and the internally defined preprocessing class needs to implement the specified interface, which will be introduced later.
3. The preprocessing depends on third-party libraries and needs to be added into the `requirements.txt` file, at least `grpcio`, `grpcio-tools`, `pyarrow`, `protobuf` should be included.

In the `preprocess` subdirectory of the online specification given above, we can see that there are many configuration files such as `*.json`, `*.txt`, etc. These are the dependencies of preprocessing, which should be provided when the model is exported.

As for the implementation of the preprocessing (`preprocessor.py`) is often related to a specific model. Our specification requires that a class suffixed with `Preprocessor` must be implemented in the file, and the class must implement at least the following interface:

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

There are also something should be noted, the input and output data of preprocessing are `dict`, where the key is of type `str` and the value is of type `bytes`. That is to say, the preprocessing needs to be responsible for the deserialization of the external upstream input data, and the serialization of the output data to pass to the downstream model tensor inference.

This is the whole content of our specification, and the following will show you how to implement this specification through a few examples.

# Examples

We'll give some examples to help you understand the specification and how to implement it.

## NLP models

Here we provide export script for the **text vector representation task**. The export model is suitable for use scenarios such as vector recall and feature service, and currently supports the most NLP pre-trained models. The usage of the export script and the specific implementation details will be introduced following. You can also extend the script to support your custom models.

### Export Model

Export command:

```bash
model_name=sentence-transformers/clip-ViT-B-32-multilingual-v1
model_key=clip-text-encoder-v1

python src/modeling_export.py --exporter text_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow

s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
```

The above code exports the HuggingFace pre-trained model [sentence-transformers/clip-ViT-B-32-multilingual-v1](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual- v1) (including preprocessing and model inference modules and configuration files, etc.), and push the exported model and data files to cloud storage (`${MY_S3_PATH}`).

The parameters of the command are described as follows:

| Name                 | Value                                                 | Desc                                        |
| -------------------- | ----------------------------------------------------- | ------------------------------------------- |
| `--exporter`         | `text_transformer_encoder`                            | The model related exporter                  |
| `--export-path`      | `./export`                                            | The path of export output                   |
| `--model-name`       | `sentence-transformers/clip-ViT-B-32-multilingual-v1` | The name of pre-trained model               |
| `--model-key`        | `clip-text-encoder-v1`                                | The unique key for exported model           |
| `--raw-inputs`       | `texts`                                               | The raw input names of preprocessor         |
| `--raw-preprocessor` | `hf_tokenizer_preprocessor`                           | The name of preprocessor                    |
| `--raw-decoding`     | `json`                                                | The decoding format for preprocessor input  |
| `--raw-encoding`     | `arrow`                                               | The encoding format for preprocessor output |

It can be seen that the export command assembles through the string key and value. Behind this, we actually wrote a set of scripts according to the specification to achieve this. The specific implementation details will be introduced following.

## Implement Specification

The export of NLP pre-trained models is split into three parts:

- **Model inference definition**  Implementation of the input and output of the model inference, forward method, and dummy input sample construction, etc. For details, see the `TextTransformerEncoder` class.
- **Preprocessing definition**  The preprocessing implements the corresponding interface according to the specification, see the `HfTokenizerPreprocessor` class for details.
- **General export and online specification prepare** This is a model/task-independent export definition. As long as the model inference and preprocessing definitions are implemented according to the previous specifications, you can use the general export method to prepare data and files that meet the  model online specification.

```
src/
├── modeling_export.py                                       # general export
├── modeling.py                                              # model inference
└── preprocessors
    ├── hf_tokenizer_preprocessor.py                         # preprocess implement
    └── hf_tokenizer_preprocessor.requirements.txt           # preprocess requires
```

## CV models

Here we provide export script for the **image vector representation task**. The export model is suitable for use scenarios such as vector recall and feature service. Currently, it supports the export of Vision Transformer series models (including **ViT/DeiT/BEiT/DINO/MAE**), the use of the export script and the specific implementation details will be introduced following. You can also extend the script to support your custom models.

Export command:

```bash
model_name=google/vit-base-patch16-224-in21k
model_key=vit-base-patch16-224-in21k

python src/modeling_export.py --exporter image_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs images --raw-preprocessor hf_extractor_preprocessor --raw-decoding bytes --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}

if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
```

The above code exports the [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) pre-trained model in a similar way to the above NLP export , the only difference is the definition of the parameters:

| Name                 | Value                               | Desc                                        |
| -------------------- | ----------------------------------- | ------------------------------------------- |
| `--exporter`         | `image_transformer_encoder`         | The model related exporter                  |
| `--export-path`      | `./export`                          | The path of export output                   |
| `--model-name`       | `google/vit-base-patch16-224-in21k` | The name of pre-trained model               |
| `--model-key`        | `vit-base-patch16-224-in21k`        | The unique key for exported model           |
| `--raw-inputs`       | `images`                            | The raw input names of preprocessor         |
| `--raw-preprocessor` | `hf_extractor_preprocessor`         | The name of preprocessor                    |
| `--raw-decoding`     | `bytes`                             | The decoding format for preprocessor input  |
| `--raw-encoding`     | `arrow`                             | The encoding format for preprocessor output |

It can be seen that the image model export command uses a different preprocessing  `hf_extractor_preprocessor` and the preprocessed input field names and decoding are also different.

Because CV and NLP follow the same specifications to implement the export script, the specific implementation details of CV export are similar to those of the NLP model, which will not be repeated here. For more information, please refer to `modeling.ImageTransformerEncoder` and `src/preprocessors/hf_extractor_preprocessor .HfExtractorPreprocessor` implementation.
