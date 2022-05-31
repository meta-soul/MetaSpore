source ./env.sh

# The most of NLP pre-trained models can be exported.

###
# Demo-1: Export `clip-ViT-B-32-multilingual-v1` model.
#   Note: you can use this model as CLIP's chinese text encoder.
###
model_name=sentence-transformers/clip-ViT-B-32-multilingual-v1
model_key=clip-text-encoder-v1
python src/modeling_export.py --exporter text_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
rm -rf ./export

###
# Demo-2: Export `sbert-chinese-qmc-domain-v1` model.
#   Note: you can use this model in question-question matching domain.
###
model_name=DMetaSoul/sbert-chinese-qmc-domain-v1
model_key=sbert-chinese-qmc-domain-v1
python src/modeling_export.py --exporter text_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs texts --raw-preprocessor hf_tokenizer_preprocessor --raw-decoding json --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
rm -rf ./export
