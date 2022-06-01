source ./env.sh

# ViT/DeiT/BEiT/DINO/ViT_MAE and other ViT variants can be exported.

###
# Demo-1: Export ViT model.
#   Note: you can use this model for image-to-image searching.
###
model_name=google/vit-base-patch16-224-in21k
model_key=vit-base-patch16-224-in21k
python src/modeling_export.py --exporter image_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs images --raw-preprocessor hf_extractor_preprocessor --raw-decoding bytes --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
rm -rf ./export


###
# Demo-2: Export BEiT model.
#   Note: you can use this model for image-to-image searching.
###
model_name=microsoft/beit-base-patch16-224-pt22k-ft22k
model_key=beit-base-patch16-224-pt22k-ft22k
#model_name=facebook/deit-base-patch16-224
#model_key=deit-base-patch16-224
python src/modeling_export.py --exporter image_transformer_encoder --export-path ./export --model-name ${model_name} --model-key ${model_key} --raw-inputs images --raw-preprocessor hf_extractor_preprocessor --raw-decoding bytes --raw-encoding arrow
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}
if [ $? == 0 ]; then
    aws s3 cp --recursive ./export ${s3_path}
fi
rm -rf ./export
