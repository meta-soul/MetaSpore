source ./env.sh

model_key=sbert-chinese-qmc-domain-v1
s3_path=${MY_S3_PATH}/demo/nlp-algos-transformer/models/${model_key}/

python src/modeling_export.py --model-name DMetaSoul/sbert-chinese-qmc-domain-v1 --export-path ./export --model-key ${model_key}

if [ $? == 0 ]; then
    # push tokenizer model to s3
    echo "Push tokenizer to S3..."
    tar cvzf ${model_key}.tar.gz ./export --exclude export/model.onnx
    aws s3 cp ${model_key}.tar.gz ${s3_path}
    echo "Done!"

    # push onnx model to s3
    echo "Push model to S3..."
    aws s3 cp ./export/model.onnx ${s3_path}
    echo "Done!"
fi
