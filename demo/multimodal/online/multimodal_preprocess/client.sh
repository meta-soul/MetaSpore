model_key=sbert-chinese-qmc-domain-v1

MY_S3_PATH='your S3 bucket'

# Push tokenizer model to service
#model_tar_url=http://172.31.0.197:8081/bert_tokenizer.tar.gz
#python client.py push ${model_key} ${model_tar_url}
aws s3 cp ${MY_S3_PATH}/demo/nlp-algos-transformer/models/sbert-chinese-qmc-domain-v1/sbert-chinese-qmc-domain-v1.tar.gz ./
python client.py push ${model_key} ./sbert-chinese-qmc-domain-v1.tar.gz

# Call tokenizer preprocessor
python client.py tokenize ${model_key} 北京天安门
