model_key=bert

# Push tokenizer model to service
model_tar_url=http://172.31.0.197:8081/bert_tokenizer.tar.gz
python client.py push ${model_key} ${model_tar_url}

# Call tokenizer preprocessor
python client.py tokenize ${model_key} 北京天安门
