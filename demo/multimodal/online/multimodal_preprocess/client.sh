#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

model_key=sbert-chinese-qmc-domain-v1

MY_S3_PATH='your S3 bucket'

# Push tokenizer model to service
#model_tar_url=http://172.31.0.197:8081/bert_tokenizer.tar.gz
#python client.py push ${model_key} ${model_tar_url}
aws s3 cp ${MY_S3_PATH}/demo/nlp-algos-transformer/models/sbert-chinese-qmc-domain-v1/sbert-chinese-qmc-domain-v1.tar.gz ./
python client.py push ${model_key} ./sbert-chinese-qmc-domain-v1.tar.gz

# Call tokenizer preprocessor
python client.py tokenize ${model_key} 北京天安门
