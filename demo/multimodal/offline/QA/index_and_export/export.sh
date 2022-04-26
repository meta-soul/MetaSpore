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
