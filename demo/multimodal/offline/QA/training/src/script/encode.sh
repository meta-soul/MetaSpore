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

mkdir -p ../output/embeddings

:<<EOF
#model_path=./output/training_afqmc-qmc/2022-02-11_17-27-23
model_path=./output/training_afqmc_distill_zh/2022-02-18_00-00-17
in_file=../datasets/processed/sentences_zh/dev.tsv
out_file=./output/embeddings/sentences_zh_dev_emb.txt
nohup python encode.py --input-file ${in_file} \
    --output-file ${out_file} \
    --model ${model_path} \
    --batch-size 32 --device cpu --max-seq-len 256 \
    > logs/encode.log-distill 2>&1 &
EOF

model_path=output/training_dtm_adaptive_domain-qmc_default_v3
in_file=../datasets/QA/baike_qa.txt
out_file=../datasets/QA/baike_qa.emb.txt
nohup python encode.py --input-file ${in_file} \
    --output-file ${out_file} \
    --model ${model_path} \
    --batch-size 256 --device cuda:0 --max-seq-len 256 \
    > logs/encode.log 2>&1 &
