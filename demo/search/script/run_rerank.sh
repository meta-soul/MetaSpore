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

# you should change `model` as your finetuned model path
model=output/train_ce_multiclass/2022_06_09_18_23_09/step_26000
pair_file=data/output/rerank.query-passage.pair.tsv
score_file=data/output/rerank.query-passage.pair.score

mkdir -p ./logs
nohup sh script/rerank.sh ${model} ${pair_file} ${score_file} > ./logs/rerank.log 2>&1 &
