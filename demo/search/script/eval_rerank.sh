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

truth_file=data/dev/dev.json
pair_file=data/output/rerank.query-passage.pair.tsv
score_file=data/output/rerank.query-passage.pair.score
pred_file=data/output/rerank.query-passage.pair.json

python src/eval/convert_rerank_res_to_json.py ${pair_file} ${score_file} ${pred_file}

python src/eval/evaluation.py $truth_file $pred_file
