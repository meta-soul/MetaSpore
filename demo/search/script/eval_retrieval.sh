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

QUERY2ID=./data/dev/q2qid.dev.json
PARA2ID=./data/passage-collection/passage2id.map.json
MODEL_OUTPUT=./data/output/dev.recall.top50
REFERENCE_FIEL=./data/dev/dev.json
PREDICTION_FILE=./data/output/dev.recall.top50.json

python src/eval/convert_recall_res_to_json.py ${QUERY2ID} ${PARA2ID} ${MODEL_OUTPUT} ${PREDICTION_FILE}

python src/eval/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
