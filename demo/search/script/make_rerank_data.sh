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

python -u src/preprocess/make_rerank_data_from_recall_result.py \
    data/output/dev.recall.top50.json \
    data/dev/q2qid.dev.json \
    data/passage-collection/passage2id.map.json \
    data/passage-collection/part-00,data/passage-collection/part-01,data/passage-collection/part-02,data/passage-collection/part-03 \
    data/output/rerank.query-passage.pair.tsv
