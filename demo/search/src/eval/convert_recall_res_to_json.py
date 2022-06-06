# adapted from: https://github.com/PaddlePaddle/RocketQA/blob/main/research/DuReader-Retrieval-Baseline/metric/convert_recall_res_to_json.py
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

# Convert the retrieval output to standard json format
# loading files: para.map.json -> mapping from row id of para to pid in md5
# loading files: q2qid.dev.json -> mapping from query in Chinese to qid in md5

import hashlib
import json
import sys
from collections import defaultdict


q2id_map = sys.argv[1]
p2id_map = sys.argv[2]
recall_result = sys.argv[3]
outputf = sys.argv[4]


# map query to its origianl ID
with open(q2id_map, "r") as fr:
    q2qid = json.load(fr)

# map para line number to its original ID
with open(p2id_map, "r") as fr:
    pcid2pid = json.load(fr)

qprank = defaultdict(list)
with open(recall_result, 'r') as f:
    for line in f.readlines():
        q, pcid, rank, score = line.strip().split('\t')
        qprank[q2qid[q]].append(pcid2pid[pcid])

# check for length
for key in list(qprank.keys()):
    assert len(qprank[key]) == 50

with open(outputf, 'w', encoding='utf-8') as fp:
    json.dump(qprank, fp, ensure_ascii=False, indent='\t')
