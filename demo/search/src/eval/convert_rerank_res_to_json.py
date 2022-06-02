# adapted from: https://github.com/PaddlePaddle/RocketQA/blob/main/research/DuReader-Retrieval-Baseline/metric/convert_rerank_res_to_json.py
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

import csv
import sys
import json
from collections import defaultdict

score_f = sys.argv[1]
id_f = sys.argv[2]

outputf = 'output/cross_res.json'

scores = []
q_ids = []
p_ids = []
q_dic = defaultdict(list)

with open(score_f, 'r') as f:
    for line in f:
        scores.append(float(line.strip()))

with open(id_f, 'r') as f:
    for line in f:
        v = line.strip().split('\t')
        q_ids.append((v[0]))
        p_ids.append((v[1]))

for q, p, s in zip(q_ids, p_ids, scores):
    q_dic[q].append((s, p))

output = []
for q in q_dic:
    rank = 0
    cands = q_dic[q]
    cands.sort(reverse=True)
    for cand in cands:
        rank += 1
        output.append([q, cand[1], rank])
        if rank > 49:
            break

with open(outputf, 'w') as f:
    res = dict()
    for line in output:
        qid, pid, rank = line
        if qid not in res:
            res[qid] = [0] * 50
        res[qid][int(rank) - 1] = pid
    json.dump(res, f, ensure_ascii=False, indent='\t')
