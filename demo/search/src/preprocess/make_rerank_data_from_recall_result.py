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

import sys
import json

recall_res_file = sys.argv[1]
q2qid_file = sys.argv[2]
passage_idmap_file = sys.argv[3]
passage_file_list = sys.argv[4].split(',')
output_pair_file = sys.argv[5]

recall_res = {}
rowid2pid = {}
qid2query = {}
pid2text = {}

with open(recall_res_file, 'r', encoding='utf8') as f:
    recall_res = json.load(f)  # dict: qid -> pid list

with open(q2qid_file, 'r', encoding='utf8') as f:
    for query, qid in json.load(f).items():
        qid2query[qid] = query  # dict: qid -> query

with open(passage_idmap_file, 'r', encoding='utf8') as f:
    for rowid, pid in json.load(f).items():
        rowid2pid[int(rowid)] = pid  # dict: rowid -> pid

for qid, pids in recall_res.items():
    for pid in pids:
        pid2text[pid] = ""  # dict: pid -> passage

i = -1
for p_file in passage_file_list:
    with open(p_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.strip('\r\n')
            if not line:
                continue
            i += 1
            text = line.split('\t')[2]
            pid = rowid2pid[i]
            if pid not in pid2text:
                continue
            pid2text[pid] = text

with open(output_pair_file, 'w', encoding='utf8') as f:
    for qid, pids in recall_res.items():
        query = qid2query[qid].replace('\n', ' ').replace('\t', ' ')
        for pid in pids:
            passage = pid2text[pid].replace('\n', ' ').replace('\t', ' ')
            print(qid, pid, query, passage, sep='\t', file=f)
