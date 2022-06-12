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

import os
import sys
import random

from tqdm import tqdm

if __name__ == '__main__':
    pos_file = sys.argv[1]
    cand_file = sys.argv[2]
    passage_file_list = sys.argv[3].split(',')
    neg_ratio = int(sys.argv[4])
    output_fmt = sys.argv[5]
    out_file = sys.argv[6]
    assert output_fmt in ['pair', 'triplet']

    print("Load positive...")
    queries = {}
    fin = open(pos_file, 'r', encoding='utf8')
    for line in fin:
        query, pos = line.strip().split('\t')
        if query not in queries:
            queries[query] = set()
        queries[query].add(pos)
    fin.close()
    print("Load positive done!")

    print("Load candidates ...")
    cands = {}
    passages = {}
    fin = open(cand_file, 'r', encoding='utf8')
    for line in fin:
        query, pid = line.strip('\r\n').split('\t')[:2]
        pid = int(pid)
        if query not in cands:
            cands[query] = set()
        cands[query].add(pid)
        passages[pid] = ""
    fin.close()
    print("Load candidates done!")

    print("Load docs ...")
    i = -1
    for p_file in passage_file_list:
        with open(p_file, 'r', encoding='utf8') as fin:
            for line in fin:
                line = line.strip('\r\n')
                if not line:
                    continue
                i += 1
                if i not in passages:
                    continue
                passages[i] = line.split('\t')[2]
    print("Load docs done!")

    print("Build negatives ...")
    fout = open(out_file, 'w', encoding='utf8')
    for query in tqdm(queries):
        pos_set = queries[query]

        if query not in cands:
            continue
        cand_set = set([passages[pid] for pid in cands[query] if passages.get(pid)])
        if not cand_set:
            continue

        neg_list = list(cand_set - pos_set)
        random.shuffle(neg_list)
        num_neg = min(int(neg_ratio*len(pos_set)), len(neg_list))
        
        if output_fmt == 'pair':
            for pos in pos_set:
                print(query, pos, 1, sep='\t', file=fout)
            for neg in neg_list[:num_neg]:
                print(query, neg, 0, sep='\t', file=fout)
        elif output_fmt == 'triplet':
            for pos in pos_set:
                for neg in neg_list[:num_neg]:
                    print(query, pos, neg, sep='\t', file=fout)
    print("Build negatives done!")
