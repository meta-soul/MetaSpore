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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from modeling import TransformerDualEncoder

def load_tsv(file):
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            yield line.strip().split('\t')

if __name__ == '__main__':
    query_file = sys.argv[1]
    para_file = sys.argv[2]
    rel_file = sys.argv[3]

    device = 'cuda:0'
    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)
    model.eval()

    queries, corpus, relevant = {}, {}, {}
    for qid, query in load_tsv(query_file):
        queries[qid] = query
    for pid, para in load_tsv(para_file):
        corpus[pid] = para
    for qid, pids in load_tsv(rel_file):
        pids = pids.split(',')
        relevant[qid] = set(pids)

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, 
        batch_size=32, corpus_chunk_size=1000,
        mrr_at_k=[50], ndcg_at_k=[50], accuracy_at_k=[1, 10, 50], precision_recall_at_k=[1, 10, 50], map_at_k=[50],  
        show_progress_bar=True, write_csv=False)
    with torch.no_grad():
        res = evaluator.compute_metrices(model)
    print(res)
