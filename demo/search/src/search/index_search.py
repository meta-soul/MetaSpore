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

# adapted from: https://github.com/PaddlePaddle/RocketQA/blob/main/research/DuReader-Retrieval-Baseline/src/index_search.py
import sys
import time
import math

import mkl  # in my computer, must import mkl first to use faiss
import faiss
import numpy as np
from tqdm import tqdm

def read_embed(file_name, dim=768, bs=3000):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        while(i < len(emb_np)):
            vec_list = emb_np[i:i+bs]
            i += bs
            yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in inp:
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list

def load_qid(file_name):
    qid_list = []
    with open(file_name) as inp:
        for line in inp:
            line = line.strip()
            qid = line.split('\t')[0]
            qid_list.append(qid)
    return qid_list

def search(q_embs, qid_list, index, outfile, p_shift_i, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in tqdm(q_embs):
            q_emb_matrix = np.array(batch_vec)
            res_dist, res_p_id = index.search(q_emb_matrix.astype('float32'), top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    rank = j+1
                    pid = res_p_id[i][j] + p_shift_i  # index from local part to gloabl collection
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, rank, score))
                q_idx += 1

def main():
    q_text_file = sys.argv[1]
    q_emb_file = sys.argv[2]
    p_shift_i = int(sys.argv[3])
    p_index_file = sys.argv[4]
    outfile = sys.argv[5]
    topk = int(sys.argv[6])
    batch_size = int(sys.argv[7])

    qid_list = load_qid(q_text_file)
    q_embs = read_embed(q_emb_file, bs=batch_size)
    engine = faiss.read_index(p_index_file)
    search(q_embs, qid_list, engine, outfile, p_shift_i, topk)
 

if __name__ == "__main__":
    main()

