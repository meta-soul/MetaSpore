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

import argparse

import mkl
import faiss
import numpy as np

def faiss_index(embs, emb_dim, index_mode):
    if index_mode == 'FlatL2':
        index = faiss.IndexFlatL2(emb_dim)
        index.add(embs)
        #return index.ntotal
    else:
        #elif index_mode == 'FlatIP':
        index = faiss.IndexFlatIP(emb_dim)
        index.add(embs)
        #return index.ntotal
    return index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--emb-file',
        required=True,
        type=str,
        help='The numpy embeddings file, of shape (data_size, emb_dim).'
    )
    parser.add_argument(
        '--index-file',
        required=True,
        type=str,
        help='The output index file.'
    )
    parser.add_argument(
        '--index-mode',
        type=str,
        default='faiss:FlatIP',
        choices=['faiss:FlatIP', 'faiss:FlatL2', 'faiss:IVFFlat', 'faiss:IVFPQ'],
        help='The embedding index mode.'
    )
    args = parser.parse_args()

    embs = np.load(args.emb_file)
    n, h = embs.shape

    if args.index_mode.startswith('faiss'):
        index_mode = 'FlatIP'
        if ':' in args.index_mode:
            index_mode = args.index_mode.split(':')[1]
        if args.index_file.endswith('.faiss'):
            index_file = args.index_file
        else:
            index_file = args.index_file + '.faiss'
        index = faiss_index(embs, h, index_mode)
        faiss.write_index(index, index_file)
    else:
        print(f"Invalid index mode: {args.index_mode}")
