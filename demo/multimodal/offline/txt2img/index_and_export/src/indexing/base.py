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
import glob
import json
import argparse

import torch
from tqdm import tqdm


def load_jsonline(file, encoding='utf8'):
    with open(file, 'r', encoding=encoding) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_index_data(index_file, with_shard=True, return_doc="all"):
    assert return_doc in ['all', 'doc', 'index']
    index_file_list = []
    if not with_shard:
        index_file_list = [index_file]
    else:
        index_file_list = [f for f in glob.glob(f'{index_file}.shard.*')]

    for index_path in index_file_list:
        for item in load_jsonline(index_path):
            if 'id' in item:
                item['id'] = int(item['id'])
            if return_doc == "doc":
                doc_item = {k:v for k, v in item.items() if not k.endswith('_emb')}
                yield doc_item
            elif return_doc == "index":
                idx_item = {k:v for k, v in item.items() if k.endswith('_emb') or k in ['id']}
                yield idx_item
            else:
                yield item

class Builder(object):

    def __init__(self, index_key, emb_key, values_key, shard_size):
        self.index_key = index_key
        self.emb_key = emb_key
        self.values_key = values_key
        self.shard_size = shard_size

    def dump_shard(self, index_file, shard_n, doc_list):
        shard_file = f'{index_file}.shard.{shard_n}'
        with open(shard_file, 'w', encoding='utf8') as fout:
            for doc in doc_list:
                fout.write('{}\n'.format(json.dumps(doc, ensure_ascii=False)))
        return shard_file

    def load(self, file, fmt='jsonline', **kwargs):
        assert fmt in ['jsonline'], f'not supported doc format: {fmt}'
        if fmt == 'jsonline':
            for item in load_jsonline(file, **kwargs):
                yield item

    def build(self, model, data_iter, index_file, encode_kwargs={}):
        shard_n = 0
        shard_doc, shard_index = [], []
        for item in tqdm(data_iter):

            shard_index.append(item[self.index_key])
            shard_doc.append({k:item.get(k, "") for k in self.values_key})

            if len(shard_index) == self.shard_size:
                index_embs = self.encode(model, shard_index, **encode_kwargs)
                for i, embs in enumerate(index_embs):
                    shard_doc[i][self.emb_key] = embs.tolist()
                    del embs
                shard_file = self.dump_shard(index_file, shard_n, shard_doc)
                print("dump shard file {} with {} size".format(shard_file, len(shard_doc)))
                shard_n += 1
                shard_doc, shard_index = [], []

        if shard_index:
            index_embs = self.encode(model, shard_index, **encode_kwargs)
            for i, embs in enumerate(index_embs):
                shard_doc[i][self.emb_key] = embs.tolist()
                del embs
            shard_file = self.dump_shard(index_file, shard_n, shard_doc)
            print("dump shard file {} with {} size".format(shard_file, len(shard_doc)))
            shard_n += 1
            shard_doc, shard_index = [], []

        print("Total {} shards be dumped in the {} index".format(shard_n, index_file))

    def encode(self, model, docs, **kwargs):
        """Return the embeddings of docs"""
        raise NotImplementedError


def get_builder_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="DMetaSoul/sbert-chinese-qmc-domain-v1", help="the encoder model"
    )
    parser.add_argument(
        "--device", default="cpu"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="The batch size of encoding"
    )
    parser.add_argument(
        "--doc-file", type=str, required=True, help="The doc file, json-line format"
    )
    parser.add_argument(
        "--index-file", type=str, required=True, help="The index file, json-line format"
    )
    parser.add_argument(
        "--shard-size", type=int, default=102400, help="The size of each shard"
    )
    parser.add_argument(
        "--doc-key-index", type=str, default="question:question_emb", help="The index:output field of doc"
    )
    parser.add_argument(
        "--doc-key-values", type=str, default="id,question,answer,category", help="The value fields of doc"
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="The number of threads for torch be used"
    )

    return parser


