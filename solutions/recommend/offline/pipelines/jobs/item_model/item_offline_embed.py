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
import json
import pickle
import argparse

from sentence_transformers import SentenceTransformer

if __package__ is None:
    # run as a script
    sys.path.append('..')
    from common import init_spark
else:
    from ..common import init_spark

def run(item_data, dump_data, text_model_name, 
        batch_size=256, device="cuda:0", write_mode="overwrite", 
        job_name='item-model-embed', spark_conf=""):
    spark = init_spark(job_name, spark_conf)

    # load raw data
    item_df = spark.read.parquet(item_data)
    item_ids = item_df.select('item_id').rdd.flatMap(lambda x: x).collect()
    item_texts = item_df.select('content').rdd.flatMap(lambda x: x).collect()

    # load model
    model = SentenceTransformer(text_model_name, device=device)

    # encode item
    item_embs = model.encode(item_texts, batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # dump data
    items = []
    for i, (item_id, item_emb) in enumerate(zip(item_ids, item_embs)):
        items.append({'id': i, 'item_id': item_id, 'item_emb': item_emb.tolist()})
    emb_df = spark.createDataFrame(items)
    emb_df.write.parquet(dump_data, mode=write_mode)

    spark.sparkContext.stop()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--item-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dump-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--text-model-name',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0'
    )
    parser.add_argument(
        '--spark-conf',
        type=str,
        default='spark.executor.memory:10G,spark.executor.instances:4,spark.executor.cores:4'
    )
    parser.add_argument(
        '--write-mode',
        type=str,
        choices=['overwrite', 'append'],
        default='overwrite'
    )
    #return parser.parse_args()
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run(args.item_data, args.dump_data, args.text_model_name, 
        args.batch_size, args.device, args.write_mode, args.job_name, args.spark_conf)

if __name__ == '__main__':
    main()
