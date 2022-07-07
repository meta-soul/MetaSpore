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

import torch
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, IntegerType, FloatType
from sentence_transformers import SentenceTransformer


if __package__ is None:
    # run as a script
    sys.path.append('..')
    from common import init_spark
    from common import request_metaspore_serving, make_metaspore_serving_payload
else:
    from ..common import init_spark
    from ..common import request_metaspore_serving, make_metaspore_serving_payload


def item_emb_by_local(spark, item_df, model_name, batch_size, device):
    item_ids = item_df.select('item_id').rdd.flatMap(lambda x: x).collect()
    item_texts = item_df.select('content').rdd.flatMap(lambda x: x).collect()
    # load model
    model = SentenceTransformer(model_name, device=device)
    # encode item
    item_embs = model.encode(item_texts, batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # to df
    items = []
    for i, (item_id, item_emb) in enumerate(zip(item_ids, item_embs)):
        items.append({'id': i, 'item_id': item_id, 'item_emb': item_emb.tolist()})
    emb_df = spark.createDataFrame(items)
    return emb_df

def item_emb_by_spark(spark, item_df, model_name, batch_size, device):
    use_cuda = "cuda" in device and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_state = SentenceTransformer(model_name, device=device).eval()
    bc_model_state = spark.sparkContext.broadcast(model_state.state_dict())

    def get_model_for_eval():
        model = SentenceTransformer(model_name, device=device)
        model.load_state_dict(bc_model_state.value)
        model.eval()
        return model

    def item_emb(it):
        model = get_model_for_eval()
        for df in it:
            item_ids = df['item_id'].tolist()
            item_texts = df['content'].tolist()
            item_embs = model.encode(item_texts, batch_size=batch_size,
                show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            items = []
            for item_id, item_emb in zip(item_ids, item_embs):
                items.append({'item_id': item_id, 'item_emb': item_emb.tolist()})
            yield pd.DataFrame(items)
    schema = StructType([
        StructField("item_id", StringType()),
        StructField("item_emb", ArrayType(FloatType()))
    ])
    item_embs = item_df.mapInPandas(item_emb, schema)
    #item_embs.show()
    emb_df = item_embs.withColumn("id", F.monotonically_increasing_id())
    return emb_df

def item_emb_by_serving(spark, item_df, serving_host, serving_port, serving_model):
    def request_serving(x):
        item_id = x['item_id']
        texts = [x['content']]
        payload = make_metaspore_serving_payload(texts, 'text')
        res = request_metaspore_serving(serving_host, serving_port, serving_model, payload)
        item_emb = [float(v) for v in res['sentence_embedding'].flatten().tolist()]
        return (item_id, item_emb)
    item_embs = item_df.rdd.map(lambda x: request_serving(x)).toDF(['item_id', 'item_emb'])
    emb_df = item_embs.withColumn("id", F.monotonically_increasing_id())
    return emb_df

def run(item_data, dump_data, text_model_name, batch_size=256, device="cuda:0", 
        serving=False, serving_host="", serving_port=50000, serving_model="",
        write_mode="overwrite", job_name='item-model-embed', spark_conf=""):
    spark = init_spark(job_name, spark_conf)

    # load raw data
    item_df = spark.read.parquet(item_data)

    if not serving:
        #emb_df = item_emb_by_local(spark, item_df, text_model_name, batch_size, device)
        emb_df = item_emb_by_spark(spark, item_df, text_model_name, batch_size, device)
    else:
        emb_df = item_emb_by_serving(spark, item_df, serving_host, serving_port, serving_model)

    # dump data
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
