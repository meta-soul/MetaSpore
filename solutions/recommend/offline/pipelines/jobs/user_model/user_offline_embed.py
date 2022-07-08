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
import math
import json
import pickle
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

if __package__ is None:
    sys.path.append('..')
    from common import init_spark, stop_spark
else:
    from ..common import init_spark, stop_spark

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--scene-id',
        type=str,
        required=True
    )
    parser.add_argument(
        '--action-type',
        type=str,
        required=True
    )
    parser.add_argument(
        '--action-data',
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
        '--user-data',
        type=str,
        default=''
    )
    parser.add_argument(
        '--action-value-min',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--action-value-max',
        type=float,
        default=float('inf')
    )
    parser.add_argument(
        '--action-sortby-key',
        type=str,
        choices=['action_time', 'action_value'],
        default='action_time'
    )
    parser.add_argument(
        '--action-max-len',
        type=int,
        default=30
    )
    parser.add_argument(
        '--action-agg-func',
        type=str,
        choices=['latest', 'avg', 'decay'],
        default='avg'
    )
    parser.add_argument(
        '--action-decay-rate',
        type=float,
        default=0.9
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

def run(action_data, item_data, dump_data, scene_id, action_type, 
        action_value_min=0.0, action_value_max=float('inf'), action_sortby_key='action_time', 
        action_max_len=30, action_agg_func='avg', action_decay_rate=0.9, write_mode='overwrite', 
        job_name='user-model-embed', spark_conf='', spark_local=False):
    spark = init_spark(job_name, spark_conf, local=spark_local)

    # load action data
    actions = spark.read.parquet(action_data)
    print(f"Total actions: {actions.count()}")

    # load item data
    items = spark.read.parquet(item_data)
    print(f"Total items: {items.count()}")

    # load user history emb
    #user_hist = None
    #if user_data:
    #    user_hist = spark.read.parquet(user_data)
    #    print(f"User history: {user_hist.count()}")

    # filter scene&type
    actions = actions\
        .filter(actions['scene_id']==scene_id)\
        .filter(actions['action_type']==action_type)\
        .filter(actions['action_value']>action_value_min)\
        .filter(actions['action_value']<action_value_max)
    print(f"Scene actions: {actions.count()}")

    # keep user's latest top-k actions
    actions = actions.withColumn('_action_rank', 
            F.row_number().over(Window.partitionBy('user_id').orderBy(F.desc(action_sortby_key))))
    actions = actions.filter(actions['_action_rank']<=action_max_len)
    print(f"Latest actions: {actions.count()}")

    # user action items
    user_items = actions\
        .join(items, on=actions.item_id==items.item_id, how='leftouter')\
        .drop(items.item_id)\
        .select('user_id', 'item_id', 'item_emb', '_action_rank')\
        .withColumnRenamed('_action_rank', 'item_rank')

    # user embeding
    emb_dim = 0
    if items.count() > 0:
        emb_dim = len(items.head(1)[0].item_emb)
    action_decay_rate = action_decay_rate
    if action_agg_func == 'latest':
        user_emb = user_items.filter(user_items['item_rank']==1)\
            .groupBy('user_id')\
            .agg(F.first('item_emb'))\
            .withColumnRenamed('first(item_emb)', 'user_emb')
    elif action_agg_func == 'avg':
        array_avg = F.udf(lambda x: np.mean(x, axis=0).tolist(), ArrayType(FloatType()))
        user_items = user_items.groupBy('user_id')\
            .agg(F.collect_list('item_emb').alias('item_embs'))\
            .select('user_id', 'item_embs')
        user_emb = user_items.select('user_id', array_avg('item_embs').alias('user_emb'))
    else:
        @F.udf(returnType=ArrayType(FloatType()))
        def decay_avg(embs, ranks):
            emb = np.zeros(emb_dim, dtype=np.float32)
            if len(embs) == 0:
                return emb.tolist()
            elif len(embs) == 1 and embs[0] is not None:
                return embs[0].tolist()
            else:
                indices = sorted(list(range(len(ranks))), key=lambda x:ranks[x], reverse=True)  # from old to new
                for i in indices:
                    if i >= len(embs) or embs[i] is None:
                        continue
                    emb = action_decay_rate*emb + (1.0-action_decay_rate)*np.array(embs[i])
                return emb.tolist()
        user_items = user_items.groupBy('user_id')\
            .agg(F.collect_list('item_rank').alias('item_ranks'), F.collect_list('item_emb').alias('item_embs'))\
            .select('user_id', 'item_embs', 'item_ranks')
        user_emb = user_items.select('user_id', decay_avg('item_embs', 'item_ranks').alias('user_emb'))

    # dump users
    user_emb.write.parquet(dump_data, mode=write_mode)

    stop_spark(spark)

def main():
    parser = get_parser()
    args = parser.parse_args()
    run(args.action_data, args.item_data, args.dump_data, 
        args.scene_id, args.action_type, args.action_value_min, args.action_value_max,
        args.action_sortby_key, args.action_max_len, args.action_agg_func, 
        args.action_decay_rate, args.write_mode, args.job_name, args.spark_conf)

if __name__ == '__main__':
    main()
