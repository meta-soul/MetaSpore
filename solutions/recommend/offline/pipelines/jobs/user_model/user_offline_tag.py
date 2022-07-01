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

if __package__ is None:
    sys.path.append('..')
    from common import init_spark, stop_spark
else:
    from jobs.common import init_spark, stop_spark

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
        '--user-tags-topk',
        type=int,
        default=10
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

def run(action_data, item_data, dump_data, scene_id, action_type, action_value_min=0.0, action_value_max=float('inf'), action_sortby_key='action_time', action_max_len=30, user_tags_topk=10, write_mode='overwrite', job_name='user-model-tag', spark_conf=''):
    spark = init_spark(job_name, spark_conf)

    # load action data
    actions = spark.read.parquet(action_data)
    print(f"Total actions: {actions.count()}")

    # load item data
    items = spark.read.parquet(item_data)
    print(f"Total items: {items.count()}")

    # filter scene&type
    actions = actions\
        .filter(actions['scene_id']==scene_id)\
        .filter(actions['action_type']==action_type)\
        .filter(actions['action_value']>action_value_min)\
        .filter(actions['action_value']<action_value_max)
    print(f"Total actions: {actions.count()}")

    # keep user's latest top-k actions
    actions = actions.withColumn('_action_rank', 
            F.row_number().over(Window.partitionBy('user_id').orderBy(F.desc(action_sortby_key))))
    actions = actions.filter(actions['_action_rank']<=action_max_len)
    print(f"Latest actions: {actions.count()}")

    # user tag count
    user_tags = actions\
        .join(items, on=actions.item_id==items.item_id, how='leftouter')\
        .drop(items.item_id)\
        .select('user_id', F.explode('tags').alias("tag"))\
        .groupBy('user_id', 'tag')\
        .count()
    user_tags = user_tags.withColumn('_tag_rank',
            F.row_number().over(Window.partitionBy('user_id').orderBy(F.desc('count'))))
    user_tags = user_tags.filter(user_tags['_tag_rank']<=user_tags_topk)

    # user schema
    users = user_tags\
        .groupBy('user_id')\
        .agg(F.collect_list('tag').alias('tags'), F.collect_list('count').alias('tag_weights'))
    print(f"User dump: {users.count()}")

    # dump users
    users.write.parquet(dump_data, mode=write_mode)

    stop_spark(spark)

def main():
    parser = get_parser()
    args = parser.parse_args()
    run(args.action_data, args.item_data, args.dump_data, args.scene_id, 
        args.action_type, args.action_value_min, args.action_value_max, args.action_sortby_key,
        args.action_max_len, args.user_tags_topk, args.write_mode, args.job_name, args.spark_conf)

if __name__ == '__main__':
    main()
