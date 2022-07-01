import os
import sys
import json
import pickle
import argparse

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
        '--item-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--action-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dump-attr-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dump-tag-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--scene-id',
        type=str,
        default=''
    )
    parser.add_argument(
        '--action-type',
        type=str,
        default=''
    )
    parser.add_argument(
        '--action-value-min',
        type=float,
        default=None
    )
    parser.add_argument(
        '--action-value-max',
        type=float,
        default=None
    )
    parser.add_argument(
        '--tag-max-len',
        type=int,
        default=10000
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


def run(item_data, action_data, dump_attr_data, dump_tag_data, scene_id, action_type, 
        action_value_min=None, action_value_max=None, tag_max_len=10000, write_mode='overwrite', 
        job_name='item-model-attr', spark_conf=''):
    spark = init_spark(job_name, spark_conf)

    # load raw data
    items = spark.read.parquet(item_data)
    actions = spark.read.parquet(action_data)

    # filter actions
    if scene_id:
        actions = actions.filter(actions['scene_id']==scene_id)
    if action_type:
        actions = actions.filter(actions['action_type']==action_type)
    if action_value_min is not None:
        actions = actions.filter(actions['action_value']>action_value_min)
    if action_value_max is not None:
        actions = actions.filter(actions['action_value']<action_value_max)

    # item basic attributes
    item_cnts = actions.groupBy('item_id').count().withColumnRenamed('count', 'popularity')
    item_attrs = item_cnts.join(items, 
        on=item_cnts.item_id==items.item_id, 
        how='leftouter').drop(items.item_id)
    item_attrs = item_attrs.select('item_id', 'tags', 'weight', 'popularity')
    item_attrs.write.parquet(dump_attr_data, mode=write_mode)
    print(f"Total items: {item_attrs.count()}")

    # tag2item revert-index
    item_tag = items.select('item_id', F.explode('tags').alias('tag'))
    item_tag = item_tag.join(item_attrs, 
        on=item_tag.item_id==item_attrs.item_id, 
        how='leftouter').drop(item_attrs.item_id).select('item_id', 'tag', 'popularity')
    item_tag = item_tag.withColumn('_item_rank',
        F.row_number().over(Window.partitionBy('tag').orderBy(F.desc('popularity'))))
    item_tag = item_tag.filter(item_tag['_item_rank']<=tag_max_len)
    tag2item = item_tag.groupBy('tag').agg(F.collect_set('item_id').alias('item_ids'))
    tag2item.write.parquet(dump_tag_data, mode=write_mode)
    print(f"Total tags: {tag2item.count()}")

    stop_spark(spark)

def main():
    parser = get_parser()
    args = parser.parse_args()
    run(args.item_data, args.action_data, args.dump_attr_data,
        args.dump_tag_data, args.scene_id, args.action_type, args.action_value_min,
        args.action_value_max, args.tag_max_len, args.write_mode, args.job_name, args.spark_conf)

if __name__ == '__main__':
    main()
