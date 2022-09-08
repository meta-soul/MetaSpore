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
import yaml
import time
import argparse
import subprocess
import metaspore as ms

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType


def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(local, app_name, batch_size, worker_count, server_count,
               worker_memory, server_memory, coordinator_memory, **kwargs):
    subprocess.run(['zip', '-r', os.getcwd() + '/python.zip', 'python'], cwd='../../../')
    spark_confs = {
        "spark.network.timeout":"500",
        "spark.submit.pyFiles":"python.zip",
        "spark.ui.showConsoleProgress": "true",
        "spark.kubernetes.executor.deleteOnTermination":"true",
    }
    spark = ms.spark.get_session(
        local=local,
        app_name=app_name,
        batch_size=batch_size,
        worker_count=worker_count,
        server_count=server_count,
        worker_memory=worker_memory,
        server_memory=server_memory,
        coordinator_memory=coordinator_memory,
        spark_confs=spark_confs)
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark

def read_dataset(spark, user_path, item_path, interaction_path, fmt='parquet', **kwargs):
    user_dataset = spark.read.parquet(user_path)
    print('Debug -- user dataset count:', user_dataset.count())
    item_dataset = spark.read.parquet(item_path)
    print('Debug -- item dataset count:', item_dataset.count())
    interaction_dataset = spark.read.parquet(interaction_path)
    print('Debug -- interaction dataset count:', interaction_dataset.count())
    return user_dataset, item_dataset, interaction_dataset

def join_dataset(spark, user_dataset, item_dataset, interaction_dataset):
    user_dataset.registerTempTable('user_dataset')
    item_dataset.registerTempTable('item_dataset')
    interaction_dataset.registerTempTable('interaction_dataset')
    query ="""
    select distinct
        user.*, item.*, interact.timestamp
    from
        interaction_dataset interact
    join
        user_dataset user
    on interact.user_id=user.user_id
    join
        item_dataset item
    on interact.item_id=item.item_id
    """
    join_dataset = spark.sql(query)
    return join_dataset

def reserve_cate_features(spark, join_dataset, array_join_sep=u'\u0001'):
    str_cols = [f.name for f in join_dataset.schema.fields if isinstance(f.dataType, StringType)]
    array_cols = [f.name for f in join_dataset.schema.fields if isinstance(f.dataType, ArrayType)]
    for array_col in array_cols:
        join_dataset = join_dataset.withColumn(array_col, F.concat_ws(array_join_sep, F.col(array_col)))
    selected_cols = str_cols + array_cols
    filter_dataset = join_dataset.select(selected_cols)
    return filter_dataset

if __name__=="__main__":
    print('Debug -- Ecommerce Samples Preprocessing')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf)
    params = load_config(args.conf)
    
    spark_params = params['spark']
    spark = init_spark(**spark_params)
    
    dataset_params = params['dataset']
    user_dataset, item_dataset, interaction_dataset \
        = read_dataset(spark, **dataset_params)
    join_dataset = join_dataset(
        spark, 
        user_dataset, 
        item_dataset, 
        interaction_dataset
    )
    print('Debug -- join dataset sample:')
    join_dataset.show(10)
    
    join_dataset = reserve_cate_features(spark, join_dataset)
    print('Debug -- reserve cate features sample:')
    join_dataset.show(10)