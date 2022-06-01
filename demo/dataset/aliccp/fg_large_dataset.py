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

import sys
import yaml
import time
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from collections import defaultdict

aliccp_schema = StructType([
    StructField("click", StringType(), True),
    StructField("purchase", StringType(), True),
    StructField("101", StringType(), True),
    StructField("121", StringType(), True),
    StructField("122", StringType(), True),
    StructField("124", StringType(), True),
    StructField("125", StringType(), True),
    StructField("126", StringType(), True),
    StructField("127", StringType(), True),
    StructField("128", StringType(), True),
    StructField("129", StringType(), True),
    StructField("205", StringType(), True),
    StructField("206", StringType(), True),
    StructField("207", StringType(), True),
    StructField("216", StringType(), True),
    StructField("508", StringType(), True),
    StructField("509", StringType(), True),
    StructField("702", StringType(), True),
    StructField("853", StringType(), True),
    StructField("301", StringType(), True)
])

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(app_name, executor_memory, executor_instances, executor_cores, 
               default_parallelism, **kwargs):
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.default.parallelism", default_parallelism)
        .config("spark.executor.memoryOverhead", "2G")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.network.timeout","500")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .getOrCreate())
    
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark

def stop_spark(spark):
    print('Debug -- spark stop')
    spark.sparkContext.stop()

def read_dataset(spark, train_path, test_path, **kwargs):
    train_dataset = spark.read.csv(train_path, sep=',', inferSchema=False, header=True, schema=aliccp_schema)
    test_dataset = spark.read.csv(test_path, sep=',', inferSchema=False, header=True, schema=aliccp_schema)
    return train_dataset, test_dataset

def transform_dataset(train_dataset, test_dataset, verbose=False, **kwargs):
    start = time.time()
    fg_train_dataset = train_dataset.withColumn('label', F.when((F.col('click')=='1')&(F.col('purchase')=='1'), '1').otherwise('0'))
    fg_train_dataset = fg_train_dataset.select('label', 
                                                F.col('click').alias('ctr_label'),
                                                F.col('purchase').alias('cvr_label'),
                                                '101','121','122','124','125','126','127','128','129','205','206','207','216','508','509','702','853','301')

    fg_test_dataset = test_dataset.withColumn('label', F.when((F.col('click')=='1')&(F.col('purchase')=='1'), '1').otherwise('0'))
    fg_test_dataset = fg_test_dataset.select('label', 
                                            F.col('click').alias('ctr_label'),
                                            F.col('purchase').alias('cvr_label'),
                                            '101','121','122','124','125','126','127','128','129','205','206','207','216','508','509','702','853','301')

    print('Debug -- transform_dataset cost time:', time.time() - start)
    if verbose:
        print('Debug -- train dataset count:', fg_train_dataset.count())
        print('Debug -- train dataset sample:')
        fg_train_dataset.show(20)
        print('Debug -- test dataset count:', fg_test_dataset.count())
        print('Debug -- test dataset sample:')
        fg_test_dataset.show(20)
    return fg_train_dataset, fg_test_dataset

def write_fg_dataset_to_s3(fg_train_dataset, fg_test_dataset,  train_output_path, test_output_path, verbose=False, **kwargs):
    start = time.time()
    fg_train_dataset.write.parquet(train_output_path, mode="overwrite")
    fg_test_dataset.write.parquet(test_output_path, mode="overwrite")
    print('Debug -- write_fg_dataset_to_s3 cost time:', time.time() - start)
    if verbose:
        print('Debug -- train dataset count:', fg_train_dataset.count())
        print('Debug -- test dataset count:', fg_test_dataset.count())
    return True

if __name__=="__main__":
    print('Debug -- AliCCP Dataset Feature Generation')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=True, help='verbose')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='not verbose')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf, 'verbose:', args.verbose)
    ## init spark
    verbose = args.verbose
    params = load_config(args.conf)
    spark = init_spark(**params)

    ## preprocessing
    train_dataset, test_dataset = read_dataset(spark, **params)

    ## generate sparse features
    fg_train_dataset, fg_test_dataset = transform_dataset(train_dataset, test_dataset, verbose)

    ## write to s3
    write_fg_dataset_to_s3(fg_train_dataset, fg_test_dataset, **params)
    
    stop_spark(spark)