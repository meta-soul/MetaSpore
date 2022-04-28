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
import subprocess

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

sys.path.append('../') 
from common.neg_sampler import negative_sampling

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(app_name, executor_memory, executor_instances, executor_cores, 
               default_parallelism, **kwargs):
    subprocess.run(['zip', '-r', 'ml_1m/python.zip', 'common'], cwd='../')
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.default.parallelism", default_parallelism)
        .config("spark.executor.memoryOverhead", "2G")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .config("spark.submit.pyFiles", "python.zip")
        .config("spark.network.timeout","500")
        .config("spark.ui.showConsoleProgress", "true")
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

def read_dataset(fg_dataset_path, **kwargs):
    fg_dataset = spark.read.parquet(fg_dataset_path)
    print('Debug -- read dataset sample:')
    fg_dataset.show(10)
    return fg_dataset

def prepare_train(spark, train_dataset, verbose=True):
    train_dataset = train_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    train_dataset = train_dataset.drop('rand')
    train_dataset = train_dataset.select(*(F.col(c).cast('string').alias(c) for c in train_dataset.columns))
    if verbose:
        print('Debug -- match train dataset size: %d'%train_dataset.count())
        print('Debug -- match train types:', train_dataset.dtypes)
        print('Debug -- match train dataset sample:')
        train_dataset.show(10)
    return train_dataset

def prepare_test(spark, test_fg_dataset, verbose=True):
    test_dataset = test_fg_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    test_dataset = test_dataset.drop('rand', 'timestamp')
    test_dataset = test_dataset.select(*(F.col(c).cast('string').alias(c) for c in test_dataset.columns))
    if verbose:
        print('Debug -- match test dataset size: %d'%test_dataset.count())
        print('Debug -- match test types:', test_dataset.dtypes)
        print('Debug -- match test dataset sample:')
        test_dataset.show(10)
    return test_dataset

def split_train_test(dataset):
    # treat low ratings as negative ones
    dataset = dataset.withColumn('label',  F.when(F.col('rating')> 0, 1).otherwise(0))
    # split train and test
    dataset.registerTempTable('dataset')        
    query ="""
    select 
        *
    from
    (
        select
            *,
            ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY timestamp DESC) as sample_id
        from
            dataset
    ) ta
    where ta.sample_id <= 3
    order by user_id ASC
    """
    test_dataset = spark.sql(query)
    test_dataset = test_dataset.drop('sample_id')
    train_dataset = dataset.exceptAll(test_dataset)
    return train_dataset, test_dataset

def write_dataset_to_s3(match_train_dataset, match_test_dataset, 
                        match_train_dataset_out_path, match_test_dataset_out_path, **kwargs):
    start = time.time()
    match_train_dataset.write.parquet(match_train_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 train cost time:', time.time() - start)
    start = time.time()
    match_test_dataset.write.parquet(match_test_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 test cost time:', time.time() - start)
    start = time.time()
    return True

if __name__=="__main__":
    print('Debug -- MovieLens 1M Match Dataset')
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

    ## read datasets
    fg_dataset = read_dataset(**params)

    ## split train and test
    train_fg_dataset, test_fg_dataset = split_train_test(fg_dataset)
    
    ## for collaborative filtering model
    train_dataset = prepare_train(spark, train_fg_dataset, verbose)
    test_dataset = prepare_test(spark, test_fg_dataset, verbose)

    ## write to s3
    write_dataset_to_s3(train_dataset, test_dataset, **params)
    
    stop_spark(spark)
