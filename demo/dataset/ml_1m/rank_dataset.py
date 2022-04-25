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

import yaml
import time
import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.default.parallelism", default_parallelism)
        .config("spark.executor.memoryOverhead", "2G")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .config("spark.network.timeout","500")
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

def read_dataset(**kwargs):
    fg_dataset = spark.read.parquet(fg_datset_path)
    print('Debug -- fg dataset sample:')
    fg_dataset.show(10)
    return fg_dataset

def split_train_test(dataset):
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
    where ta.sample_id = 1
    order by user_id ASC
    """
    test_dataset = spark.sql(query)
    test_dataset = test_dataset.drop('sample_id')
    train_dataset = dataset.exceptAll(test_dataset)
    return train_dataset, test_dataset

def prepare_rank_train(spark, dataset, verbose=True, mode='train'):
    start = time.time()
    dataset = dataset.filter(dataset['rating'] != 3)
    dataset = dataset.withColumn('label',  F.when(F.col('rating')> 3, 1).otherwise(0))
    dataset = dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    dataset = dataset.drop('rand', 'timestamp', 'rating')
    dataset = dataset.select(*(F.col(c).cast('string').alias(c) for c in dataset.columns))
    print('Debug -- prepare_rank_train cost time:', time.time() - start)
    if verbose:
        print('Debug -- rank %s sample size:'% mode, dataset.count())
        print('Debug -- rank %s data types:'% mode, dataset.dtypes)
        print('Debug -- rank %s sample:'% mode)
        dataset.show(10)
        print('Debug -- prepare_rank_train total cost time:', time.time() - start)
    return dataset

def prepare_rank_test(spark, dataset, verbose=True):
    return prepare_rank_train(spark, dataset, verbose=verbose, mode='test')

def write_dataset_to_s3(rank_train_dataset, rank_test_dataset):
    start = time.time()
    rank_train_dataset.write.parquet(rank_train_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 train cost time:', time.time() - start)
    start = time.time()
    rank_test_dataset.write.parquet(rank_test_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 test cost time:', time.time() - start)
    return True

if __name__=="__main__":
    print('Debug -- Movielens 1M Rank Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=True, help='verbose')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='not verbose')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf, 'verbose:', args.verbose)
    ## init spark
    verbose = args.verbose
    params = load_config(args.conf)
    locals().update(params)
    spark = init_spark()

    ## read datasets
    fg_dataset = read_dataset(**params)

    ## split train and test
    train_fg_dataset, test_fg_dataset = split_train_test(fg_dataset)
   
    # for rank model
    # Ref: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    rank_train_dataset = prepare_rank_train(spark, train_fg_dataset, verbose)
    rank_test_dataset = prepare_rank_test(spark, test_fg_dataset, verbose)

    # write to s3
    write_dataset_to_s3(rank_train_dataset, rank_test_dataset)
    
    stop_spark(spark)
