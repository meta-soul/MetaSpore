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

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

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

def read_dataset(fg_dataset_path, **kwargs):
    fg_dataset = spark.read.parquet(fg_dataset_path)
    print('Debug -- fg dataset sample:')
    fg_dataset.show(10)
    return fg_dataset

def sample(row, movie_list, sep=u'\u0001'):
    recent_movie_ids, recent_movie_genres, recent_movie_years = row[3], row[4], row[5]
    movie_indices = list(range(0, len(movie_list)))

    s_indices = np.random.default_rng().choice(list(movie_indices),
            size=len(recent_movie_ids.split(sep)), replace=True).tolist()
    neg_movie_ids = sep.join([str(movie_list[0][i]) for i in s_indices])

    s_indices = np.random.default_rng().choice(list(movie_indices), 
            size=len(recent_movie_genres), replace=True).tolist()
    neg_movie_genres = sep.join([str(movie_list[1][i]) for i in s_indices])
    
    s_indices = np.random.default_rng().choice(list(movie_indices),
            size=len(recent_movie_years), replace=True).tolist()
    neg_movie_years = sep.join([str(movie_list[2][i]) for i in s_indices])

    return row[:3] + (neg_movie_ids, neg_movie_genres, neg_movie_years)

def negative_sequence(spark, fg_dataset, sep=u'\u0001'):
    rows = fg_dataset.withColumn('genre', F.regexp_replace('genre', sep, '\|'))\
                     .select('movie_id', 'genre', 'year')\
                     .distinct()\
                     .rdd.map(lambda x: (x[0], x[1], x[2])).collect()
    movie_list_br = spark.sparkContext.broadcast([[x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows]])
    # sample neg seq
    neg_df = fg_dataset.rdd\
                       .map(lambda x:(x['user_id'], x['movie_id'], x['timestamp'], x['recent_movie_ids'], x['recent_movie_genres'], x['recent_movie_years']))\
                       .map(lambda x:sample(x, movie_list_br.value))\
                       .toDF(['user_id', 'movie_id', 'timestamp', 'neg_movie_ids', 'neg_movie_genres', 'neg_movie_years'])
    # merge neg seq
    merge_df = fg_dataset.alias('t1')\
                         .join(neg_df.alias('t2'), 
                             (F.col('t1.user_id')==F.col('t2.user_id')) & (F.col('t1.timestamp')==F.col('t2.timestamp')) & (F.col('t1.movie_id')==F.col('t2.movie_id')), 
                             how="inner")\
                         .select('t1.*', 't2.neg_movie_ids', 't2.neg_movie_genres', 't2.neg_movie_years')
    return merge_df


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

def write_dataset_to_s3(rank_train_dataset, rank_test_dataset, 
                        rank_train_dataset_out_path, rank_test_dataset_out_path, **kwargs):
    start = time.time()
    rank_train_dataset.write.parquet(rank_train_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 train cost time:', time.time() - start)
    start = time.time()
    rank_test_dataset.write.parquet(rank_test_dataset_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 test cost time:', time.time() - start)
    return True

if __name__=="__main__":
    print('Debug -- Movielens 25M Rank Dataset')
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
    fg_dataset = negative_sequence(spark, fg_dataset)

    ## split train and test
    train_fg_dataset, test_fg_dataset = split_train_test(fg_dataset)
   
    # for rank model
    # Ref: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    rank_train_dataset = prepare_rank_train(spark, train_fg_dataset, verbose)
    rank_test_dataset = prepare_rank_test(spark, test_fg_dataset, verbose)

    # write to s3
    write_dataset_to_s3(rank_train_dataset, rank_test_dataset, **params)
    
    stop_spark(spark)
