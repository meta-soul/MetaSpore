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
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType

sys.path.append('../')
from common.ml_sparse_features_extractor import generate_sparse_features_25m

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', 'ml_25m/python.zip', 'common'], cwd='../')
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
        .config("spark.ui.showConsoleProgress", "true") 
        .config("spark.submit.pyFiles", "python.zip")
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
    ### read movies
    movies_schema = StructType([
            StructField("movie_id", LongType(), True),
            StructField("title", StringType(), True),
            StructField("genre", StringType(), True)
    ])

    movies = spark.read.csv(movies_path, sep=',',inferSchema=False, header=True, schema=movies_schema)
    print('Debug -- movies sample:')
    movies.show(10)

    ### read ratings
    ratings_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("movie_id", LongType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", LongType(), True)
    ])

    ratings = spark.read.csv(ratings_path, sep=',', inferSchema=False, header=True, schema=ratings_schema)
    print('Debug -- ratings sample:')
    ratings.show(10)
    
    ### read genome
    genome_schema = StructType([
            StructField("movie_id", LongType(), True),
            StructField("tag_id", LongType(), True),
            StructField("relevance", FloatType(), True)
    ])

    genomes = spark.read.csv(genome_path, sep=',', inferSchema=False, header=True, schema=genome_schema)
    print('Debug -- genome sample:')
    genomes.show(10)
    
    ### read links
    links_schema = StructType([
            StructField("movie_id", LongType(), True),
            StructField("imdb_id", StringType(), True),
            StructField("tmdb_id", LongType(), True)
    ])
    
    links = spark.read.csv(links_path, sep=',', inferSchema=False, header=True, schema=links_schema)
    print('Debug -- links sample:')
    links.show(10)

    return movies, ratings, genomes, links

def merge_dataset(movies, ratings):
    dataset = ratings
    dataset = dataset.join(movies, on=dataset.movie_id==movies.movie_id, how='leftouter').drop(movies.movie_id)
    dataset = dataset.select('user_id', \
                             'movie_id', \
                             'title', \
                             'genre', \
                             'rating', \
                             'timestamp')
    print('Debug -- dataset sample:')
    dataset.show(10)
    return dataset

def downsample_user_ratings(spark, dataset):
    if max_reservation_rating_len <= 0:
        print('Debug -- max_reservation_len is 0, there is no downsampling user ratings.')
        return dataset
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
    where ta.sample_id <=%d
    order by user_id ASC
    """ % (max_reservation_rating_len)
    dataset_ = spark.sql(query)
    return dataset_.drop('sample_id')

def write_dataset_to_s3(fg_dataset):
    start = time.time()
    fg_dataset.write.parquet(fg_datast_out_path, mode="overwrite")
    print('Debug -- write_fg_dataset_to_s3 cost time:', time.time() - start)

if __name__=="__main__":
    print('Debug -- Movielens 25M Feature Genearation')
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

    ## preprocessing
    movies, ratings, genomes, links = read_dataset(**params)
    merged_dataset = merge_dataset(movies, ratings)
    merged_dataset = downsample_user_ratings(spark, merged_dataset)

    ## generate sparse features
    fg_dataset = generate_sparse_features_25m(merged_dataset, verbose=verbose)

    ## write to s3
    write_dataset_to_s3(fg_dataset)
       
    stop_spark(spark)
