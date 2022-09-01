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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType

sys.path.append('../')
from common.ml_sparse_features_extractor import generate_sparse_features_1m

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
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.network.timeout","500")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
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

def read_dataset(movies_path, ratings_path, users_path, imdb_path, **kwargs):
    ### read movies
    movies_schema = StructType([
            StructField("movie_id", LongType(), True),
            StructField("title", StringType(), True),
            StructField("genre", StringType(), True)
    ])

    movies = spark.read.csv(movies_path, sep='::',inferSchema=False, header=False, schema=movies_schema)
    print('Debug -- movies sample:')
    movies.show(10)

    ### read ratings
    ratings_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("movie_id", LongType(), True),
            StructField("rating", FloatType(), True),
            StructField("timestamp", LongType(), True)
    ])

    ratings = spark.read.csv(ratings_path, sep='::', inferSchema=False, header=False, schema=ratings_schema)
    print('Debug -- ratings sample:')
    ratings.show(10)

    ### read users
    users_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("gender", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("occupation", StringType(), True),
            StructField("zip", StringType(), True)
    ])

    users = spark.read.csv(users_path, sep='::', inferSchema=False, header=False, schema=users_schema)
    print('Debug -- users sample:')
    users.show(10)

    ### read imdb datasets
    imdb = spark.read.csv(imdb_path, sep=r'\t', inferSchema=False, header=True)
    imdb = imdb.withColumn('imdb_url', F.concat(F.lit("https://www.imdb.com/title/"), F.col("tconst"), F.lit("/")))
    print('Debug -- imdb sample:')
    imdb.show(10)

    return users, movies, ratings, imdb

def merge_dataset(users, movies, ratings):
    # merge movies, users, ratings
    movies = movies.withColumn('year', F.regexp_extract('title', r'(.+)\s*\((\d+)\)', 2))
    dataset = ratings.join(users, on=ratings.user_id==users.user_id, how='leftouter').drop(users.user_id)
    dataset = dataset.join(movies, on=dataset.movie_id==movies.movie_id,how='leftouter').drop(movies.movie_id)
    dataset = dataset.select('user_id', \
                             'gender', \
                             'age', \
                             'occupation', \
                             'zip', \
                             'movie_id', \
                             'title', \
                             'genre', \
                             'rating', \
                             'timestamp',\
                             'year')
    print('Debug -- dataset sample:')
    dataset.show(10)
    return dataset

def write_dataset_to_s3(fg_dataset, fg_dataset_out_path, **kwargs):
    start = time.time()
    fg_dataset.write.parquet(fg_dataset_out_path, mode="overwrite")
    print('Debug -- write_fg_dataset_to_s3 cost time:', time.time() - start)

if __name__=="__main__":
    print('Debug -- Movielens 1M Feature Generation')
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
    users, movies, ratings, imdb = read_dataset(**params)
    merged_dataset = merge_dataset(users, movies, ratings)

    ## generate sparse features
    fg_dataset = generate_sparse_features_1m(merged_dataset, verbose=verbose)

    ## write to s3
    write_dataset_to_s3(fg_dataset, **params)
    
    stop_spark(spark)
