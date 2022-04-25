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

def init_spark():
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

def read_dataset(**kwargs):
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
                             'timestamp')
    print('Debug -- dataset sample:')
    dataset.show(10)
    return dataset

# split train, test
def split_train_test(dataset):
    dataset.registerTempTable('dataset')        
    query ="""
    select 
        label, user_id, gender, age, occupation, zip, movie_id, recent_movie_ids, genre, rating, last_movie, last_genre, timestamp  
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
    train_dataset = dataset.exceptAll(test_dataset)
    return train_dataset, test_dataset

def prepare_train(spark, train_fg_dataset, train_neg_sample, verbose=True):
    train_fg_dataset = train_fg_dataset.drop('timestamp')
    train_dataset = train_fg_dataset.union(train_neg_sample)
    train_dataset = train_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    train_dataset = train_dataset.drop('rand', 'rating')
    train_dataset = train_dataset.select(*(F.col(c).cast('string').alias(c) for c in train_dataset.columns))
    if verbose:
        print('Debug -- match train dataset size: %d'%train_dataset.count())
        print('Debug -- match train types:', train_dataset.dtypes)
        print('Debug -- match train dataset sample:')
        train_dataset.show(10)
    return train_dataset

def prepare_test(spark, test_fg_dataset, verbose=True):
    test_dataset = test_fg_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    test_dataset = test_dataset.drop('rand', 'timestamp', 'rating')
    test_dataset = test_dataset.select(*(F.col(c).cast('string').alias(c) for c in test_dataset.columns))
    if verbose:
        print('Debug -- match test dataset size: %d'%test_dataset.count())
        print('Debug -- match test types:', test_dataset.dtypes)
        print('Debug -- match test dataset sample:')
        test_dataset.show(10)
    return test_dataset

def prepare_item(spark, train_fg_dataset, test_fg_dataset, verbose=True):
    temp_table = train_fg_dataset.where(train_fg_dataset['label'] == '1').union(test_fg_dataset).distinct()
    temp_table.registerTempTable('temp_table')        
    query = """
    select
        label, user_id, gender, age, occupation, zip, movie_id, recent_movie_ids, genre, last_movie, last_genre
    from
    (
        select
            *,
            ROW_NUMBER() OVER(PARTITION BY movie_id ORDER BY recent_movie_ids DESC) as sample_id
        from
            temp_table
    ) ta
    where 
        sample_id=1
    """
    item_dataset = spark.sql(query)
    item_dataset = item_dataset.select(*(F.col(c).cast('string').alias(c) for c in item_dataset.columns))
    if verbose:
        print('Debug -- match item dataset size: %d'%item_dataset.count())
        print('Debug -- match item types:', item_dataset.dtypes)
        print('Debug -- match item dataset sample:')
        item_dataset.show(10)
    return item_dataset

def write_dataset_to_s3(fg_dataset):
    start = time.time()
    fg_dataset.write.parquet(fg_datast_out_path, mode="overwrite")
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
    locals().update(params)
    spark = init_spark()

    ## preprocessing
    users, movies, ratings, imdb = read_dataset(**params)
    merged_dataset = merge_dataset(users, movies, ratings)

    ## generate sparse features
    fg_dataset = generate_sparse_features_1m(merged_dataset, verbose=verbose)

    ## write to s3
    write_dataset_to_s3(fg_dataset)
    
    stop_spark(spark)
