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
import argparse
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType
from functools import reduce

from fg_gbm_features_extractor import generate_gbm_features
from fg_sparse_features_extractor import generate_sparse_features
from fg_neg_sampler import negative_sampling_train_dataset

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', './python.zip', 'fg_neg_sampler.py', 'fg_sparse_features_extractor.py', 'fg_gbm_features_extractor.py' ], cwd='./')
    spark = (SparkSession.builder
        .appName('Demo -- movielens')
        .config("spark.executor.memory","10G")
        .config("spark.executor.instances","4")
        .config("spark.network.timeout","500")
        # .config("spark.ui.showConsoleProgress", "false") ## close stage log
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
                            'timestamp'
                            )
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

def prepare_match_train(spark, train_fg_dataset, train_neg_sample):
    train_fg_dataset = train_fg_dataset.drop('timestamp')
    train_dataset = train_fg_dataset.union(train_neg_sample)
    train_dataset = train_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    train_dataset = train_dataset.drop('rand', 'rating')
    train_dataset = train_dataset.select(*(F.col(c).cast('string').alias(c) for c in train_dataset.columns))
    print('Debug -- match train dataset size: %d'%train_dataset.count())
    print('Debug -- match train types:', train_dataset.dtypes)
    print('Debug -- match train dataset sample:')
    train_dataset.show(10)
    return train_dataset

def prepare_match_test(spark, test_fg_dataset):
    test_dataset = test_fg_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    test_dataset = test_dataset.drop('rand', 'timestamp', 'rating')
    test_dataset = test_dataset.select(*(F.col(c).cast('string').alias(c) for c in test_dataset.columns))
    print('Debug -- match test dataset size: %d'%test_dataset.count())
    print('Debug -- match test types:', test_dataset.dtypes)
    print('Debug -- match test dataset sample:')
    test_dataset.show(10)
    return test_dataset

def prepare_match_item(spark, train_fg_dataset, test_fg_dataset):
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
    print('Debug -- match item dataset size: %d'%item_dataset.count())
    print('Debug -- match item types:', item_dataset.dtypes)
    print('Debug -- match item dataset sample:')
    item_dataset.show(10)
    return item_dataset

def prepare_user_feaures(spark, dataset, gbm_features):
    user_gbm_features = gbm_features['user']
    dataset.registerTempTable('temp_table') 
    user_query = """
    select distinct
        user_id, gender, age, occupation, zip, recent_movie_ids, last_movie, last_genre
    from
    (
        select
            *,
            ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY timestamp DESC) as sample_id
        from
            temp_table
    )  ta
    where
        sample_id=1
    """
    user_features = spark.sql(user_query)
    user_features = user_features.join(user_gbm_features,on=user_features.user_id==user_gbm_features.user_id,how='leftouter').drop(user_gbm_features.user_id)
    
    print('Debug -- match user features size: %d'%user_features.count())
    print('Debug -- match item features sample:')
    user_features.show(10)
    return user_features

def prepare_item_features(spark, dataset, imdb, gbm_features, sep=u'\u0001'):
    dataset = dataset.withColumn('genre', F.regexp_replace('genre', '\|', sep))
    result = dataset.alias('t1')\
                    .join(imdb.alias('t2'), (F.col('t1.movie_id') == F.col('t2.movieId')), how = 'leftouter')\
                    .select('t1.*', 't2.imdb_url')
    ## merge gbm numeric features
    gbm_movie_features = gbm_features['movie']
    gbm_genre_features = gbm_features['genre']
    result = result.join(gbm_movie_features, on=gbm_movie_features.movie_id==result.movie_id,how='leftouter')\
                    .drop(gbm_movie_features.movie_id)
    result = result.join(gbm_genre_features, on=gbm_genre_features.genre==result.genre,how='leftouter')\
                    .drop(gbm_genre_features.genre)
    ## final result
    item_summary = result.select('movie_id',
                                 'genre',
                                 'title',
                                 'imdb_url')
    item_features = result.select('movie_id',
                                  'watch_volume',
                                  'genre',
                                  'movie_avg_rating',
                                  'movie_greater_than_three_rate',
                                  'genre_watch_volume',
                                  'genre_movie_avg_rating',
                                  'genre_greater_than_three_rate')
    return item_summary, item_features
    
def prepare_rank_train(spark, fg_dataset, mode='train'):
    fg_dataset = fg_dataset.filter(fg_dataset['rating'] != 3)
    fg_dataset = fg_dataset.withColumn('label',  F.when(F.col('rating')> 3, 1).otherwise(0))
    fg_dataset = fg_dataset.withColumn('rand', F.rand(seed=100)).orderBy('rand')
    fg_dataset = fg_dataset.drop('rand', 'timestamp', 'rating')
    fg_dataset = fg_dataset.select(*(F.col(c).cast('string').alias(c) for c in fg_dataset.columns))
    print('Debug -- rank %s sample size:'%mode, fg_dataset.count())
    print('Debug -- rank %s data types:'%mode, fg_dataset.dtypes)
    print('Debug -- rank %s sample:'%mode)
    fg_dataset.show(10)
    return fg_dataset

def prepare_rank_test(spark, fg_dataset):
    return prepare_rank_train(spark, fg_dataset, mode='test')

def prepare_rank_lgbm_train(spark, gbm_features, rank_dataset, mode='train'):  
    user_feaures = gbm_features['user']
    rank_dataset = rank_dataset.join(user_feaures, on=rank_dataset.user_id==user_feaures.user_id, how='leftouter')\
                                .drop(user_feaures.user_id)
    movie_features = gbm_features['movie']
    rank_dataset = rank_dataset.join(movie_features, on=rank_dataset.movie_id==movie_features.movie_id, how='leftouter')\
                                .drop(movie_features.movie_id)
    genre_features = gbm_features['genre']
    rank_dataset = rank_dataset.join(genre_features, on=rank_dataset.genre==genre_features.genre, how='leftouter')\
                                .drop(genre_features.genre)

    rank_dataset = rank_dataset.withColumn("label", rank_dataset["label"].cast('int'))
    rank_dataset = rank_dataset.select('label',
                                       'user_greater_than_three_rate',
                                       'user_movie_avg_rating',
                                       'watch_volume',
                                       'movie_avg_rating',
                                       'movie_greater_than_three_rate',
                                       'genre_watch_volume',
                                       'genre_movie_avg_rating',
                                       'genre_greater_than_three_rate')
    print('Debug -- rank %s sample size:'%mode, rank_dataset.count())
    print('Debug -- rank %s data types:'%mode, rank_dataset.dtypes)
    print('Debug -- rank %s sample:'%mode)
    rank_dataset.show(10)
    return rank_dataset
    
def prepare_rank_lgbm_test(spark, gbm_features, rank_dataset):
    return prepare_rank_lgbm_train(spark, gbm_features, rank_dataset, 'test')

if __name__=="__main__":
    print('Debug -- Movielens Feature Generation Demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)
    spark = init_spark()

    ## preprocessing
    users, movies, ratings, imdb = read_dataset(**params)
    gbm_features = generate_gbm_features(spark, users, movies, ratings)
    merged_dataset = merge_dataset(users, movies, ratings)

    fg_dataset = generate_sparse_features(merged_dataset)
    train_fg_dataset, test_fg_dataset = split_train_test(fg_dataset)
    train_neg_sample = negative_sampling_train_dataset(spark, train_fg_dataset, num_negs)
    
    # for match model
    # Ref: SimpleX: A Simple and Strong Baseline for Collaborative Filtering
    match_train_dataset = prepare_match_train(spark, train_fg_dataset, train_neg_sample)
    match_test_dataset = prepare_match_test(spark, test_fg_dataset)
    match_item_dataset = prepare_match_item(spark, train_fg_dataset, test_fg_dataset)
    
    match_train_dataset.write.parquet(match_train_dataset_out_path, mode="overwrite")
    match_test_dataset.write.parquet(match_test_dataset_out_path, mode="overwrite")
    match_item_dataset.write.parquet(match_item_dataset_out_path, mode="overwrite")

    # for rank model
    # Ref: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    rank_train_dataset = prepare_rank_train(spark, train_fg_dataset)
    rank_test_dataset = prepare_rank_test(spark, test_fg_dataset)
    rank_lgbm_train_dataset = prepare_rank_lgbm_train(spark, gbm_features, rank_train_dataset)
    rank_lgbm_test_dataset = prepare_rank_lgbm_test(spark, gbm_features, rank_test_dataset)
    
    rank_train_dataset.write.parquet(rank_train_dataset_out_path, mode="overwrite")
    rank_test_dataset.write.parquet(rank_test_dataset_out_path, mode="overwrite")
    rank_lgbm_train_dataset.write.parquet(rank_lgbm_train_dataset_out_path, mode="overwrite")
    rank_lgbm_test_dataset.write.parquet(rank_lgbm_test_dataset_out_path, mode="overwrite")

    ## for user and item features stored in mongo
    final_user_features = prepare_user_feaures(spark, fg_dataset, gbm_features)
    final_user_features.write.parquet(user_mongo_dataset_out_path, mode="overwrite")
    final_item_summary, final_item_features = prepare_item_features(spark, movies, imdb, gbm_features)
    final_item_summary.write.parquet(item_summary_mongo_dataset_out_path, mode="overwrite")
    final_item_features.write.parquet(item_fearture_mongo_dataset_out_path, mode="overwrite")

    stop_spark(spark)
