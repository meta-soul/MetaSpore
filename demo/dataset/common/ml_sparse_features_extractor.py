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

import time
from functools import reduce
from pyspark.sql import functions as F

def get_recent_items(kv_pairs, max_len, sep):
    ''' generte user behaviour sequence features

    Args 
      - kv_pairs: [movie_id, timestamp, genre, year]
      - max_len: maximum user behavior sequence length
      - seq: squence splitter
    '''

    # sort by timestatamp
    kv_pairs.sort(key=lambda x: x[1])
    # get recent list
    recent_items = []
    for i in range(0, len(kv_pairs)):
        current, hist_time, genre, year = kv_pairs[i]
        # get last max_len items
        hist_list = [0] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[0]], kv_pairs[max(0, i-max_len):i]))
        last_movie = str(hist_list[-1])
        hist_list = str.join(sep, map(str, hist_list))
        last_genre = 'None' if i == 0 else kv_pairs[i-1][2]
        # history item's genre
        hist_genre_list = ['None'] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[2]], kv_pairs[max(0, i-max_len):i]))
        #hist_genre_list = str.join(sep, map(lambda x:x.split('|')[0], hist_genre_list))  # just keep the first genre
        hist_genre_list = str.join(sep, hist_genre_list)
        # history item's year
        hist_year_list = ['None'] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[3]], kv_pairs[max(0, i-max_len):i]))
        hist_year_list = str.join(sep, hist_year_list)
        recent_items.append((hist_time, current, hist_list, hist_genre_list, hist_year_list, last_movie, last_genre))
    return recent_items

def generate_sparse_features_1m(dataset, max_len=10, sep=u'\u0001', verbose=True):
    ''' generate sparese features for MovieLens-1M dataset

    Args
      - dataset: pyspark dataframe
      - max_len: length of user sequences
      - sep: seperator char
      - verbose: whether to print more details
    '''

    start = time.time()
    # label
    dataset = dataset.withColumn('label',  F.when(F.col('rating')> 0, 1).otherwise(0))
    
    # generate user recent behaviors features
    hist_item_list_df = dataset.filter(dataset['rating']>0).select('user_id','movie_id','timestamp', 'genre', 'year').distinct().rdd\
                               .map(lambda x: (x['user_id'], [(x['movie_id'], x['timestamp'], x['genre'], x['year'])]))\
                               .reduceByKey(lambda x, y: x + y)\
                               .map(lambda x: (x[0], get_recent_items(x[1], max_len, sep)))\
                               .flatMapValues(lambda x: x)\
                               .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6]))\
                               .toDF(['user_id', 'timestamp', 'movie_id', 'recent_movie_ids', 'recent_movie_genres', 'recent_movie_years', 'last_movie', 'last_genre'])
    
    # merge features
    fg_result = dataset.alias('t1')\
                       .join(hist_item_list_df.alias('t2'), \
                             (F.col('t1.user_id')==F.col('t2.user_id')) & (F.col('t1.timestamp')==F.col('t2.timestamp')) & (F.col('t1.movie_id')==F.col('t2.movie_id')),
                             how='leftouter')\
                       .select('t1.label', 't1.user_id', 't1.gender', 't1.age', 't1.occupation', 't1.zip', 't1.movie_id', \
                               't2.recent_movie_ids', 't2.recent_movie_genres', 't2.recent_movie_years', 't1.year', 't1.genre', 't1.rating','t2.last_movie', 't2.last_genre', 't1.timestamp')
    
    # replace sep in genre column
    fg_result = fg_result.withColumn('genre', F.regexp_replace('genre', '\|', sep))
    fg_result = fg_result.withColumn('last_genre', F.regexp_replace('last_genre', '\|', sep))
    
    print('Debug -- movliens-1m generate_sparse_features_1m sparse features cost time:', time.time() - start)
    if verbose:
        print('Debug -- movliens-1m generate_sparse_features_1m result sample:')
        fg_result.show(10)
        print('Debug -- movliens-1m generate_sparse_features_1m sparse features total cost time:', time.time() - start)
    return fg_result

def generate_sparse_features_25m(dataset, max_len=10, sep=u'\u0001', verbose=True):  
    ''' generate sparese features for MovieLens-25M dataset
    
    Args
      - dataset: pyspark dataframe
      - max_len: length of user sequences
      - sep: seperator char
      - verbose: whether to print more details
    '''  
    
    start = time.time()
    # label
    dataset = dataset.withColumn('label',  F.when(F.col('rating')> 0, 1).otherwise(0))
    
    # generate user recent behaviors features
    hist_item_list_df = dataset.filter(dataset['rating']>0).select('user_id','movie_id','timestamp', 'genre', 'year').distinct().rdd\
                               .map(lambda x: (x['user_id'], [(x['movie_id'], x['timestamp'], x['genre'], x['year'])]))\
                               .reduceByKey(lambda x, y: x + y)\
                               .map(lambda x: (x[0], get_recent_items(x[1], max_len, sep)))\
                               .flatMapValues(lambda x: x)\
                               .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6]))\
                               .toDF(['user_id', 'timestamp', 'movie_id', 'recent_movie_ids', 'recent_movie_genres', 'recent_movie_years', 'last_movie', 'last_genre'])
    
    # merge features
    fg_result = dataset.alias('t1')\
                       .join(hist_item_list_df.alias('t2'), \
                             (F.col('t1.user_id')==F.col('t2.user_id')) & (F.col('t1.timestamp')==F.col('t2.timestamp')) & (F.col('t1.movie_id')==F.col('t2.movie_id')),
                             how='leftouter')\
                       .select('t1.label', 't1.user_id', 't1.movie_id', \
                               't2.recent_movie_ids', 't2.recent_movie_genres', 't2.recent_movie_years', 't1.year', 't1.genre', 't1.rating','t2.last_movie', 't2.last_genre', 't1.timestamp')
    
    # replace sep in genre column
    fg_result = fg_result.withColumn('genre', F.regexp_replace('genre', '\|', sep))
    fg_result = fg_result.withColumn('last_genre', F.regexp_replace('last_genre', '\|', sep))
    
    print('Debug -- movielens-25m generate_sparse_features_25m sparse features cost time:', time.time() - start)
    if verbose:
        print('Debug -- movielens-25m generate_sparse_features_25m result sample:')
        fg_result.show(10)
        print('Debug -- movielens-25m generate_sparse_features_25m sparse features total cost time:', time.time() - start)
    return fg_result
