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
import numpy as np
from pyspark.sql.functions import lit, col, pow

def gen_sample_prob(dataset, group_by, alpha=0.75):
    ''' generate the sample probabilities of the dataset using:
        p(w_i) = \frac{f(w_i)^\alpha}{\sum_{j=0}{N} f(w_j)^\alpha}

    Args
      - dataset: pyspark dataframe
      - group_by: movie_id, item_id that sampled by
      - alpha: discounting factor
    '''

    start = time.time()
    item_weight = dataset.groupBy(col(group_by)).count()
    item_weight = item_weight.withColumn('norm_weight', pow(item_weight['count'], alpha))
    total_norm = item_weight.select('norm_weight').groupBy().sum().collect()[0][0]
    item_weight = item_weight.withColumn('sampling_prob', item_weight['norm_weight']/total_norm)
    print('Debug -- neg_sampler.gen_sample_prob cost time:', time.time() - start)
    return item_weight

def sample(user_id, user_item_list, item_list, dist_list, negative_sample):
    ''' Sample negative list for a user through numpy

    Args
      - user_id: user id
      - user_item_list: item list that use have iteractions
      - item_list: item total list
      - dist_list: item sampling probabilities
      - negative_sample: how many negative samples for one positive sample
    '''

    # sample negative list
    candidate_list = np.random.default_rng().choice(list(item_list), size=len(user_item_list)*negative_sample, \
                                    replace=True, p=list(dist_list)).tolist()
    # remove the positive sample from the sampling result
    candidate_list = list(set(candidate_list)-set(user_item_list))
    # sample trigger list
    trigger_list = np.random.default_rng().choice(list(user_item_list), size=len(candidate_list), \
                                    replace=True).tolist()
    return list(zip(trigger_list, candidate_list))

def negative_sampling(spark,
                      dataset,
                      user_column,
                      item_column,
                      time_column,
                      negative_item_column,
                      negative_sample=3,
                      reserve_other_columns=[]):
    ''' negative sampling on original dataset

    Args
      spark: spark session
      dataset: original dataset
      user_column: user id column
      item_column: item id column
      time_column: timestamp for positive sample
      negative_item_column: negative item id column
      negative_sample: how many negative samples for one positive sample
    '''

    # sampling distribution
    item_weight = gen_sample_prob(dataset, item_column)
    # unzip a list of tupples
    items_dist = item_weight.select(item_column, 'sampling_prob')\
                            .rdd.map(lambda x: (x[0], x[1])).collect()
    zipped_dist = [list(t) for t in zip(*items_dist)]
    item_list, dist_list = zipped_dist[0], zipped_dist[1]
    # normlazation
    dist_list = np.asarray(dist_list).astype('float64')
    dist_list = list(dist_list / np.sum(dist_list))
    print('Debug -- neg_sampler.negative_sampling item_list.size:', len(item_list))
    print('Debug -- neg_sampler.negative_sampling dist_list.size:', len(dist_list))
    # broadcast
    item_list_br = spark.sparkContext.broadcast(item_list)
    dist_list_br = spark.sparkContext.broadcast(dist_list)
    # generate sampling dataframe
    start = time.time()
    sampling_df = dataset.select(user_column, item_column).distinct().rdd\
        .map(lambda x: (x[user_column], [x[item_column]]))\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda x: (x[0], sample(x[0], x[1], item_list_br.value, dist_list_br.value, negative_sample)))\
        .flatMapValues(lambda x: x)\
        .map(lambda x: (x[0], x[1][0], x[1][1]))\
        .toDF([user_column, negative_item_column, item_column])
    # generate timestamp by table join
    user_join_cond = col('t1.{}'.format(user_column))==col('t2.{}'.format(user_column))
    item_join_cond = col('t1.{}'.format(negative_item_column))==col('t2.{}'.format(item_column))
    result_df = sampling_df.alias('t1').\
        join(dataset.alias('t2'), user_join_cond & item_join_cond, how='inner').\
        select(
          ['t1.{}'.format(user_column), 't1.{}'.format(negative_item_column), 't1.{}'.format(item_column)] +
          ['t2.{}'.format(column_name) for column_name in [time_column] + reserve_other_columns]
        )
    print('Debug -- negative_sampling.negative_sampling sampling map-reduce cost time:', time.time() - start)
    return result_df
