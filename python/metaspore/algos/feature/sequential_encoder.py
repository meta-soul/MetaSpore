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

def gen_user_bhv_seq(spark,
                     dataset,
                     user_column = 'user_id',
                     item_coulmn = 'item_id',
                     time_column = 'timestamp',
                     item_seq_column = 'user_bhv_item_seq',
                     last_item_column = 'user_bhv_last_item',
                     max_len=10,
                     sep=u'\u0001'):
    ''' generate sparese features for MovieLens-1M dataset

    Args
      - spark: spark session
      - dataset: pyspark dataframe
      - user_column: user_id column
      - item_column: item_id column
      - time_column: timestamp column
      - item_seq_column: generated item sequence of user behavoir
      - item_last_column: generated last item of user behavoir
      - max_len: length of user sequences
      - sep: seperator char
    '''

    def gen_bhv_seq_by_user(kv_pairs, max_len, sep):
        ''' generte user behaviour sequence features

        Args
        - kv_pairs: [item_id, timestamp]
        - max_len: maximum user behavior sequence length
        - seq: squence splitter
        '''
        # sort by timestatamp
        kv_pairs.sort(key=lambda x: x[1])
        # get recent list
        recent_items = []
        for i in range(0, len(kv_pairs)):
            current, hist_time = kv_pairs[i]
            # get last max_len items
            hist_list = [0] if i == 0 else reduce(lambda x, y:x+y, map(lambda x:[x[0]], kv_pairs[max(0, i-max_len):i]))
            last_item = str(hist_list[-1])
            hist_list = str.join(sep, map(str, hist_list))
            recent_items.append((current, hist_list, last_item, hist_time))
        return recent_items

    start = time.time()
    # generate user recent behaviors features
    hist_item_list_df = dataset.rdd\
        .map(lambda x: (x[user_column], [(x[item_coulmn], x[time_column])]))\
        .reduceByKey(lambda x, y: x + y)\
        .map(lambda x: (x[0], gen_bhv_seq_by_user(x[1], max_len, sep)))\
        .flatMapValues(lambda x: x)\
        .map(lambda x: (x[0], x[1][0], x[1][1], x[1][2], x[1][3]))\
        .toDF([user_column, item_coulmn, item_seq_column, last_item_column, time_column])

    print('Debug -- generate sequential features cost time:', time.time() - start)
    return hist_item_list_df
