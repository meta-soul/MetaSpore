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

import metaspore as ms
import attrs
import cattrs
import logging

from logging import Logger
from typing import Dict, Tuple, Optional
from pyspark.sql import DataFrame, Window, functions as F 
from pyspark.mllib.evaluation import RankingMetrics

from ..constants import *

logger = logging.getLogger(__name__)

@attrs.frozen(kw_only=True)
class FriendsOfFriendsRetrievalConfig:
    max_recommendation_count = attrs.field(validator=attrs.validators.instance_of(int))
    decay_max = attrs.field(validator=attrs.validators.instance_of(int))
    decay_factor = attrs.field(validator=attrs.validators.instance_of(float))
    walk_length = attrs.field(validator=attrs.validators.instance_of(int))
    filer_u2f = attrs.field(validator=attrs.validator(bool))
    model_out_path = attrs.field(default=None, 
        validator=attrs.validators.optional(attrs.validators.instance_of(str)))

class FriendsOfFriendsRetrievalModule:
    def __init__(self, conf):
        if isinstance(conf, dict):
            self.conf = FriendsOfFriendsRetrievalConfig.convert(conf)
        elif isinstance(conf, FriendsOfFriendsRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or FriendsOfFriendsRetrievalConfig. Current type is {}".format(type(conf)))

    @staticmethod
    def convert(conf: dict) -> FriendsOfFriendsRetrievalConfig:
        if not 'estimator_params' in conf:
            raise ValueError("Dict of FriendsOfFriendsRetrievalModule must have key 'estimator_params' !")
        conf = cattrs.structure(conf['estimator_params'], FriendsOfFriendsRetrievalConfig)
        return conf
    
    def u2f_table(self, dataset, user_id, item_id, label, label_value, act_time, time_format='yyyy-MM-dd HH:mm:ss', 
                  decay_max=10000, decay_factor=-0.3):
        dataset = dataset.filter(F.col(label)==label_value)
        item_count = dataset.groupBy(F.col(user_id))\
            .agg(F.countDistinct(item_id).alias('friend_count'))
        rel_score = dataset.alias('ta')\
            .join(item_count.alias('tb'), F.col('ta.'+user_id)==F.col('tb.'+user_id), how='inner')\
            .select('ta.*', 'tb.friend_count')\
            .withColumn('rel_score', F.lit(1)/F.sqrt(F.col('tb.friend_count')))
        if act_time:
            rel_score = rel_score.withColumn('day_diff',\
                F.datediff(F.to_date(F.current_timestamp()), F.to_date(act_time, time_format)))
            rel_score = rel_score.withColumn('day_diff',\
                F.when(F.col('day_diff') > F.lit(decay_max), F.lit(decay_max)).otherwise(F.col('day_diff')))
            rel_score = rel_score.withColumn('rel_score', F.col('rel_score') * F.pow(F.col('day_diff'),decay_factor))
        return rel_score

    def u2fof_table(self, u2f_table, user_id, item_id, walk_length=1, filter_u2f=True, max_recommendation_count=100):
        u2fof_table = u2f_table
        for i in range(0, walk_length):
            u2fof_table = u2fof_table.alias('ta')\
                .join(u2f_table.alias('tb'), F.col('ta.'+item_id)==F.col('tb.'+user_id), how='inner') \
                .filter(F.col('ta.'+user_id)!=F.col('tb.'+item_id)) \
                .groupBy(F.col('ta.'+user_id), F.col('tb.'+item_id)) \
                .agg(F.sum(F.col('ta.rel_score') * F.col('tb.rel_score')).alias('rel_score'))
        
        if filter_u2f:
            on_cond = (F.col('ta.'+user_id)==F.col('tb.'+user_id))&(F.col('ta.'+item_id)==F.col('tb.'+item_id))
            u2fof_table = u2fof_table\
                .alias('ta').join(u2f_table.alias('tb'), on=on_cond, how='left_outer') \
                .select(F.col('ta.*'))\
                .filter(F.col('tb.'+item_id).isNull())

        u2fof_table =u2fof_table\
            .withColumn('rn',F.row_number().over(Window.partitionBy(user_id).orderBy(F.desc('rel_score')))) \
            .filter(f'rn <= %d'%max_recommendation_count)  \
            .groupBy(user_id) \
            .agg(F.collect_list(F.struct(F.col(item_id), F.col('rel_score'))).alias('value_list')) \
            .withColumnRenamed(user_id, 'key')
        return u2fof_table
    
    def train(self, train_dataset, time_column, time_format, label_column, label_value, 
              user_id_column, item_id_column, decay_max, decay_factor, max_recommendation_count):
        u2f_table = self.u2f_table(train_dataset, user_id_column, item_id_column, label_column,
            label_value, time_column, time_format, decay_max, decay_factor)
        recall_result = self.u2fof_table(u2f_table, user_id_column, item_id_column, 
            max_recommendation_count=max_recommendation_count)
        return recall_result

    def transform(self, recall_result, test_dataset, user_id_column):
        cond = test_dataset[user_id_column]==recall_result['key']
        test_result = test_dataset.join(recall_result, on=cond, how='left')
        str_schema = 'array<struct<name:string,_2:double>>'
        test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
        return test_result

    def evaluate(self, test_result, user_id_column='user_id', item_id_column='item_id'):
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, item_id_column)]))
        metrics = RankingMetrics(prediction_label_rdd)
        metric_dict = {}
        metric_dict['Precision@{}'.format(self.metric_position_k)] = metrics.precisionAt(self.metric_position_k)
        metric_dict['Recall@{}'.format(self.metric_position_k)] = metrics.recallAt(self.metric_position_k)
        metric_dict['MAP@{}'.format(self.metric_position_k)] = metrics.meanAveragePrecisionAt(self.metric_position_k)
        metric_dict['NDCG@{}'.format(self.metric_position_k)] = metrics.ndcgAt(self.metric_position_k)
        logger.info('Metrics: {}'.format(metric_dict))
        logger.info('TwoTwoers - evaluation: done')
        return metric_dict

    def run(self, train_dataset: DataFrame, test_dataset: DataFrame, time_column=None, time_format=None, label_column='label', label_value='1',
            user_id_column='user_id', item_id_column='item_id') -> Tuple[DataFrame, Dict[str, float]]:
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        
        fof_match_result = self.train(self, train_dataset, time_column, time_format, label_column, label_value, 
            user_id_column, item_id_column, 
            self.conf.decay_max, self.conf.decay_factor, 
            self.conf.max_recommendation_count
        )
        
        if self.conf.model_out_path:
            fof_match_result.write.parquet(self.conf.model_out_path, mode="overwrite")
            logger.info('Friends of Friends - persistence: done')

        metric_dict = {}
        if test_dataset:
            if not isinstance(test_dataset, DataFrame):
                raise ValueError("Type of test_dataset must be DataFrame.") 
            test_result = self.transform(fof_match_result, test_dataset, user_id_column)
            metric_dict = self.evaluate(test_result, user_id_column, item_id_column)
            logger.info('Friends of Friends evaluation metrics: {}'.format(metric_dict))
        
        return fof_match_result, metric_dict
