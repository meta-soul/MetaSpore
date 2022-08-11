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
import string
from tokenize import group
import metaspore as ms
import attrs
import cattrs
import logging

from pyspark.sql import DataFrame, Window, functions as F
from pyspark.mllib.evaluation import RankingMetrics

from ..utils import get_class, remove_none_value
from ..constants import *

logger = logging.getLogger(__name__)

@attrs.frozen
class PopularsRetrievalConfig:
    recall_nums = attrs.field(validator=attrs.validators.instance_of(int))
    group_nums = attrs.field(validator=attrs.validators.instance_of(int))
    model_out_path = attrs.field(validator=attrs.validators.instance_of(string))

class PopularRetrievalModule():
    def __init__(self, conf: PopularsRetrievalConfig):
        if isinstance(conf, dict):
            self.conf = PopularRetrievalModule.convert(conf)
        elif isinstance(conf, PopularsRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or PopularsRetrievalConfig. Current type is {}".format(type(conf)))

    def convert(self, conf: dict) -> PopularsRetrievalConfig:
        model_params = cattrs.structure(conf['model_params'], PopularsRetrievalConfig)
        return model_params

    def train(self, train_dataset, label_column, label_value, user_id_column, item_id_column, group_nums, recall_nums):
        recall_result = train_dataset.filter(F.col(label_column)==label_value) \
                            .groupBy(item_id_column)\
                            .agg(F.countDistinct(user_id_column))\
                            .sort(F.col('count('+ user_id_column +')').desc())\
                            .limit(group_nums * recall_nums) 
        recall_result = recall_result.withColumn('key', F.floor(F.rand() * group_nums))
        
        ## sort according to count value in each group
        recall_result = recall_result.withColumn('rank', F.dense_rank().over(
                            Window.partitionBy('key').orderBy(F.col('count(' + user_id_column + ')'))))
        
        ## calculate the score
        recall_result = recall_result.withColumn('score', 1 / (1 + F.col('rank')))\
                            .drop(F.col('rank'))\
                            .drop(F.col('count(' + user_id_column + ')'))        
        recall_result = recall_result.withColumn('value', F.struct(item_id_column, 'score'))\
                            .drop(F.col(item_id_column))\
                            .drop(F.col('score'))
        recall_result = recall_result.groupBy('key').agg(F.collect_list('value').alias('value_list'))
        return recall_result

    def transform(self, recall_result, test_dataset):
        recall_result = recall_result.filter(F.col('key')==0)        
        test_result = test_dataset.join(recall_result.select('value_list'), None, 'full') 
        str_schema = 'array<struct<name:string,_2:double>>'
        test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
        
    def evaluate(test_result):
        logger.info("test result sample:")
        test_result.show(20)

        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, ITEM_ID_COLUMN_NAME)]))
        metrics = RankingMetrics(prediction_label_rdd)
        metric_dict = {}
        metric_dict['Precision@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.precisionAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['Recall@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.recallAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['MAP@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.meanAveragePrecisionAt(METRIC_RETRIEVAL_COUNT)
        metric_dict['NDCG@{}'.format(METRIC_RETRIEVAL_COUNT)] = metrics.ndcgAt(METRIC_RETRIEVAL_COUNT)
        print('Debug -- metric_dict: ', metric_dict)
        
        logger.info('I2I - evaluation: done')

    def run(self, train_dataset, test_dataset,  label_column='label', label_value='1',
            user_id_column='user_id', item_id_column='item_id'):
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        
        popular_match = self.train(train_dataset, label_column, label_value, user_id_column, item_id_column, 
                                   self.conf.group_nums, self.conf.recall_nums)
        
        if self.conf.model_out_path:
            popular_match.write.parquet(self.conf.model_out_path, mode="overwrite")
            logger.info('Popular - persistence: done')

        metric_dict = {}
        if test_dataset:
            if not isinstance(test_dataset, DataFrame):
                raise ValueError("Type of test_dataset must be DataFrame.") 
            metric_dict = self.evaluate(test_dataset, item_id_column=item_id_column)
            logger.info('Popular evaluation metrics: {}'.format(metric_dict))

        return popular_match, metric_dict
        