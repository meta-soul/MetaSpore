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

from logging import Logger

import metaspore as ms
from ..utils import get_class

import attrs
from typing import Dict, Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F 


@attrs.frozen
class I2IRetrievalConfig:
    i2i_estimator_class = attrs.field(validator=attrs.validators.instance_of(Dict))
    model_out_path = attrs.field(default=None, validator=attrs.validators.instance_of(str))
    max_recommendation_count = attrs.field(default=20, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0), attrs.validators.le(100)])

class I2IRetrievalModule():
    def __init__(self, conf: I2IRetrievalConfig, logger: Logger):
        self.conf = conf
        self.logger = logger
        self.model = None
        self.metric_position_k = 20
        
    def train(self, train_dataset):
        estimator_class = get_class(**self.conf.i2i_estimator_class)
        
        estimator = estimator_class(user_id_column_name = 'user_id',
                                    item_id_column_name = 'item_id',
                                    behavior_column_name = 'label',
                                    behavior_filter_value = '1',
                                    max_recommendation_count = self.conf.max_recommendation_count)
        self.model = estimator.fit(train_dataset)
        self.logger.info('I2I - training: done')
    
    def predict(self, test_dataset):
        # prepare trigger item id 
        original_item_id ='original_item_id'
        test_df = test_dataset.withColumnRenamed('item_id', original_item_id)
        test_df = test_df.withColumnRenamed('last_item_id', 'item_id')
        
        # transform test dataset
        test_result = self.model.transform(test_df)
        
        # revert original item id
        test_result = test_result.withColumnRenamed('item_id', 'last_item_id')
        test_result = test_result.withColumnRenamed(original_item_id, 'item_id')
        
        str_schema = "array<struct<name:string,_2:double>>"
        test_result = test_result.withColumn('rec_info', F.col("value").cast(str_schema))
        
        self.logger.info('I2I - inference: done')
        return test_result
    
    def evaluate(self, test_result):
        print('Debug -- test sample:')
        test_result.select('user_id', (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)
        
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, 'item_id')]))

        metrics = RankingMetrics(prediction_label_rdd)
        
        metric_dict = {}
        metric_dict['Precision@{}'.format(self.metric_position_k)] = metrics.precisionAt(self.metric_position_k)
        metric_dict['Recall@{}'.format(self.metric_position_k)] = metrics.recallAt(self.metric_position_k)
        metric_dict['MAP@{}'.format(self.metric_position_k)] = metrics.meanAveragePrecisionAt(self.metric_position_k)
        metric_dict['NDCG@{}'.format(self.metric_position_k)] = metrics.ndcgAt(self.metric_position_k)
        print('Debug -- metric_dict: ', metric_dict)
        
        self.logger.info('I2I - evaluation: done')
        return metric_dict
        
    
    def run(self, train_dataset: DataFrame, test_dataset: DataFrame) -> Tuple[DataFrame, Dict[str, float]]:
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        if not isinstance(test_dataset, DataFrame):
            raise ValueError("Type of test_dataset must be DataFrame.")
        
        # 1. create estimator and fit model
        self.train(train_dataset)
        
        # 2. transform test data using self.model
        test_result = self.predict(test_dataset)
        
        # 3. get metric dictionary (metric name -> metric value)
        metric_dict = self.evaluate(test_result)
        
        # 4. save model.df to storage if needed.
        if self.conf.model_out_path:
            self.model.df.write.parquet(self.conf.model_out_path, mode="overwrite")
            self.logger.info('I2I - persistence: done')
        
        return self.model.df, metric_dict