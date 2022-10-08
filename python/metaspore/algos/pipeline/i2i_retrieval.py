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
from pyspark.sql import DataFrame
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F
from attrs import frozen, field
from attrs.validators import optional, instance_of

from .common_validators import recommendation_count_validator
from .utils import get_class, remove_none_value
from .utils.constants import *

logger = logging.getLogger(__name__)

@frozen(kw_only=True)
class SwingEstimatorConfig:
    user_id_column_name = field(default=None, validator=optional(instance_of(str)))
    item_id_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_filter_value = field(default=None, validator=optional(instance_of(str)))
    use_plain_weight = field(default=None, validator=optional(instance_of(bool)))
    smoothing_coefficient = field(default=None, validator=optional(instance_of(float)))
    max_recommendation_count = field(default=None, validator=optional(recommendation_count_validator))
    key_column_name = field(default=None, validator=optional(instance_of(str)))
    value_column_name = field(default=None, validator=optional(instance_of(str)))
    item_score_delimiter = field(default=None, validator=optional(instance_of(str)))
    item_score_pair_delimiter = field(default=None, validator=optional(instance_of(str)))
    cassandra_catalog = field(default=None, validator=optional(instance_of(str)))
    cassandra_host_ip = field(default=None, validator=optional(instance_of(str)))
    cassandra_port = field(default=None, validator=optional(instance_of(int)))
    cassandra_user_name = field(default=None, validator=optional(instance_of(str)))
    cassandra_password = field(default=None, validator=optional(instance_of(str)))
    cassandra_db_name = field(default=None, validator=optional(instance_of(str)))
    cassandra_db_properties = field(default=None, validator=optional(instance_of(str)))
    cassandra_table_name = field(default=None, validator=optional(instance_of(str)))

@frozen(kw_only=True)
class ItemCFEstimatorConfig:
    user_id_column_name = field(default=None, validator=optional(instance_of(str)))
    item_id_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_filter_value = field(default=None, validator=optional(instance_of(str)))
    max_recommendation_count = field(default=None, validator=optional(recommendation_count_validator))
    key_column_name = field(default=None, validator=optional(instance_of(str)))
    value_column_name = field(default=None, validator=optional(instance_of(str)))
    item_score_delimiter = field(default=None, validator=optional(instance_of(str)))
    item_score_pair_delimiter = field(default=None, validator=optional(instance_of(str)))
    debug = field(default=None, validator=optional(instance_of(bool)))

@attrs.frozen(kw_only=True)
class I2IRetrievalConfig:
    i2i_estimator_class = attrs.field()
    i2i_estimator_config_class = attrs.field()
    model_out_path = attrs.field(default=None, validator=optional(instance_of(str)))
    estimator_params = attrs.field()

class I2IRetrievalModule:
    def __init__(self, conf):
        if isinstance(conf, dict):
            self.conf = I2IRetrievalModule.convert(conf)
        elif isinstance(conf, I2IRetrievalConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or I2IRetrievalConfig. Current type is {}".format(type(conf)))

        self.model = None

    @staticmethod
    def convert(conf: dict) -> I2IRetrievalConfig:
        if not 'i2i_estimator_class' in conf:
            raise ValueError("Dict of I2IRetrievalModule must have key 'i2i_estimator_class' !")
        if not 'estimator_params' in conf:
            raise ValueError("Dict of I2IRetrievalModule must have key 'estimator_params' !")

        conf = conf.copy()

        i2i_estimator_class = get_class(conf['i2i_estimator_class'])
        i2i_estimator_config_class = get_class(conf['i2i_estimator_config_class'])

        conventional_param_dict = {'user_id_column_name': USER_ID_COLUMN_NAME,
                                   'item_id_column_name': ITEM_ID_COLUMN_NAME,
                                   'behavior_column_name': BEHAVIOR_COLUMN_NAME,
                                   'behavior_filter_value': BEHAVIOR_FILTER_VALUE}
        conf['estimator_params'].update(conventional_param_dict)
        estimator_params = cattrs.structure(conf['estimator_params'], i2i_estimator_config_class)

        conf['i2i_estimator_class'] = i2i_estimator_class
        conf['estimator_params'] = estimator_params

        return I2IRetrievalConfig(**conf)

    def train(self, train_dataset):
        estimator = self.conf.i2i_estimator_class(**remove_none_value(cattrs.unstructure(self.conf.estimator_params)))

        self.model = estimator.fit(train_dataset)
        logger.info('I2I - training: done')

    def predict(self, test_dataset):
        # prepare trigger item id
        original_item_id ='original_item_id'
        test_df = test_dataset.withColumnRenamed(ITEM_ID_COLUMN_NAME, original_item_id)
        test_df = test_df.withColumnRenamed(LAST_ITEM_ID_COLUMN_NAME, ITEM_ID_COLUMN_NAME)

        # transform test dataset
        test_result = self.model.transform(test_df)

        # revert original item id
        test_result = test_result.withColumnRenamed(ITEM_ID_COLUMN_NAME, LAST_ITEM_ID_COLUMN_NAME)
        test_result = test_result.withColumnRenamed(original_item_id, ITEM_ID_COLUMN_NAME)

        str_schema = "array<struct<name:string,_2:double>>"
        test_result = test_result.withColumn('rec_info', F.col("value").cast(str_schema))

        logger.info('I2I - inference: done')
        return test_result

    def evaluate(self, test_result):
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
        return metric_dict


    def run(self, train_dataset: DataFrame, test_dataset: DataFrame) -> Tuple[DataFrame, Dict[str, float]]:
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        # 1. create estimator and fit model
        self.train(train_dataset)
        metric_dict = {}
        if test_dataset:
            if not isinstance(test_dataset, DataFrame):
                raise ValueError("Type of test_dataset must be DataFrame.")
            # 2. transform test data using self.model
            test_result = self.predict(test_dataset)
            # 3. get metric dictionary (metric name -> metric value)
            metric_dict = self.evaluate(test_result)
        # 4. save model.df to storage if needed.
        if self.conf.model_out_path:
            self.model.df.write.parquet(self.conf.model_out_path, mode="overwrite")
            logger.info('I2I - persistence: done')

        return self.model.df, metric_dict
