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

import logging
import attrs

from attrs import frozen, field
from attrs.validators import optional, instance_of
from typing import Optional, Dict
from pyspark.sql import SparkSession, DataFrame

logger = logging.getLogger(__name__)

@frozen(kw_only=True)
class DataLoaderConfig:
    train_path = field(validator=instance_of(str))
    test_path = field(default=None, validator=optional(instance_of(str)))
    item_path = field(default=None, validator=optional(instance_of(str)))
    label_column = field(default=None, validator=optional(instance_of(str)))
    label_value = field(default=None, validator=optional(instance_of(str)))
    user_id_column = field(default=None, validator=optional(instance_of(str)))
    item_id_column = field(default=None, validator=optional(instance_of(str)))

class DataLoaderModule:
    def __init__(self, conf: DataLoaderConfig, spark: SparkSession):
        self.conf = conf
        self.spark = spark

    def _add_key_value_pair(self, dict, key, value):
        if value:
            dict[key] = value
        return dict

    def run(self) -> Dict[str, DataFrame]:
        dataset_dict = {}

        dataset_dict['train'] = self.spark.read.parquet(self.conf.train_path)
        logger.info('Train dataset is loaded: {}'.format(self.conf.train_path))

        if self.conf.test_path:
            dataset_dict['test'] = self.spark.read.parquet(self.conf.test_path)
            logger.info('Test dataset is loaded: {}'.format(self.conf.test_path))

        if self.conf.item_path:
            dataset_dict['item'] = self.spark.read.parquet(self.conf.item_path)
            logger.info('Item dataset is loaded: {}'.format(self.conf.item_path))

        dataset_dict = self._add_key_value_pair(dataset_dict, 'label_column', self.conf.label_column)
        dataset_dict = self._add_key_value_pair(dataset_dict, 'label_value', self.conf.label_value)
        dataset_dict = self._add_key_value_pair(dataset_dict, 'user_id_column', self.conf.user_id_column)
        dataset_dict = self._add_key_value_pair(dataset_dict, 'item_id_column', self.conf.item_id_column)

        return dataset_dict
