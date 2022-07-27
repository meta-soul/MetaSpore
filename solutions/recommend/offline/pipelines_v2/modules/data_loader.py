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

import attrs
from attrs import frozen
from typing import Optional, Dict
from pyspark.sql import SparkSession, DataFrame

@frozen
class DataLoaderConfig:
    train_path: str
    test_path: str
    item_path: Optional[str]

class DataLoaderModule():
    def __init__(self, conf: DataLoaderConfig, spark: SparkSession, logger: Logger):
        self.conf = conf
        self.spark = spark
        self.logger = logger
    
    def run(self) -> Dict[str, DataFrame]:
        dataset_dict = {}
        
        dataset_dict['train_dataset'] = self.spark.read.parquet(self.conf.train_path)
        self.logger.info('Train dataset is loaded: {}'.format(self.conf.train_path))
        
        dataset_dict['test_dataset'] = self.spark.read.parquet(self.conf.test_path)
        self.logger.info('Test dataset is loaded: {}'.format(self.conf.test_path))
        
        if item_path:
            dataset_dict['item_dataset'] = self.spark.read.parquet(self.conf.item_path)
            self.logger.info('Item dataset is loaded: {}'.format(self.conf.item_path))
        
        return dataset_dict
