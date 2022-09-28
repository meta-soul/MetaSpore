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
from .data_loader import DataLoaderModule, DataLoaderConfig
from .init_spark import InitSparkModule, InitSparkConfig
from .i2i_retrieval import I2IRetrievalModule, I2IRetrievalConfig
from .i2i_retrieval import SwingEstimatorConfig, ItemCFEstimatorConfig
from .popular_retrieval import PopularsRetrievalConfig, PopularRetrievalModule
from .deep_ctr import DeepCTRModule, DeepCTRConfig, DeepCTREstimatorConfig
from .deep_ctr import WideDeepConfig, DeepFMConfig
from .mongodb_dumper import DumpToMongoDBConfig, DumpToMongoDBModule
'''
Initalize the logging settings for this package, avoid the conflicts with the root logging.
'''
from .utils.logger import setup_logging as setup_logging_core
def setup_logging(loglevel=None, **params):
    setup_logging_core(loglevel, __name__)
