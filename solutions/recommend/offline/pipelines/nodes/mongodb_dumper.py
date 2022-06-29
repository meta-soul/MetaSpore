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

from .node import PipelineNode
from ..utils import start_logging

class MongoDBDumperNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        mongodb_conf = payload['conf']['mongodb']
        logger = start_logging(**confs['logging'])
        spark = payload['spark']
        df_to_mongodb = payload['df_to_mongodb']
        
        logger.info('Dump to MongoDB: prepare')
        spark.conf.set('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1')
        spark.conf.set('spark.mongodb.input.uri', mongodb_conf['input_url'])
        spark.conf.set('spark.mongodb.output.uri', mongodb_conf['output_url'])
        
        logger.info('Dump to MongoDB: start')
        df_to_mongodb.write.format("mongo").mode("overwrite").save()
        logger.info('Dump to MongoDB: done')
        
        return payload
    
    