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

from pyspark.sql import SparkSession
import pymongo
from .node import PipelineNode
from ..utils import start_logging

class MongoDBDumperNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        
        mongodb = payload['conf']['mongodb']
        df_to_mongodb = payload['df_to_mongodb']
        logger = start_logging(**payload['conf']['logging'])

        logger.info('Dump to MongoDB: start')
        df_to_mongodb.write \
            .format("mongo") \
            .mode(mongodb['write_mode']) \
            .option("uri", mongodb['uri']) \
            .option("database", mongodb['database']) \
            .option("collection", mongodb['collection']) \
            .save()
        
        logger.info('Dump to MongoDB: index')
        if len(mongodb['index_fields']) > 0:
            client = pymongo.MongoClient(mongodb['connection_uri'])
            collection = client[mongodb['database']][mongodb['collection']]
            for field_name in mongodb['index_fields']:
                collection.create_index([(field_name, pymongo.ASCENDING)], unique=mongodb['index_unique'])
                
        logger.info('Dump to MongoDB: done')
        
        return payload