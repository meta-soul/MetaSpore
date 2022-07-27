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

import pymongo
import attrs
from typing import Optional, List


@attrs.frozen
class DumpToMongoDBConfig:
    write_mode: attrs.field(validator=attrs.validators.matches_re('^append$|^overwrite$'))
    uri: attrs.field(validator=attrs.validators.matches_re('^mongodb://.+$'))
    database: str
    collection: str
    index_fields: Optional[List[int]]
    index_unique: Optional[bool]
    

class DumpToMongoDBModule():
    def __init__(self, conf: DumpToMongoDBConfig, logger: Logger):
        self.conf = conf
        self.logger = logger
    
    def run(self, df_to_mongodb) -> dict:
        self.logger.info('Dump to MongoDB: start')
        df_to_mongodb.write \
            .format("mongo") \
            .mode(self.conf.write_mode) \
            .option("uri", self.conf.uri) \
            .option("database", self.conf.database) \
            .option("collection", self.conf.collection) \
            .save()
        
        self.logger.info('Dump to MongoDB: index')
        if len(self.conf.index_fields) > 0:
            client = pymongo.MongoClient(self.conf.uri)
            collection = client[self.conf.database][self.conf.collection]
            for field_name in self.conf.index_fields:
                collection.create_index([(field_name, pymongo.ASCENDING)], unique=self.conf.index_unique)           
        self.logger.info('Dump to MongoDB: done')