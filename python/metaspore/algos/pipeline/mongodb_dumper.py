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

import pymongo
import attrs
import logging

from logging import Logger
from typing import Optional, List
from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

@attrs.frozen(kw_only=True)
class DumpToMongoDBConfig:
    write_mode = attrs.field(validator=attrs.validators.matches_re('^append$|^overwrite$'))
    uri = attrs.field(validator=attrs.validators.matches_re('^mongodb://.+$'))
    database = attrs.field(validator=attrs.validators.instance_of(str))
    index_fields = attrs.field(default=[], validator=attrs.validators.instance_of(List))
    index_unique = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(bool)))
    collection = attrs.field(default=None, validator=attrs.validators.optional(attrs.validators.instance_of(str)))

class DumpToMongoDBModule:
    def __init__(self, conf: DumpToMongoDBConfig):
        self.conf = conf

    def run(self, df_to_mongodb, mongo_collection=None) -> None:
        if not isinstance(df_to_mongodb, DataFrame):
            raise ValueError("Type of df_to_mongodb must be DataFrame.")
        if mongo_collection is None:
            mongo_collection = self.conf.collection
        if mongo_collection is None:
            raise ValueError("mongo collection name should not be None.")

        logger.info('Dump to MongoDB: start')
        df_to_mongodb.write \
            .format("mongo") \
            .mode(self.conf.write_mode) \
            .option("uri", self.conf.uri) \
            .option("database", self.conf.database) \
            .option("collection", mongo_collection) \
            .save()

        logger.info('Dump to MongoDB: index')
        if len(self.conf.index_fields) > 0:
            client = pymongo.MongoClient(self.conf.uri)
            collection = client[self.conf.database][mongo_collection]
            for field_name in self.conf.index_fields:
                collection.create_index([(field_name, pymongo.ASCENDING)], unique=self.conf.index_unique)
        logger.info('Dump to MongoDB: done')
