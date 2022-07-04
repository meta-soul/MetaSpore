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
from pyspark.sql import SparkSession

def push_mongo(mongo_uri, mongo_database, mongo_collection, data_path, fields, 
        index_fields=[], write_mode="overwrite", job_name="push-mongodb", spark_conf={}):
    spark = SparkSession \
        .builder \
        .appName(job_name) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.input.uri", mongo_uri) \
        .config("spark.mongodb.output.uri", mongo_uri) \
        .getOrCreate()

    data_df = spark.read.parquet(data_path)
    data_df = data_df.select(fields)
    
    data_df.write \
        .format("mongo") \
        .mode(write_mode) \
        .option("database", mongo_database) \
        .option("collection", mongo_collection) \
        .save()

    spark.sparkContext.stop()

    if index_fields:
        client = pymongo.MongoClient(mongo_uri)
        collection = client[mongo_database][mongo_collection]
        for field_name in index_fields:
            collection.create_index([(field_name, pymongo.ASCENDING)], unique=True)
