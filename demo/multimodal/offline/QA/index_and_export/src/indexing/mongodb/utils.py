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

import argparse

from pyspark.sql import SparkSession

def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo-uri", type=str, required=True
    )
    parser.add_argument(
        "--mongo-table", type=str, required=True
    )
    return parser

def create_spark_session(mongodb_uri):
    spark = SparkSession \
        .builder \
        .master("local") \
        .config("spark.mongodb.input.uri", mongodb_uri) \
        .config("spark.mongodb.output.uri", mongodb_uri) \
        .getOrCreate()
    return spark

def create_spark_RDD(spark, collection):
    return spark.sparkContext.parallelize(collection)
