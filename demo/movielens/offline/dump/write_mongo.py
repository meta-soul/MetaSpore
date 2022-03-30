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
from pyspark.sql.types import *
from pyspark.sql import SparkSession

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', type=str, action='store', default='', help='s3 origin path')
    parser.add_argument('--dest', type=str, action='store', default='', help='mongo table name')
    parser.add_argument('--queryid', type=str, action='store', default='', help='query id column name')
    args = parser.parse_args()
    
    s3_file_name = args.origin
    mongo_table_name = args.dest
    query_id_col = args.queryid
    mongodb_uri = "{MY_MONGO.GPA}" + mongo_table_name
    data_path = "${MY_S3_BUCKET}/movielens/mango/" + s3_file_name + ".parquet/*"
    print("Debug --- data_path: %s, mongodb_uri: %s, query_id_col: %s" % (data_path, mongodb_uri, query_id_col))

    spark = SparkSession \
        .builder \
        .master("local") \
        .config("spark.mongodb.input.uri", mongodb_uri) \
        .config("spark.mongodb.output.uri", mongodb_uri) \
        .getOrCreate()

    if query_id_col != "":
        read_df = spark.read.parquet(data_path)
        format_df = read_df.withColumn("queryid", read_df[query_id_col].cast(StringType()))
        format_df.write.format("mongo").mode("overwrite").save()
    else:
        print("Debug --- query id col is None, please check the input parameters.")

