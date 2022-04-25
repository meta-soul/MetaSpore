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

from indexing.mongodb.utils import get_base_parser, create_spark_session, create_spark_RDD


def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    return args

def main(args):
    print("Spark session...")
    mongodb_uri = args.mongo_uri.rstrip("/") + "/" + args.mongo_table
    spark = create_spark_session(mongodb_uri)

    df = spark.read.format('mongo').load()
    print(df)
    print(df.printSchema())
    print(df.show())
