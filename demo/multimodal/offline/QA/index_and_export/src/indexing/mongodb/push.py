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

from pyspark.sql.types import *
from tqdm import tqdm

from indexing.base import load_index_data
from indexing.mongodb.utils import get_base_parser, create_spark_session, create_spark_RDD


def parse_args():
    parser = get_base_parser()
    parser.add_argument(
        "--index-file", type=str, required=True
    )
    parser.add_argument(
        "--id-field", type=str, required=True
    )
    parser.add_argument(
        "--batch-size", type=int, default=20480
    )
    args = parser.parse_args()
    return args

def write_into_mongo(spark, items, id_field):
    read_df = spark.read.json(create_spark_RDD(spark, items))
    format_df = read_df.withColumn("queryid", read_df[id_field].cast(StringType()))
    #format_df.write.format("mongo").mode("overwrite").save()
    format_df.write.format("mongo").mode("append").save()

def batchify(data, batch_size):
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main(args):
    print("Loading data...")
    data_iter = iter(load_index_data(args.index_file, return_doc='doc'))

    print("Spark session...")
    mongodb_uri = args.mongo_uri.rstrip("/") + "/" + args.mongo_table
    spark = create_spark_session(mongodb_uri)

    #rdd = create_spark_RDD(spark, [item for item in data_iter])
    #rddCollect = rdd.collect()
    #print("Number of Partitions: "+str(rdd.getNumPartitions()))
    #print("Action: First element: "+str(rdd.first()))
    #exit()

    items = [item for item in data_iter]
    print(f"Writing into MongoDB total {len(items)} records...")
    # write by batch to avoid OOM
    for batch in tqdm(batchify(items, args.batch_size)):
        write_into_mongo(spark, batch, args.id_field)

if __name__ == '__main__':
    args = parse_args()
    main(args)

