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

import os
import sys
import time
import pickle
import argparse

from tqdm import tqdm
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

if __package__ is None:
    from spark import init_spark, stop_spark
else:
    from .spark import init_spark, stop_spark

def get_collection(host, port, collection_name):
    connections.connect(host=host, port=port)
    if not utility.has_collection(collection_name):
        return None
    collection = Collection(collection_name)
    collection.load()  # load in memory for online retrieval
    return collection

def get_or_create_collection(host, port, collection_name, 
        item={}, collection_id_field="id", collection_emb_field="emb", 
        collection_desc='', collection_shards=2):
    collection = get_collection(host, port, collection_name)
    if collection is None:
        #print("\nCreating milvus collection...")
        schema = create_schema_from_data(item, 
            id_field=collection_id_field, emb_field=collection_emb_field, desc=collection_desc)
        collection = Collection(
            name=collection_name,
            schema=schema, 
            shards_num=collection_shards
        )
        #print("\nCreated collection: {}".format(collection.schema))
    return collection

def create_schema_from_data(item, id_field="id", emb_field="emb", desc=''):
    fields = []
    for k, v in item.items():
        if k == id_field:
            fields.append(FieldSchema(
                name=k, 
                dtype=DataType.INT64, 
                is_primary=True, 
            ))
        elif k == emb_field:
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.FLOAT_VECTOR,
                dim=len(v)
            ))
        elif isinstance(v, int):
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.INT64
            ))
        elif isinstance(v, float):
            fields.append(FieldSchema(
                name=k,
                dtype=DataType.FLOAT
            ))
    schema = CollectionSchema(fields=fields, description=desc)
    return schema

def insert_into_collection(collection, data_iter, fields, batch_size=1024, interval=0.0):
    data = [[] for i in range(len(fields))]
    for item in tqdm(data_iter):
        for i in range(len(fields)):
            data[i].append(item[fields[i]])
        if len(data[0]) == batch_size:
            collection.insert(data)
            data = [[] for i in range(len(fields))]
            time.sleep(interval)  # sleep
    if not data and not data[0]:
        collection.insert(data)

def push_milvus(milvus_host, milvus_port, milvus_collection, data_path, fields, 
        id_field="id", emb_field="emb", collection_desc="", collection_shards=2, 
        write_batch=1024, write_interval=0.1, index_type="IVF_FLAT", index_metric="IP", 
        index_nlist=1024, job_name="push-milvus", spark_conf=None):
    if isinstance(spark_conf, str):
        spark = init_spark(job_name, conf_str=spark_conf)
    elif isinstance(spark_conf, dict):
        spark = init_spark(job_name, conf=spark_conf)
    else:
        spark = init_spark(job_name)

    #print("Load data...")
    data_df = spark.read.parquet(data_path)
    data_df = data_df.select(fields)
    if data_df.count() == 0:
        #print("data is empty!")
        return
    item = data_df.limit(1).rdd.collect()[0].asDict()

    #print("\nConnect milvus connection...")
    #print(utility.list_collections())
    collection = get_or_create_collection(milvus_host, milvus_port, milvus_collection,
        item=item, collection_id_field=id_field, collection_emb_field=emb_field,
        collection_desc=collection_desc, collection_shards=collection_shards)

    schema = collection.schema.to_dict()
    fields = [x['name'] for x in schema['fields']]
    index_field = emb_field
    assert index_field in fields

    #print("\nInsert into collection...")
    insert_into_collection(collection, data_df.collect(), fields,
        batch_size=write_batch, interval=write_interval)

    spark.sparkContext.stop()

    #print("\nBuilding index...")
    index_params = {
        "index_type": index_type,
        "metric_type": index_metric,
        "params": {"nlist": index_nlist}
    }
    res = collection.create_index(field_name=index_field, index_params=index_params)
    #print(f"\nResults: {res}")
