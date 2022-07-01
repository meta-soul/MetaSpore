import os
import sys
import time
import pickle
import argparse

from tqdm import tqdm
from pyspark.sql import SparkSession
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

def init_spark(app_name, executor_memory='10G', executor_instances='4', executor_cores='4',
               default_parallelism='100', **kwargs):
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.default.parallelism", default_parallelism)
        .config("spark.executor.memoryOverhead", "2G")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.network.timeout","500")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .getOrCreate())
    return spark

def get_collection(host, port, collection_name):
    connections.connect(host=host, port=port)
    if not utility.has_collection(collection_name):
        return None
    collection = Collection(collection_name)
    collection.load()  # 执行向量相似性搜索之前将 collection 加载到内存中，以便在内存中执行检索计算
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--milvus-host", type=str, required=True
    )
    parser.add_argument(
        "--milvus-port", type=int, required=True
    )
    parser.add_argument(
        "--milvus-collection", type=str, required=True
    )
    parser.add_argument(
        "--data", type=str, required=True
    )
    parser.add_argument(
        "--fields", type=str, required=True
    )
    parser.add_argument(
        "--id-field", type=str, default="id"
    )
    parser.add_argument(
        "--emb-field", type=str, default="emb"
    )
    parser.add_argument(
        "--desc", type=str, default=""
    )
    parser.add_argument(
        "--shards", type=int, default=2
    )
    parser.add_argument(
        "--write-batch", type=int, default=1024
    )
    parser.add_argument(
        "--write-interval", type=float, default=0.1
    )
    parser.add_argument(
        "--index-type", type=str, default="IVF_FLAT"
    )
    parser.add_argument(
        "--index-metric", type=str, default="IP"
    )
    parser.add_argument(
        "--index-nlist", type=int, default=1024
    )
    args = parser.parse_args()
    return args

def main(args):
    spark = init_spark("push-milvus")

    print("Load data...")
    data_df = spark.read.parquet(args.data)
    data_df = data_df.select(args.fields.split(','))
    if data_df.count() == 0:
        exit()
    item = data_df.limit(1).rdd.collect()[0].asDict()

    print("\nConnect milvus connection...")
    #print(utility.list_collections())
    collection = get_or_create_collection(args.milvus_host, args.milvus_port, args.milvus_collection,
        item=item, collection_id_field=args.id_field, collection_emb_field=args.emb_field,
        collection_desc=args.desc, collection_shards=args.shards)

    schema = collection.schema.to_dict()
    fields = [x['name'] for x in schema['fields']]
    index_field = args.emb_field
    assert index_field in fields

    print("\nInsert into collection...")
    insert_into_collection(collection, data_df.collect(), fields,
        batch_size=args.write_batch, interval=args.write_interval)

    print("\nBuilding index...")
    index_params = {
        "index_type": args.index_type,
        "metric_type": args.index_metric,
        "params": {"nlist": args.index_nlist}
    }
    res = collection.create_index(field_name=index_field, index_params=index_params)
    print(f"\nResults: {res}")

    spark.sparkContext.stop()


if __name__ == '__main__':
    args = parse_args()
    main(args)
