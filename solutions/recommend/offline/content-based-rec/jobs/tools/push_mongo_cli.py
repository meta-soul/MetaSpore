import argparse

import pymongo
from pyspark.sql import SparkSession

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mongo-uri", type=str, required=True
    )
    parser.add_argument(
        "--mongo-database", type=str, required=True
    )
    parser.add_argument(
        "--mongo-collection", type=str, required=True
    )
    parser.add_argument(
        "--data", type=str, required=True
    )
    parser.add_argument(
        "--fields", type=str, required=True
    )
    parser.add_argument(
        "--index-fields", type=str, default=""
    )
    parser.add_argument(
        "--write-mode", type=str, default="append", choices=["append", "overwrite"]
    )
    args = parser.parse_args()
    return args


def main(args):
    fields = [n for n in args.fields.split(',')]
    index_fields = [n for n in args.index_fields.split(',') if n in fields]

    spark = SparkSession \
        .builder \
        .appName("push-mongodb") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.input.uri", args.mongo_uri) \
        .config("spark.mongodb.output.uri", args.mongo_uri) \
        .getOrCreate()

    data_df = spark.read.parquet(args.data)
    data_df = data_df.select(fields)
    
    data_df.write \
        .format("mongo") \
        .mode(args.write_mode) \
        .option("database", args.mongo_database) \
        .option("collection", args.mongo_collection) \
        .save()

    spark.sparkContext.stop()

    if index_fields:
        client = pymongo.MongoClient(args.mongo_uri)
        collection = client[args.mongo_database][args.mongo_collection]
        for field_name in index_fields:
            collection.create_index([(field_name, pymongo.ASCENDING)], unique=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)

