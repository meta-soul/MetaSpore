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
