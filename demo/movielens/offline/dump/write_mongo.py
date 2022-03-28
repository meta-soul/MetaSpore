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
    mongodb_uri = "mongodb://jpa:Dmetasoul_123456@172.31.37.47:27017/jpa." + mongo_table_name
    data_path = "s3://dmetasoul-bucket/demo/movielens/mango/" + s3_file_name + ".parquet/*"
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

