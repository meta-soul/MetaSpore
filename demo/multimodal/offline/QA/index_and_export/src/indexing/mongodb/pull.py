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
