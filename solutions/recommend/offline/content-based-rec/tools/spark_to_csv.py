import pyspark

def init_spark(app_name, executor_memory='10G', executor_instances='4', executor_cores='4', 
               default_parallelism='400', **kwargs):
    spark = pyspark.sql.SparkSession.builder\
            .appName(app_name) \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.executor.instances", executor_instances) \
            .config("spark.executor.cores", executor_cores) \
            .config("spark.default.parallelism", default_parallelism) \
            .getOrCreate()
    sc = spark.sparkContext
    print(sc.version)
    print(sc.applicationId)
    print(sc.uiWebUrl)
    return spark

def stop_spark(spark):
    spark.sparkContext.stop()

def read_dataset(spark, train_path, test_path):
    train_dataset = spark.read.parquet(train_path)
    test_dataset = spark.read.parquet(test_path)
    return train_dataset, test_dataset

if __name__ == '__main__':
    spark = init_spark('download itemcf')
    #'s3://dmetasoul-bucket/demo/movielens/1m/cf/train.parquet'
    #'s3://dmetasoul-bucket/demo/movielens/1m/cf/test.parquet'
    train_dataset, test_dataset = read_dataset(spark, './data/ml-1m-split/train.parquet', './data/ml-1m-split/test.parquet')
    train_dataset.repartition(1).write.option("header", "true").csv("./data/ml-1m-split/train")
    test_dataset.repartition(1).write.option("header", "true").csv("./data/ml-1m-split/test")
    stop_spark(spark)
