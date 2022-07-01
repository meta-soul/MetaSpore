from pyspark.sql import SparkSession


def init_spark(app_name, conf_str="", conf={}):
    for kv in conf_str.split(','):
        if not kv:
            continue
        k, v = kv.split(':')
        conf[k] = v
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", conf.get('spark.executor.memory', '10G'))
        .config("spark.executor.instances", conf.get('spark.executor.instances', '4'))
        .config("spark.executor.cores", conf.get('spark.executor.cores', '4'))
        .config("spark.default.parallelism", conf.get('spark.default.parallelism', '100'))
        .config("spark.executor.memoryOverhead", conf.get('spark.executor.memoryOverhead', "2G"))
        .config("spark.sql.adaptive.enabled", conf.get('spark.sql.adaptive.enabled', 'true'))
        .config("spark.sql.autoBroadcastJoinThreshold", conf.get('spark.sql.autoBroadcastJoinThreshold', "64MB"))
        .config("spark.network.timeout", conf.get('spark.network.timeout', "500"))
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .getOrCreate())
    return spark

def stop_spark(spark):
    spark.sparkContext.stop()

def get_spark_session(app_name, executor_memory='10G', executor_instances='4', executor_cores='4',
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
