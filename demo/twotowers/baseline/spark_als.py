from cgi import test
from ctypes import cast
import os
import argparse
import yaml
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import FloatType, LongType
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(app_name, executor_memory, executor_instances, executor_cores, 
               default_parallelism, **kwargs):
    spark = (SparkSession.builder
        .appName(app_name)
        .config("spark.executor.memory", executor_memory)
        .config("spark.executor.instances", executor_instances)
        .config("spark.executor.cores", executor_cores)
        .config("spark.default.parallelism", default_parallelism)
        .config("spark.executor.memoryOverhead", "2G")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
        .config("spark.network.timeout","500")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate())
    
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark

def stop_spark(spark):
    print('Debug -- spark stop')
    spark.sparkContext.stop()

def read_dataset(spark, train_path, test_path, **kwargs):
    train_dataset = spark.read.parquet(train_path)
    test_dataset = spark.read.parquet(test_path)
    test_dataset = test_dataset.filter(test_dataset['label'] == '1')
    
    print('Debug -- match train dataset sample:')
    train_dataset.show(10)
    print('Debug -- match test dataset sample:')
    test_dataset.show(10)

    print('Debug -- train dataset positive count:', train_dataset[train_dataset['label']=='1'].count())
    print('Debug -- train dataset negative count:', train_dataset[train_dataset['label']=='0'].count())
    print('Debug -- test dataset count:', test_dataset.count())

    return train_dataset, test_dataset

def convert_datatype(dataset, user_id_column_name, item_id_column_name, rating_coloumn_name):
    dataset = dataset.withColumn(user_id_column_name, dataset[user_id_column_name].cast(LongType())) \
                     .withColumn(item_id_column_name, dataset[item_id_column_name].cast(LongType())) \
                     .withColumn(rating_coloumn_name, dataset[rating_coloumn_name].cast(FloatType()))
    return dataset

def train(spark, train_dataset, user_id_column_name, item_id_column_name, rating_column_name,
          max_iter, rank, reg_param, **kwargs):
    train_dataset = convert_datatype(train_dataset, user_id_column_name, item_id_column_name, rating_column_name)
    als = ALS(maxIter=max_iter, rank=rank, regParam=reg_param, 
              userCol=user_id_column_name, itemCol=item_id_column_name, 
              ratingCol=rating_column_name, coldStartStrategy="drop", nonnegative = True)
    model = als.fit(train_dataset)
    print('Debug -- train Spark ALS model:', model)
    return model

def transform(spark, model, train_dataset, test_dataset, user_id_column_name, item_id_column_name, rating_column_name,
              last_item_col_name, max_recommendation_count, **kwargs):
    train_dataset = convert_datatype(train_dataset, user_id_column_name, item_id_column_name, rating_column_name)
    test_dataset = convert_datatype(test_dataset, user_id_column_name, item_id_column_name, rating_column_name)
    ## score all userxitem combinations
    dataset = train_dataset.union(test_dataset)
    users = dataset.select(user_id_column_name).distinct()
    items = dataset.select(item_id_column_name).distinct()
    user_item = users.crossJoin(items)
    prediction_result = model.transform(user_item)
    ## retrieval the top k result
    recall_result = test_dataset.alias("test").join(
        prediction_result.alias("pred"),
        prediction_result[user_id_column_name] == test_dataset[user_id_column_name],
        how='inner'
    )
    w = Window.partitionBy('pred.' + user_id_column_name).orderBy(F.col('prediction').desc())
    recall_topk = recall_result.withColumn("row_number", F.row_number().over(w))\
                               .filter(F.col('row_number') < max_recommendation_count)\
                               .select('pred.' + user_id_column_name, 'pred.' + item_id_column_name, 'prediction')\
                               .withColumn('rec_info', F.collect_list('pred' + item_id_column_name).over(w)) \
                               .groupBy('pred' + user_id_column_name)\
                               .agg(F.max('rec_info').alias('rec_info'))
    ## join with the original result
    test_df = test_dataset.select(user_id_column_name, last_item_col_name, item_id_column_name)\
            .groupBy(user_id_column_name, last_item_col_name)\
            .agg(F.collect_set(item_id_column_name).alias('label_items'))

    test_result = test_df.alias('test').join(
        recall_topk.alias('recall'),
        recall_topk['recall.' + user_id_column_name] == test_dataset['test.' + user_id_column_name],
        how='inner'
    )
    return test_result

def evaluate(spark, test_result, test_user=100):
    print('Debug -- test sample:')
    test_result.show(10)
    print('Debug -- test user:%d sample:' % test_user)
    test_result[test_result['user_id']==100].show(10)

    prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                    [xx._1 for xx in x.rec_info] if x.rec_info is not None else [], \
                                     x.label_items))
    return RankingMetrics(prediction_label_rdd)

if __name__=="__main__":
    print('Debug -- Spark ALS Matrix Factorization')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    spark = init_spark(**params)
    ## read datasets
    train_dataset, test_dataset = read_dataset(spark, **params)
    ## train
    model = train(spark, train_dataset, **params)
    ## transform
    test_result = transform(spark, model, train_dataset, test_dataset, **params)
    ## evaluate
    recall_metrics = evaluate(spark, test_result)
    topk = params['max_recommendation_count']
    print("Debug -- Precision@%d: %f" % (topk, recall_metrics.precisionAt(topk)))
    print("Debug -- Recall@%d: %f" % (topk, recall_metrics.recallAt(topk)))
    print("Debug -- MAP@%d: %f" % (topk, recall_metrics.meanAveragePrecisionAt(topk)))
    print("Debug -- NDCG@%d: %f" % (topk, recall_metrics.ndcgAt(topk)))
    stop_spark(spark)