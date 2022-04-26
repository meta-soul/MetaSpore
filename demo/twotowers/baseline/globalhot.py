import yaml
import argparse
import numpy as np

from pyspark.sql import functions as F
from pyspark.mllib.evaluation import RankingMetrics

import metaspore as ms

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(app_name, local, batch_size, worker_count, server_count,
               worker_memory, server_memory, coordinator_memory, **kwargs):
    spark_confs={
        "spark.network.timeout":"500",
        "spark.ui.showConsoleProgress": "true",
        "spark.kubernetes.executor.deleteOnTermination":"true",
    }
    spark = ms.spark.get_session(local=local,
                                 app_name=app_name,
                                 batch_size=batch_size,
                                 worker_count=worker_count,
                                 server_count=server_count,
                                 worker_memory=worker_memory,
                                 server_memory=server_memory,
                                 coordinator_memory=coordinator_memory,
                                 spark_confs=spark_confs)
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark

def stop_spark(spark):
    print('Debug -- spark stop')
    spark.sparkContext.stop()

def read_dataset(spark, train_path, test_path):
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

def train(spark, train_dataset, max_recommendation_count, **kwargs):
    global_hot = train_dataset.filter(train_dataset.label == '1') \
                              .groupBy(train_dataset.movie_id).count() \
                              .orderBy(['count'], ascending=[0])
    global_hot_topk = global_hot.limit(max_recommendation_count)
    print('Debug -- globalhot@top%d:' % max_recommendation_count)
    global_hot_topk.show(max_recommendation_count)
    return global_hot_topk

def transform(spark, global_hot, test_dataset):
    topk_result = global_hot.agg(F.collect_list('movie_id').alias('prediction'))
    topk_result.registerTempTable('topk_result')
    test_dataset.registerTempTable('test_dataset')
    query = """
    select
        ta.*,
        tb.prediction
    from
        test_dataset ta
    join
        topk_result tb
    """
    test_result = spark.sql(query)
    print('Debug -- test result sample:')
    test_result.show(20)
    return test_result

def evaluate(spark, test_result, test_user=100):
    prediction_label_rdd = test_result.rdd.map(lambda x:( x.prediction, [x.movie_id]))
    metrics = RankingMetrics(prediction_label_rdd)
    return metrics

if __name__=="__main__":
    print('Debug -- Global Hot Recall Demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    spark = init_spark(**params)
    ## read datasets
    train_dataset, test_dataset = read_dataset(spark, **params)
    ## train
    global_hot = train(spark, train_dataset, **params)
    ## transform
    test_result = transform(spark, global_hot, test_dataset)
    ## evaluate
    recall_metrics = evaluate(spark, test_result)
    topk = params['max_recommendation_count']
    print("Debug -- Precision@%d: %f" % (topk, recall_metrics.precisionAt(topk)))
    print("Debug -- Recall@%d: %f" % (topk, recall_metrics.recallAt(topk)))
    print("Debug -- MAP@%d: %f" % (topk, recall_metrics.meanAveragePrecisionAt(topk)))
    print("Debug -- NDCG@%d: %f" % (topk, recall_metrics.ndcgAt(topk)))

    stop_spark(spark)
