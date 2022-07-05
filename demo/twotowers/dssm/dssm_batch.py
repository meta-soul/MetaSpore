import numpy as np
import yaml
import argparse
import sys
import subprocess

import metaspore as ms

sys.path.append('../../../') 
from python.algos.twotowers import UserModule, ItemModule, SimilarityModule
from python.algos.twotowers import TwoTowerBatchNegativeSamplingAgent, TwoTowerBatchNegativeSamplingModule

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', 'demo/twotowers/dssm/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.ui.showConsoleProgress": "true",
        "spark.kubernetes.executor.deleteOnTermination":"true",
        "spark.submit.pyFiles":"python.zip",
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

def read_dataset(spark):
    train_dataset = spark.read.parquet(train_path)
    test_dataset = spark.read.parquet(test_path)
    item_dataset = spark.read.parquet(item_path)
  
    train_dataset = train_dataset.filter(train_dataset['label'] == '1')
    test_dataset = test_dataset.filter(test_dataset['label'] == '1')

    print('Debug -- train dataset:')
    train_dataset.show(20)

    print('Debug -- test dataset:')
    test_dataset.show(20)

    print('Debug -- item dataset:')
    item_dataset.show(20)

    return train_dataset, test_dataset, item_dataset

def train(spark, train_dataset, item_dataset, **model_params):
    ## init user module, item module, similarity module
    user_module = UserModule(column_name_path = model_params['user_column_name'], \
                             combine_schema_path = model_params['user_combine_schema'], \
                             embedding_dim = model_params['vector_embedding_size'], \
                             sparse_init_var = model_params['sparse_init_var'], \
                             ftrl_l1 = model_params['ftrl_l1_regularization'], \
                             ftrl_l2 = model_params['ftrl_l2_regularization'], \
                             ftrl_alpha = model_params['ftrl_learning_rate'], \
                             ftrl_beta = model_params['ftrl_smothing_rate'], \
                             dnn_hidden_units = model_params['dnn_hidden_units'], \
                             dnn_hidden_activations = model_params['dnn_hidden_activations'])
    item_module = ItemModule(column_name_path = model_params['item_column_name'], \
                             combine_schema_path = model_params['item_combine_schema'], \
                             embedding_dim = model_params['vector_embedding_size'], \
                             sparse_init_var = model_params['sparse_init_var'], \
                             ftrl_l1 = model_params['ftrl_l1_regularization'], \
                             ftrl_l2 = model_params['ftrl_l2_regularization'], \
                             ftrl_alpha = model_params['ftrl_learning_rate'], \
                             ftrl_beta = model_params['ftrl_smothing_rate'], \
                             dnn_hidden_units = model_params['dnn_hidden_units'], \
                             dnn_hidden_activations = model_params['dnn_hidden_activations'])
    similarity_module = SimilarityModule(model_params['tau'])

    ## init module class
    module = TwoTowerBatchNegativeSamplingModule(user_module, item_module, similarity_module)
    ## import two tower module
    import importlib
    module_lib = importlib.import_module(two_tower_module)
    ## init estimator class
    estimator_class_ = getattr(module_lib, two_tower_estimator_class)
    estimator = estimator_class_(module = module,
                                 item_dataset = item_dataset,
                                 item_ids_column_indices = [6],
                                 retrieval_item_count = 20,
                                 metric_update_interval = 500,
                                 agent_class = TwoTowerBatchNegativeSamplingAgent,
                                 **model_params)
    ## dnn learning rate
    estimator.updater = ms.AdamTensorUpdater(adam_learning_rate)
    ## model train
    model = estimator.fit(train_dataset)
    print('Debug -- traing is completed')
    return model

def transform(spark, model, test_dataset):
    test_result = model.transform(test_dataset)
    print('Debug -- test result sample:')
    test_result.show(20)
    return test_result

def evaluate(spark, test_result, test_user=100):
    from pyspark.sql import functions as F
    print('Debug -- test sample:')
    test_result.select('user_id', (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)
    
    print('Debug -- test user:%d sample:'%test_user)
    test_result[test_result['user_id']==100]\
                .select('user_id', (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)
    
    ## evaluation
    from pyspark.mllib.evaluation import RankingMetrics
    prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                            [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                            [x.movie_id]))
    return RankingMetrics(prediction_label_rdd)

if __name__=="__main__":
    print('Debug -- TwoTowers DSSM In Batch Negative Sampling Demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)

    spark = init_spark()
    ## read datasets
    train_dataset, test_dataset, item_dataset = read_dataset(spark)
    ## train
    model = train(spark, train_dataset, item_dataset, **params)
    ## transform
    test_result = transform(spark, model, test_dataset)
    ## evaluate
    recall_metrics = evaluate(spark, test_result)
    print("Debug -- Precision@20: ", recall_metrics.precisionAt(20))
    print("Debug -- Recall@20: ", recall_metrics.recallAt(20))
    print("Debug -- MAP@20: ", recall_metrics.meanAveragePrecisionAt(20))
    print("Debug -- NDCG@20: ", recall_metrics.ndcgAt(20))
    stop_spark(spark)