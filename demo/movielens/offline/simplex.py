#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import metaspore as ms
import numpy as np
import yaml
import argparse
import sys 

sys.path.append('../../../') 
from python.algos.twotower.simplex import UserModule, ItemModule, SimilarityModule
from python.algos.twotower.simplex import SimpleXAgent

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    import subprocess
    subprocess.run(['zip', '-r', 'demo/movielens/offline/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.submit.pyFiles":"python.zip",
        # "spark.ui.showConsoleProgress": "false",
        # "spark.kubernetes.executor.deleteOnTermination":"false",
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
    
    print('Debug -- match train dataset sample:')
    train_dataset.show(10)
    print('Debug -- match test dataset sample:')
    test_dataset.show(10)
    print('Debug -- match item dataset sample:')
    item_dataset.show(10)

    print('Debug -- train dataset positive count:', train_dataset[train_dataset['label']=='1'].count())
    print('Debug -- train dataset negative count:', train_dataset[train_dataset['label']=='0'].count())
    print('Debug -- test dataset count:', test_dataset.count())
    print('Debug -- item dataset count:', item_dataset.count())

    return train_dataset, test_dataset, item_dataset

def train(spark, train_dataset, item_dataset, **model_params):
    ## init user module, item module, similarity module
    user_module = UserModule(user_column_name, user_combine_schema, interacted_items_combine_schema, \
                            emb_size = vector_embedding_size, \
                            g=g, \
                            alpha=ftrl_learning_rate,\
                            beta=ftrl_smothing_rate,\
                            l1=ftrl_l1_regularization,\
                            l2=ftrl_l2_regularization)
    item_module = ItemModule(item_column_name, item_combine_schema, \
                            emb_size = vector_embedding_size, \
                            alpha=ftrl_learning_rate,\
                            beta=ftrl_smothing_rate,\
                            l1=ftrl_l1_regularization,\
                            l2=ftrl_l2_regularization)
    similarity_module = SimilarityModule(net_dropout=net_dropout)
    ## import two tower module
    import importlib
    module_lib = importlib.import_module(two_tower_module)
    ## init module class
    module_class_ = getattr(module_lib, two_tower_module_class)
    module = module_class_(user_module, item_module, similarity_module)
    ## init estimator class
    estimator_class_ = getattr(module_lib, two_tower_estimator_class)
    estimator = estimator_class_(module = module,
                                item_dataset = item_dataset,
                                item_ids_column_indices = [6],
                                agent_class = SimpleXAgent,
                                retrieval_item_count = 20,
                                metric_update_interval = 500,
                                **model_params)
    ## dnn learning rate
    estimator.updater = ms.AdamTensorUpdater(adam_learning_rate)
    ## model train
    model = estimator.fit(train_dataset)
    ## show model path
    import subprocess
    run_res = subprocess.run(['aws', 's3', 'ls', 
                            's3://dmetasoul-bucket/demo/movielens/model/simplex/model_out/'], 
                            stdout=subprocess.PIPE)
    print('Debug -- model output:')
    print("%s\n"%run_res.stdout.decode("utf-8"))
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

def dump_milvus_item_ids(spark):
    item_ids_path = '%smilvus/item_ids/' % model_out_path
    print('Debug -- read milvus and item id mappings from:', item_ids_path)
    item_ids_dataset = ms.input.read_s3_csv(spark, item_ids_path, delimiter=u'\002')
    item_ids_dataset = item_ids_dataset.withColumnRenamed('_c0', 'milvus_id').withColumnRenamed('_c1', 'item_id')
    print('Debug -- write milvus and item id mappings to:', milvus_item_id_path)
    item_ids_dataset.write.parquet(milvus_item_id_path, mode="overwrite")

if __name__=="__main__":
    print('Debug -- Movielens Recall Demo')
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
    ## dump item ids
    dump_milvus_item_ids(spark)

    stop_spark(spark)
