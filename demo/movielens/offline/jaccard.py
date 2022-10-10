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
import subprocess
import yaml
import argparse
import sys

sys.path.append('../../../') 
from python.algos.jaccard_retrieval import JaccardModel, JaccardEstimator

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', 'demo/movielens/offline/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.submit.pyFiles":"python.zip",
        "spark.sql.shuffle.partitions":spark_sql_shuffle_partitions, # default 200
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


def train(spark, train_dataset, item_dataset, **params):
    estimator = JaccardEstimator(user_id_column_name=user_id_column_name,
                                item_id_column_name=item_id_column_name,
                                behavior_column_name=behavior_column_name,
                                behavior_filter_value=behavior_filter_value,
                                key_column_name=key_column_name,
                                value_column_name=value_column_name,
                                max_recommendation_count=max_recommendation_count,
                                jaccard_distance_threshold=jaccard_distance_threshold,
                                debug=use_debug)
    print('Debug -- train item cf model...')
    model = estimator.fit(train_dataset)
    print('Debug -- train model.df:', model.df)
    return model

def transform(spark, model, test_dataset):
    print('Debug -- transform cf model...')
    import pyspark.sql.functions as F
    test_df = test_dataset.select(user_id_column_name, last_item_col_name, item_id_column_name)\
            .groupBy(user_id_column_name, last_item_col_name)\
            .agg(F.collect_set(item_id_column_name).alias('label_items'))
    test_df = test_df.withColumnRenamed(last_item_col_name, item_id_column_name)
    prediction_df = model.transform(test_df)
    prediction_df = prediction_df.withColumnRenamed('value', 'rec_info')
    print('Debug -- transform result sample:')
    prediction_df.show(10)
    return prediction_df

def evaluate(spark, test_result, test_user=100):
    ## evaluation
    from pyspark.mllib.evaluation import RankingMetrics
    prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                    [xx._1 for xx in x.rec_info] if x.rec_info is not None else [], \
                                     x.label_items))
    return RankingMetrics(prediction_label_rdd)

def publish_model_to_s3(spark, model):
    model.df.write.parquet(jaccard_out_path, mode="overwrite")

if __name__=="__main__":
    print('Debug -- Movielens jaccard Demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)
    spark = init_spark()
    ## read datasets
    train_dataset, test_dataset, item_dataset = read_dataset(spark)
    # ## train
    model = train(spark, train_dataset, item_dataset, **params)
    ## transform
    test_result = transform(spark, model, test_dataset)
    ## evaluate
    recall_metrics = evaluate(spark, test_result)
    print("Debug -- Precision@20: ", recall_metrics.precisionAt(20))
    print("Debug -- Recall@20: ", recall_metrics.recallAt(20))
    print("Debug -- MAP@20: ", recall_metrics.meanAveragePrecisionAt(20))
    ## write model
    publish_model_to_s3(spark, model)
    stop_spark(spark)
