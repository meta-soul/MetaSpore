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

import os
import sys
import yaml
import time
import argparse
import subprocess
import metaspore as ms
import cattrs

sys.path.append('../../../') 

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType
from python.algos.graph import Node2VecEstimator

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'

def init_spark(conf):
    session_conf = conf['session_confs']
    extended_conf = conf.get('extended_confs') or {}
    if conf.get('pyzip'):
        pyzip_conf = conf['pyzip']
        cwd_path = pyzip_conf['cwd_path']
        zip_file_path = os.getcwd() + '/python.zip'
        subprocess.run(['zip', '-r', zip_file_path, 'python'], cwd=cwd_path)
        extended_conf['spark.submit.pyFiles'] = 'python.zip'
    spark = ms.spark.get_session(
        local=session_conf['local'],
        app_name=session_conf['app_name'] or 'metaspore',
        batch_size=session_conf['batch_size'] or 100,
        worker_count=session_conf['worker_count'] or 1,
        server_count=session_conf['server_count'] or 1,
        worker_cpu=session_conf.get('worker_cpu') or 1,
        server_cpu=session_conf.get('server_cpu') or 1,
        worker_memory=session_conf['worker_memory'] or '5G',
        server_memory=session_conf['server_memory'] or '5G',
        coordinator_memory=session_conf['coordinator_memory'] or '5G',
        spark_confs=extended_conf)
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark

def load_dataset(spark, conf, fmt='parquet', debug=False, verbose=False):
    train_dataset = spark.read.parquet(conf['train_path'])
    print('Debug -- train dataset count:', train_dataset.count())
    
    test_dataset = spark.read.parquet(conf['test_path'])
    print('Debug -- test dataset count:', test_dataset.count())
    
    return train_dataset, test_dataset

def train(spark, train_dataset, **params):
    estimator = Node2VecEstimator(source_vertex_column_name=params['user_id'],
                                    destination_vertex_column_name=params['friend_id'],
                                    trigger_vertex_column_name=params['user_id'],
                                    behavior_column_name=params['label_column'],
                                    behavior_filter_value=params['label_value'],
                                    max_recommendation_count=params['max_recommendation_count'],
                                    max_out_degree=params['max_out_degree'],
                                    random_walk_p=params['random_walk_p'],
                                    random_walk_q=params['random_walk_q'],
                                    random_walk_Z=params['random_walk_Z'],
                                    random_walk_steps=params['random_walk_steps'],
                                    walk_times=params['walk_times'],
                                    w2v_vector_size=params['w2v_vector_size'],
                                    w2v_window_size=params['w2v_window_size'],
                                    w2v_min_count=params['w2v_min_count'],
                                    w2v_max_iter=params['w2v_max_iter'],
                                    w2v_num_partitions=params['w2v_num_partitions'],
                                    euclid_bucket_length=params['euclid_bucket_length'],
                                    euclid_distance_threshold=params['euclid_distance_threshold'],
                                    debug=params['debug'])
    print('Debug -- train node2vec model...')
    model = estimator.fit(train_dataset)
    print('Debug -- train model.df:', model.df)
    model.df.show(20, False)
    return model
'''
def transform(spark, model, test_dataset, user_id_column_name, last_item_col_name, 
              item_id_column_name, **kwargs):
    print('Debug -- transform swing model...')
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
    print('Debug -- test sample:')
    test_result.select('user_id', 'rec_info').show(10)
    print('Debug -- test user:%d sample:' % test_user)
    test_result[test_result['user_id']==100].select('user_id', 'rec_info').show(10)

    prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                    [xx._1 for xx in x.rec_info] if x.rec_info is not None else [], \
                                     x.label_items))
    return RankingMetrics(prediction_label_rdd)
'''
if __name__=="__main__":
    print('Debug -- Node2Vec Demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    parser.add_argument('--debug', type=boolean_string, default=False, help='whether to open debug mode')
    parser.add_argument('--verbose', type=boolean_string, default=False, help='whether to print more debug info, default to False.')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf)
    print('Debug -- verbose:', args.verbose)
    params = load_config(args.conf)
    # init logging
    # setup_logging(**params['logging'])
    # init spark
    spark = init_spark(params['spark'])
    # load datasets
    train_dataset, test_dataset = load_dataset(spark,  params['dataset'], debug=args.debug, verbose=args.verbose)
    # fit model
    model = train(spark, train_dataset, **params['training'])
    spark.stop()