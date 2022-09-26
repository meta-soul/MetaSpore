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

from functools import reduce
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType

sys.path.append('../../../') 
from python.algos.feature import negative_sampling, gen_user_bhv_seq, gen_numerical_features
from python.algos.pipeline import DumpToMongoDBConfig, DumpToMongoDBModule
from python.algos.pipeline import setup_logging

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        params = params['spec']
        print('Debug -- load config: ', params)
    return params

def boolean_string(string):
    low_string = string.lower()
    if low_string not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return low_string == 'true'

def init_spark(conf):
    session_conf = conf['session_conf']
    extended_conf = conf.get('extended_conf') or {}
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

def load_dataset(spark, conf, debug=False, verbose=False):
    url = f"jdbc:mysql://{conf['host']}/{conf['database']}?useSSL=false&useUnicode=true&characterEncoding=utf8"
    
    user_dataset = spark.read.format(conf['format']).options(
        url =url,
        driver=conf['driver'],
        dbtable=conf['user_table'],
        user=conf['user'],
        password=conf['password']
        ).load()
    print('Debug -- user dataset count:', user_dataset.count())
    
    item_dataset = spark.read.format(conf['format']).options(
        url =url,
        driver=conf['driver'],
        dbtable=conf['item_table'],
        user=conf['user'],
        password=conf['password']
        ).load()
    print('Debug -- item dataset count:', item_dataset.count())
    
    interaction_dataset = spark.read.format(conf['format']).options(
        url =url,
        driver=conf['driver'],
        dbtable=conf['interaction_table'],
        user=conf['user'],
        password=conf['password']
        ).load()
    print('Debug -- interaction dataset count:', interaction_dataset.count())
    if debug:
        print('Debug -- open debug mode, interaction_dataset will be limited to 1000')
        interaction_dataset = interaction_dataset.limit(1000)

    return user_dataset, item_dataset, interaction_dataset

def save_dataset(user_dataset, item_dataset, interaction_dataset, conf, verbose=False):
    user_dataset.write.parquet(conf['user_path'], mode="overwrite")
    item_dataset.write.parquet(conf['item_path'], mode="overwrite")
    interaction_dataset.write.parquet(conf['interaction_path'], mode="overwrite")
    

if __name__=="__main__":
    print('Debug -- Ecommerce Samples Preprocessing')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    parser.add_argument('--debug', type=boolean_string, default=False, help='whether to open debug mode')
    parser.add_argument('--verbose', type=boolean_string, default=False, help='whether to print more debug info, default to False.')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf)
    print('Debug -- verbose:', args.verbose)
    params = load_config(args.conf)
    # init logging
    setup_logging(**params['logging'])
    # init spark
    spark = init_spark(params['init_spark'])
    # load dataset
    user_dataset, item_dataset, interaction_dataset \
        = load_dataset(spark,  params['load_dataset'], debug=args.debug, verbose=args.verbose)
    # save dataset
    save_dataset(user_dataset, item_dataset, interaction_dataset, params['save_dataset'], verbose=args.verbose)
    spark.stop()
