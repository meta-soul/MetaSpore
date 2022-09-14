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
from python.algos.feature import negative_sampling
from python.algos.pipeline import DumpToMongoDBConfig, DumpToMongoDBModule
from python.algos.pipeline import setup_logging

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

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

def load_dataset(spark, conf, fmt='parquet'):
    user_path = conf['user_path']
    user_dataset = spark.read.parquet(user_path)
    print('Debug -- user dataset count:', user_dataset.count())
    
    item_path = conf['item_path']
    item_dataset = spark.read.parquet(item_path)
    print('Debug -- item dataset count:', item_dataset.count())
    
    interaction_path = conf['interaction_path']
    interaction_dataset = spark.read.parquet(interaction_path)
    # debug mode
    interaction_dataset.limit(1000)
    print('Debug -- interaction dataset count:', interaction_dataset.count())
    
    return user_dataset, item_dataset, interaction_dataset

def sample_join(spark, user_dataset, item_dataset, interaction_dataset, conf):
    def negsample(interaction_dataset, user_key_col, item_key_col, timestamp_col, sample_ratio):
        neg_sample_df = negative_sampling(
            spark, 
            dataset=interaction_dataset, 
            user_column=user_key_col, 
            item_column=item_key_col, 
            time_column=timestamp_col, 
            negative_item_column='neg_item_id', 
            negative_sample=sample_ratio
        )
        return neg_sample_df

    def merge_negsample(interaction_dataset, negsample_datasst, 
                        user_key_col, item_key_col, timstamp_col):
        negsample_datasst.registerTempTable('negsample_datasst')
        interaction_dataset.registerTempTable('interaction_dataset')
        query = '''
        select '1' as label, {0}, {1}, {2}
        from interaction_dataset
        union all
        select '0' as label, {0}, {1}, 0 as {2}
        from interaction_dataset
        '''.format(
            user_key_col,
            item_key_col,
            timestamp_col
        )
        merge_dataset = spark.sql(query)
        return merge_dataset

    def join_dataset(user_dataset, item_dataset, interaction_dataset, 
                     user_key_col, item_key_col, timestamp_col):
        user_dataset.registerTempTable('user_dataset')
        item_dataset.registerTempTable('item_dataset')
        interaction_dataset.registerTempTable('interaction_dataset')
        query ="""
        select distinct
            label, user.*, item.*, interact.{2}
        from
            interaction_dataset interact
        join
            user_dataset user
        on interact.{0}=user.{0}
        join
            item_dataset item
        on interact.{1}=item.{1}
        """.format(
            user_key_col, 
            item_key_col, 
            timestamp_col
        )
        join_dataset = spark.sql(query)
        return join_dataset
    
    user_key_col = conf['join_on']['user_key']
    item_key_col = conf['join_on']['item_key']
    timestamp_col = conf['join_on']['timestamp']
    if conf.get('negative_sample'):
        sample_ratio = conf['negative_sample']['sample_ratio']
        negsample = negsample(interaction_dataset, user_key_col, item_key_col, timestamp_col, sample_ratio)
        print('Debug -- negative smapling sample result:')
        negsample.show(10) 
        interaction_dataset = merge_negsample(interaction_dataset, negsample, user_key_col, item_key_col, timestamp_col)
        print('Debug -- merge negative sampling result:')
        interaction_dataset.show(20)

    join_dataset = join_dataset(
        user_dataset, item_dataset, interaction_dataset,
        user_key_col, item_key_col, timestamp_col
    )
    print('Debug -- join dataset sample:')
    join_dataset.show(10)
    join_dataset = join_dataset
    return join_dataset 

def gen_features(spark, dataset, conf):
    def reserve_only_cate_features(dataset, array_join_sep=u'\u0001'):
        str_cols = [f.name for f in dataset.schema.fields if isinstance(f.dataType, StringType)]
        array_cols = [f.name for f in dataset.schema.fields if isinstance(f.dataType, ArrayType)]
        for array_col in array_cols:
            dataset = dataset.withColumn(array_col, F.concat_ws(array_join_sep, F.col(array_col)))
        selected_cols = str_cols + array_cols
        print('Debug -- reserve selected cols:', selected_cols)
        filter_dataset = dataset.select(selected_cols)
        return filter_dataset 
    if conf.get('reserve_only_cate_cols'):
        dataset = reserve_only_cate_features(dataset)
        print('Debug -- reserve cate features sample:')
        dataset.show(10) 
    # convert to all columns to string
    dataset = dataset.select(*(F.col(c).cast('string').alias(c) for c in dataset.columns))
    return dataset

def gen_model_samples(spark, dataset, conf_list):
    def train_test_split(dataset, conf):
        test_ratio = conf.get('split_test') or 0.1
        dataset = dataset.withColumn('is_test', F.when(F.rand(seed=2022) < test_ratio , 1).otherwise(0))
        train_dataset = dataset.filter(dataset['is_test']==0).drop('is_test')
        test_dataset = dataset.filter(dataset['is_test']==1).drop('is_test')
        return train_dataset, test_dataset

    def gen_ctr_nn_samples(dataset, conf):
        train_path = conf['train_path']
        test_path = conf.get('test_path')
        if conf.get('split_test'):
            train_dataset, test_dataset = train_test_split(dataset, conf)
        else:
            train_dataset = dataset
        if conf.get('shuffle'):
            train_dataset = train_dataset\
                .withColumn('rnd', F.rand(seed=2022))\
                .orderBy('rnd')\
                .drop('rnd')
        train_dataset.write.parquet(train_path, mode='overwrite')
        if test_path and test_dataset:
            test_dataset.write.parquet(test_path, mode='overwrite')
        return train_dataset, test_dataset 

    def gen_ctr_gbm_samples(dataset, conf):
        train_path = conf['train_path']
        test_path = conf.get('test_path')
        if conf.get('split_test'):
            train_dataset, test_dataset = train_test_split(dataset, conf)
        else:
            train_dataset = dataset
        if conf.get('shuffle'):
            train_dataset = train_dataset\
                .withColumn('rnd', F.rand(seed=2022))\
                .orderBy('rnd')\
                .drop('rnd')
        train_dataset.write.parquet(train_path, mode='overwrite')
        if test_path and test_dataset:
            test_dataset.write.parquet(test_path, mode='overwrite')
        return train_dataset, test_dataset 

    def gen_match_icf_samples(dataset, conf):
        dataset = dataset.filter(dataset['label']=='1')
        train_path = conf['train_path']
        test_path = conf.get('test_path')
        if conf.get('split_test'):
            train_dataset, test_dataset = train_test_split(dataset, conf)
        else:
            train_dataset = dataset
        if conf.get('shuffle'):
            train_dataset = train_dataset\
                .withColumn('rnd', F.rand(seed=2022))\
                .orderBy('rnd')\
                .drop('rnd')
        train_dataset.write.parquet(train_path, mode='overwrite')
        if test_path and test_dataset:
            test_dataset.write.parquet(test_path, mode='overwrite')
        return train_dataset, test_dataset 

    def gen_match_nn_samples(dataset, conf):
        return None, None 

    for conf in conf_list:
        if conf['model_type'] == 'ctr_nn':
            train_dataset, test_dataset = gen_ctr_nn_samples(dataset, conf)
        elif conf['model_type'] == 'ctr_gbm':
            train_dataset, test_dataset = gen_ctr_gbm_samples(dataset, conf)    
        elif conf['model_type'] == 'match_icf':
            train_dataset, test_dataset = gen_match_icf_samples(dataset, conf)
        elif conf['model_type'] == 'match_nn':
            train_dataset, test_dataset = gen_match_nn_samples(dataset, conf)
        else:
            raise ValueError(f"model_type must be one of: 'ctr_nn', 'ctr_gbm', 'match_icf', 'match_nn'; {conf['model_type']!r} is invalid")
        if train_dataset:
            print('Debug -- generate train samples for {}:'.format(conf['model_type']))
            train_dataset.show(20)
        if test_dataset:
            print('Debug -- generate test samples for {}:'.format(conf['model_type']))
            test_dataset.show(20)

def dump_feature_mongo(spark, fg_samples, conf):
    if not conf.get('mongodb'):
        raise ValueError(f"mongodb should not be None")
    if not conf.get('tables'):
        raise ValueError(f"feature table list should not be None")
    mongo_conf = conf['mongodb']
    feature_table_conf_list = conf['tables']
    dumper = DumpToMongoDBModule(cattrs.structure(mongo_conf, DumpToMongoDBConfig))
    for table_conf in feature_table_conf_list:
        df_to_mongo = fg_samples.select(table_conf['feature_column'])
        mongo_collection = table_conf['mongo_collection']
        dumper.run(df_to_mongo, mongo_collection)

if __name__=="__main__":
    print('Debug -- Ecommerce Samples Preprocessing')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf)
    params = load_config(args.conf)
    # init logging
    setup_logging(**params['logging'])
    # init spark
    spark = init_spark(params['init_spark'])
    # load datasets
    user_dataset, item_dataset, interaction_dataset \
        = load_dataset(spark,  params['load_dataset'])
    raw_samples = sample_join(spark, user_dataset, item_dataset, interaction_dataset, params['join_dataset'])
    fg_samples = gen_features(spark, raw_samples, params['gen_feature'])
    gen_model_samples(spark, fg_samples, params['gen_sample'])
    dump_feature_mongo(spark, fg_samples, params['dump_feature'])
    