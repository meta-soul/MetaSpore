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

import sys
import yaml
import time
import argparse

from pyspark.sql import SparkSession
from collections import defaultdict

all_field_list = [
    '101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
    '125', '126', '127', '128', '129', '205', '206', '207', '210',
    '216', '508', '509', '702', '853', '301'
]

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
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "64MB")
        .config("spark.network.timeout","500")
        .config("spark.hadoop.mapreduce.outputcommitter.factory.scheme.s3a", "org.apache.hadoop.fs.s3a.commit.S3ACommitterFactory")
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
    train_dataset = spark.read.csv(train_path,  sep=',')
    test_dataset = spark.read.csv(test_path,  sep=',')
    return train_dataset, test_dataset

def get_aliccp_fields():
    all_field_dict = defaultdict(int)
    for i, field_id in enumerate(all_field_list):
        all_field_dict[field_id] = i
    return all_field_list, all_field_dict

def get_aliccp_columns():
    return ['label', 'ctr_label', 'cvr_label'] + all_field_list
    
def transform(row, max_len=10, sep=u'\u0001', default_padding='-1'):
    all_field_list, all_field_dict = get_aliccp_fields()
    output_buffer = [(field_id, []) for field_id in all_field_dict]
    
    ctr_label = 0
    ctr_label = 0
    for key, value in row.asDict().items():
        if key == '_c0': # row number
            continue
        elif key == '_c1':
            ctr_label = int(value)
        elif key == '_c2':
            cvr_label = int(value)
        else:
            if value is None or value =='':
                continue
            else:
                field_id, feature_id = value.strip().split(':')
                if field_id not in all_field_dict:
                    continue
                index = all_field_dict[field_id]
                output_buffer[index][1].append(int(feature_id))
    
    output_list=[]
    output_list.append(str(ctr_label * cvr_label))
    output_list.append(str(ctr_label))
    output_list.append(str(cvr_label))
    for i in range(len(all_field_list)):
        if len(output_buffer[i][1]) == 0:
            output_list.append(default_padding)
        else:
            seqs = output_buffer[i][1]
            if len(output_buffer[i][1]) > max_len:
                seqs = output_buffer[i][1][:max_len]
            output_list.append(sep.join([str(x) for x in seqs]))
    return output_list

def transform_dataset(train_dataset, test_dataset, verbose=False, **kwargs):
    start = time.time()
    fg_train_dataset = train_dataset.rdd.map(lambda x: transform(x)).toDF(get_aliccp_columns())
    fg_test_dataset = test_dataset.rdd.map(lambda x: transform(x)).toDF(get_aliccp_columns())
    print('Debug -- transform_dataset cost time:', time.time() - start)
    if verbose:
        print('Debug -- train dataset count:', fg_train_dataset.count())
        print('Debug -- test dataset count:', fg_test_dataset.count())
    return fg_train_dataset, fg_test_dataset

def write_fg_dataset_to_s3(fg_train_dataset, fg_test_dataset,  train_output_path, test_output_path, verbose=False, **kwargs):
    start = time.time()
    fg_train_dataset.write.parquet(train_output_path, mode="overwrite")
    fg_test_dataset.write.parquet(test_output_path, mode="overwrite")
    print('Debug -- write_fg_dataset_to_s3 cost time:', time.time() - start)
    if verbose:
        print('Debug -- train dataset count:', fg_train_dataset.count())
        print('Debug -- test dataset count:', fg_test_dataset.count())
    return True

if __name__=="__main__":
    print('Debug -- AliCCP Dataset Feature Generation')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=True, help='verbose')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='not verbose')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf, 'verbose:', args.verbose)
    ## init spark
    verbose = args.verbose
    params = load_config(args.conf)
    spark = init_spark(**params)

    ## preprocessing
    train_dataset, test_dataset = read_dataset(spark, **params)

    ## generate sparse features
    fg_train_dataset, fg_test_dataset = transform_dataset(train_dataset, test_dataset, verbose)

    ## write to s3
    write_fg_dataset_to_s3(fg_train_dataset, fg_test_dataset, **params)
    
    stop_spark(spark)

