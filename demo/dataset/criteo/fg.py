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
from cgi import test
import sys
import yaml
import time
import argparse
import subprocess

sys.path.append('../')
from common.criteo_sparse_features_extractor import read_crieto_files
from common.criteo_sparse_features_extractor import feature_generation

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(**kwargs):
    subprocess.run(['zip', '-r', 'criteo/python.zip', 'common'], cwd='../')
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
        .config("spark.submit.pyFiles", "python.zip")
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

def read_dataset(**kwargs):
    train_dataset = read_crieto_files(s3_root_dir, train_day_count, 'train')
    test_dataset = read_crieto_files(s3_root_dir, test_day_count, 'test')
    return train_dataset, test_dataset

def write_dataset_to_s3(fg_train_dataset, fg_test_dataset, **kwargs):
    train_out_path = output_root_dir + '/train_%d.parquet' % train_day_count
    test_out_path = output_root_dir + '/test_%d.parquet' % test_day_count
    print('Debug write_dataset_to_s3 --train:%s'%train_out_path)
    print('Debug write_dataset_to_s3 --test:%s'%test_out_path)
    fg_train_dataset.write.parquet(train_out_path, mode="overwrite")
    fg_test_dataset.write.parquet(test_out_path, mode="overwrite")
    return True

if __name__=="__main__":
    print('Debug -- Criteo 5D Feature Generation')
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
    train_dataset, test_dataset = read_dataset(**params)

    ## generate sparse features
    fg_train_dataset = feature_generation(train_dataset, verbose)
    fg_test_dataset = feature_generation(test_dataset, verbose)

    ## write to s3
    write_dataset_to_s3(fg_train_dataset, fg_test_dataset, **params)
    
    stop_spark(spark)