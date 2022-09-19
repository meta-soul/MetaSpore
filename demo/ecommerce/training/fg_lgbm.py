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
import pyspark
import subprocess
import yaml
import argparse
import onnxmltools
import lightgbm as lgb
import numpy as np
import time

from lightgbm import Booster, LGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
    print('Debug -- load config: ', params)
    return params

def init_spark(app_name, executor_memory, executor_instances, executor_cores, 
               default_parallelism, **kwargs):
    spark = pyspark.sql.SparkSession.builder\
            .appName(app_name) \
            .master("local") \
            .config("spark.executor.memory", executor_memory) \
            .config("spark.executor.instances", executor_instances) \
            .config("spark.executor.cores", executor_cores) \
            .config("spark.default.parallelism", default_parallelism) \
            .getOrCreate()
    sc = spark.sparkContext
    print(sc.version)
    print(sc.applicationId)
    print(sc.uiWebUrl)
    return spark

def stop_spark(spark):
    print('Debug -- spark stop')
    spark.sparkContext.stop()

def read_dataset(spark, train_path, test_path, label_col, **kwargs):
    def read_data(data_path, label_col):
        # dataset = spark.read.parquet(data_path)
        dataset = spark.read.load(path=data_path, format='parquet')
        dataset = dataset.select(label_col, 'user_id', 'item_id', 'category') 
        dataset = dataset.withColumn(label_col, dataset[label_col].cast('double'))
        return dataset
    train_dataset = read_data(train_path, label_col)
    test_dataset = read_data(test_path, label_col)
                              
    print('train dataset sample:')
    train_dataset.show(20, False)
    print('test dataset sample:')
    test_dataset.show(20, False)
    return train_dataset, test_dataset

def generate_numerical_features(train_dataset: DataFrame, label_col, categorical_cols, **kwargs):
    for col in categorical_cols:
        col_list = col.split('#')
        df_cr_pv = train_dataset.groupBy(*col_list).agg((F.sum(label_col) / F.count(label_col)).alias('_'.join(col_list) + '_cr'), F.sum(label_col).alias('_'.join(col_list) + '_pv'))
        train_dataset = train_dataset.join(df_cr_pv, on=col_list, how='left_outer')
    
    train_dataset = train_dataset.select(train_dataset.colRegex("`^.*_(cr|pv)$|^label$`"))
    print("train_dataset: ", train_dataset.printSchema())
    
    return train_dataset

def save_dataset(spark, train_dataset, train_out_path, test_out_path, **kwargs):
    train_dataset.write.parquet(train_out_path, mode="overwrite")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    spark = init_spark(**params)

    ## get train and test data
    train_dataset, test_dataset = read_dataset(spark, **params)
    
    ## numerical fg
    train_dataset = generate_numerical_features(train_dataset, **params)
    
    ## save dataset
    save_dataset(spark, train_dataset, **params)
    
