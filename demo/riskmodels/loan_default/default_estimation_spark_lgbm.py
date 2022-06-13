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

import pyspark
import subprocess
import yaml
import argparse
import onnxmltools
import lightgbm as lgb
import numpy as np
import pandas as pd
import time

from lightgbm import Booster, LGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

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

def read_dataset(spark, train_path, test_path, 
                 format='csv', header=True, inferSchema=True, **kwargs):
    train_dataset = spark.read.format(format)\
                              .option("header",  header)\
                              .option("inferSchema",  inferSchema)\
                              .load(train_path) 
    print('train dataset sample:')
    train_dataset.show(20, False)
    train_dataset, test_dataset = train_dataset.randomSplit([0.80, 0.20], seed=2020)
    return train_dataset, test_dataset

def get_feature_and_label_cols(dataset, **params):
    label_col = params['label_col']
    feature_cols = [x for x in dataset.columns if x not in [label_col]]
    return feature_cols, label_col

def get_vectorassembler(dataset, feature_cols, features='features', label='label'):
    featurizer = VectorAssembler(
        inputCols = feature_cols,
        outputCol = 'features',
        handleInvalid = 'skip'
    )
    dataset = featurizer.transform(dataset)[label, features]
    return dataset

def train(spark, train_dataset, label_col, **model_params):
    print('Debug -- model hyper params:\n', model_params)
    from synapse.ml.lightgbm import LightGBMClassifier
    model = LightGBMClassifier(isProvideTrainingMetric=True, 
                               featuresCol="features", labelCol=label_col, 
                               isUnbalance=True, 
                               **model_params)
    model = model.fit(train_dataset)
    return model

def evaluate(spark, test_result, label_col):
    evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
    auc = evaluator.evaluate(test_result)
    return auc

def write_dataset_to_s3(eval_dataset, eval_out_path, **kwargs):
    start = time.time()
    eval_dataset.write.parquet(eval_out_path, mode="overwrite")
    print('Debug -- write_dataset_to_s3 cost time:', time.time() - start)

def convert_model(lgbm_model: LGBMClassifier or Booster, input_size: int) -> bytes:
    initial_types = [("input", FloatTensorType([-1, input_size]))]
    onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset = 9)
    return onnx_model   

def get_onnx_model(model, len_data_columns):
    booster_model_str = model.getLightGBMBooster()
    booster_model_str = booster_model_str.modelStr()
    booster_model_str = booster_model_str.get()
    booster = lgb.Booster(model_str = booster_model_str)
    onnx_model = convert_model(booster, len_data_columns)
    return onnx_model

def save_onnx_model(onnx_model, model_onnx_path, **kwargs):
    onnxmltools.utils.save_model(onnx_model,'lightgbm.onnx')
    loaded_model = onnxmltools.utils.load_model('lightgbm.onnx')
    subprocess.run(['aws', 's3', 'cp', 'lightgbm.onnx', model_onnx_path], cwd='./')
    print(type(onnx_model))
    print(type(loaded_model))
    return loaded_model

def load_onnx(onnx_ml):
    onnx_ml = onnx_ml.setDeviceType("CPU") \
                     .setFeedDict({"input": "features"}) \
                     .setFetchDict({"probability": "probabilities", "prediction": "label"}) \
                     .setMiniBatchSize(5000)
    return onnx_ml

def get_onnx_model(model, len_data_columns):
    booster_model_str = model.getLightGBMBooster()
    booster_model_str = booster_model_str.modelStr()
    booster_model_str=booster_model_str.get()
    booster = lgb.Booster(model_str = booster_model_str)
    onnx_model = convert_model(booster, len_data_columns)
    return onnx_model

def onnx_transform(onnx_ml, test_data):
    result = onnx_ml.transform(test_data)
    print("ONNX transform sample:", result.limit(10).toPandas())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    spark = init_spark(**params)

    ## get train and test data
    train_dataset, test_dataset = read_dataset(spark, **params)
    
    ## feature transformation
    feature_cols, label_col = get_feature_and_label_cols(train_dataset, **params)
    train_data = get_vectorassembler(train_dataset, feature_cols=feature_cols, label=label_col)
    print("Debug -- train input features:")
    train_data.show(10, False)
    test_data = get_vectorassembler(test_dataset, feature_cols=feature_cols, label=label_col)
    print("Debug -- test input features:")
    test_data.show(10, False)

    ## fit model and test
    model = train(spark, train_data, label_col, **params['model_params'])

    ## eval the train dataset
    print("Debug -- train sample prediction:")
    predictions = model.transform(train_data)
    predictions.show(10, False)
    auc = evaluate(spark, predictions, label_col)
    print("Debug -- train auc:", auc)

    ## eval the test dataset
    print("Debug -- test sample prediction:") 
    predictions = model.transform(test_data)
    predictions.show(10, False)
    auc = evaluate(spark, predictions, label_col)
    print("Debug -- test auc:", auc)
    write_dataset_to_s3(predictions, **params)

    ## convert model to onnx format
    print("Debug -- transform lightgbm model into ONNX format...") 
    onnx_model = get_onnx_model(model,len(train_dataset.columns)-1)
    loaded_model = save_onnx_model(onnx_model, **params)
    from synapse.ml.onnx import ONNXModel
    onnx_ml = ONNXModel().setModelPayload(loaded_model.SerializeToString())
    print("Model inputs:" + str(onnx_ml.getModelInputs()))
    print("Model outputs:" + str(onnx_ml.getModelOutputs()))
    print("Model type:" )
    print(type(onnx_ml))

    ## predicting using onnx lightgbm model
    onnx_ml = load_onnx(onnx_ml)
    onnx_transform(onnx_ml, test_data)
