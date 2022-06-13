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
import yaml
import argparse
import onnxmltools
import subprocess
import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm import Booster, LGBMClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorAssembler
from onnxmltools.convert import convert_lightgbm
from onnxconverter_common.data_types import FloatTensorType

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark():
    spark = pyspark.sql.SparkSession.builder.appName("synapse") \
            .config("spark.executor.memory","5G") \
            .config("spark.executor.memory","5G") \
            .config("spark.executor.instances","4") \
            .getOrCreate()
    sc = spark.sparkContext
    print(sc.version)
    print(sc.uiWebUrl)
    return spark

def get_vectorassembler(train,test):
    feature_cols = train.columns[1:]
    featurizer = VectorAssembler(
        inputCols = feature_cols,
        outputCol = 'features',
        handleInvalid = 'skip'
    )
    train_data = featurizer.transform(train)['label','features']
    test_data = featurizer.transform(test)['label','features']
    return train_data,test_data

def convert_model(lgbm_model: LGBMClassifier or Booster, input_size: int) -> bytes:
    initial_types = [("input", FloatTensorType([-1, input_size]))]
    onnx_model = convert_lightgbm(lgbm_model, initial_types=initial_types, target_opset = 9)
    return onnx_model   

def save_onnx_model(onnx_model):
    onnxmltools.utils.save_model(onnx_model,'lightgbm.onnx')
    loaded_model = onnxmltools.utils.load_model('lightgbm.onnx')
    # aws s3 cp lightgbm.onnx 's3://dmetasoul-bucket/demo/movielens/model/lightgbm_test/lightgbm.onnx'
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
    result.limit(10).toPandas()
    n = 10 * 10
    m = 95
    test = np.random.rand(n, m)
    testPdf = pd.DataFrame(test)
    cols = list(map(str, testPdf.columns))
    testDf = spark.createDataFrame(testPdf)
    testDf = testDf.union(testDf).repartition(200)
    testDf = VectorAssembler().setInputCols(cols).setOutputCol("features").transform(testDf).drop(*cols).cache()
    print('Debug -- onnx model transform test dataframe result:')
    testDf.show(10, False)
    
def read_dataset(train_data_path,test_data_path):
    train = spark.read.format("parquet")\
      .option("header", True)\
      .option("inferSchema", True)\
      .load(train_data_path)

    test = spark.read.format("parquet")\
      .option("header", True)\
      .option("inferSchema", True)\
      .load(test_data_path)
    
    return train,test

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)
    spark = init_spark()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    ## get train and test data
    train, test = read_dataset(train_data_path, test_data_path)
    print("Debug --- train dataset sample:")
    train.show(10, False)
    print("Debug --- test dataset sample:")
    test.show(10, False)
    print("Debug --- train schema:")
    train.printSchema()
    print("Debug --- test schema:")
    test.printSchema()
    
    ## feature transformation
    train_data, test_data = get_vectorassembler(train,test)
    print("Debug --- train input features:")
    train_data.show(10, False)
    print("Debug --- test input features:")
    test_data.show(10, False)

    ## fit model and test
    from synapse.ml.lightgbm import LightGBMClassifier
    model = LightGBMClassifier(objective="binary", featuresCol="features", labelCol="label", isUnbalance=True)
    model = model.fit(train_data)
    print("Debug --- test sample prediction:") 
    predictions = model.transform(test_data)
    predictions.show(10, False)
    print("Debug --- test metrics:") 
    from synapse.ml.train import *
    cms = (ComputeModelStatistics()
          .setLabelCol("label")
          .setScoredLabelsCol("prediction")
          .setEvaluationMetric("classification"))
    print(cms.transform(predictions).toPandas())
    
    ## convert model to onnx format
    onnx_model = get_onnx_model(model,len(train.columns)-1)
    loaded_model = save_onnx_model(onnx_model)
    from synapse.ml.onnx import ONNXModel
    onnx_ml = ONNXModel().setModelPayload(loaded_model.SerializeToString())
    print("Model inputs:" + str(onnx_ml.getModelInputs()))
    print("Model outputs:" + str(onnx_ml.getModelOutputs()))
    print("Model type:" )
    print(type(onnx_ml))

    ## predicting using onnx lightgbm model
    onnx_ml=load_onnx(onnx_ml)
    onnx_transform(onnx_ml, test_data)
