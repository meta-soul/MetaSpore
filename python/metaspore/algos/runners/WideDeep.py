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

from ..models import WideDeep
import metaspore as ms
import pyspark
import numpy as np
import yaml
import subprocess
import argparse
import sys 
from operator import itemgetter

def init_spark(**kwargs):
    spark_session = ms.spark.get_session(local=kwargs['local'],
                                        app_name=kwargs['app_name'],
                                        batch_size=kwargs['batch_size'],
                                        worker_count=kwargs['worker_count'],
                                        server_count=kwargs['server_count'],
                                        worker_memory=kwargs['worker_memory'],
                                        server_memory=kwargs['server_memory'],
                                        coordinator_memory=kwargs['coordinator_memory'],
                                        spark_confs=kwargs['spark_confs']
                                        )
    return spark_session

def stop_spark(spark):
    spark.sparkContext.stop()

def read_dataset(experiment, spark):
    if experiment.get_scheduledTime() is None:      # one-off run
        train_dataset=spark.read.parquet(experiment.get_train_path())
        test_dataset=spark.read.parquet(experiment.get_test_path())
    else:                                           # recurring run -> read from version
        train_dataset=spark.read.parquet(experiment.get_train_path_by_version())
        test_dataset=spark.read.parquet(experiment.get_test_path_by_version())

    return train_dataset, test_dataset
    
def train(experiment, spark, trian_dataset, **kwargs):
    ## init wide and deep model
    module = WideDeep.WideDeep(use_wide=kwargs['use_wide'],
                               wide_embedding_dim=kwargs['embedding_size'],
                               deep_embedding_dim=kwargs['embedding_size'],
                               wide_column_name_path=kwargs['column_name_path'],
                               wide_combine_schema_path=kwargs['wide_combine_schema_path'],
                               deep_column_name_path=kwargs['column_name_path'],
                               deep_combine_schema_path=kwargs['combine_schema_path'],
                               dnn_hidden_units=kwargs['dnn_hidden_units']
                               )
    
    estimator = ms.PyTorchEstimator(module=module,
                                    worker_count=kwargs['worker_count'],
                                    server_count=kwargs['server_count'],
                                    model_out_path=kwargs['model_out_path'],
                                    model_export_path=kwargs['model_export_path'],
                                    model_version=kwargs['model_version'],
                                    experiment_name=kwargs['experiment_name'],
                                    input_label_column_index=kwargs['input_label_column_index'],
                                    metric_update_interval=kwargs['metric_update_interval']
                                    )
    
    experiment.fill_parameter(estimator)
    # notify online
    model = estimator.fit(trian_dataset)
    ## dnn learning rate
    estimator.updater = ms.AdamTensorUpdater(kwargs['adam_learning_rate'])
    return model

def transform(spark, model, test_dataset):
    test_result = model.transform(test_dataset)
    return test_result

def evaluate(spark, test_result):
    evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator()
    auc = evaluator.evaluate(test_result)
    return auc

def experiment_run_me(experiment,**kwargs):
    spark = init_spark(**kwargs)
    ## read datasets
    train_dataset, test_dataset = read_dataset(experiment, spark)
    ## train
    model = train(experiment, spark, train_dataset, **kwargs)
    if test_dataset is not None:
        ## transform
        test_result = transform(spark, model, test_dataset)
        ## evaluate
        test_auc = evaluate(spark, test_result)
    print('Test AUC: ', test_auc)
    stop_spark(spark)