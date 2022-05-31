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
import pyspark
import numpy as np
import yaml
import subprocess
import argparse
import sys 
from operator import itemgetter

sys.path.append('../../../') 
from python.algos.multitask.mmoe.mmoe_net import MMoE
from python.algos.multitask.mmoe.mmoe_agent import MMoEAgent


def load_config(path):
    params=dict()
    with open(path,'r') as stream:
        params=yaml.load(stream,Loader=yaml.FullLoader)
        print('Debug--load config:',params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', 'demo/multitask/mmoe/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.submit.pyFiles":"python.zip",
    }
    spark_session = ms.spark.get_session(local=local,
                                        app_name=app_name,
                                        batch_size=batch_size,
                                        worker_count=worker_count,
                                        server_count=server_count,
                                        worker_memory=worker_memory,
                                        server_memory=server_memory,
                                        coordinator_memory=coordinator_memory,
                                        spark_confs=spark_confs)
    
    sc = spark_session.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark_session

def stop_spark(spark):
    print('Debug--spark stop')
    spark.sparkContext.stop()

def read_dataset(spark):
    train_dataset=spark.read.csv(train_path, sep=',', inferSchema=False, header=True)
    test_dataset=spark.read.csv(test_path, sep=',', inferSchema=False, header=True)
    print('Debug -- census train dataset sample:')
    train_dataset.show(10)
    print('Debug-- census test dataset sample:')
    test_dataset.show(10)
    print('Debug -- train dataset count:', train_dataset.count())
    print('Debug -- test dataset count:', test_dataset.count())
    return train_dataset, test_dataset
    
def train(spark, trian_dataset, **model_params):
    module = MMoE(embedding_dim=embedding_dim,
                 column_name_path=column_name_path,
                 combine_schema_path=combine_schema_path,
                 expert_numb=expert_numb,
                 task_numb=task_numb,
                 expert_hidden_units=expert_hidden_units,
                 expert_out_dim=expert_out_dim,
                 gate_hidden_units=gate_hidden_units,
                 tower_hidden_units=tower_hidden_units,
                 dnn_activations=dnn_activations,
                 use_bias=use_bias,
                 batch_norm=batch_norm,
                 net_dropout=net_dropout,
                 net_regularizer=net_regularizer,
                 sparse_init_var=sparse_init_var,
                 ftrl_l1=ftrl_l1,
                 ftrl_l2=ftrl_l2,
                 ftrl_alpha=ftrl_alpha,
                 ftrl_beta=ftrl_beta)
    
    estimator = ms.PyTorchEstimator(module=module,
                                  worker_count=worker_count,
                                  server_count=server_count,
                                  agent_class = MMoEAgent,
                                  model_out_path=model_out_path,
                                  # model_export_path=model_export_path,
                                  model_version=model_version,
                                  experiment_name=experiment_name,
                                  input_label_column_indexes=input_label_column_indexes,
                                  metric_update_interval=metric_update_interval,
                                  training_epoches=training_epoches,
                                  shuffle_training_dataset=shuffle_training_dataset)
    model = estimator.fit(trian_dataset)
    estimator.updater = ms.AdamTensorUpdater(adam_learning_rate)
    import subprocess
    run_res = subprocess.run(['aws', 's3', 'ls', model_out_path], 
                            stdout=subprocess.PIPE)
    print('Debug -- model output:')
    print("%s\n"%run_res.stdout.decode("utf-8"))
    return model

def transform(spark, model, test_dataset):
    test_result = model.transform(test_dataset)
    print('Debug -- test result sample:')
    test_result.show(20)
    return test_result

def evaluate(spark, test_result):
    marital_stat_evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator(rawPredictionCol='rawPrediction_0', labelCol='label_0')
    income_50k_evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator(rawPredictionCol='rawPrediction_1', labelCol='label_1')
    marital_stat_auc = marital_stat_evaluator.evaluate(test_result)
    income_50k_auc = income_50k_evaluator.evaluate(test_result)
    return marital_stat_auc, income_50k_auc

if __name__=="__main__":
    print('Debug -- Multi-task Demo MMoE')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)
    print('params: ', params)
    spark = init_spark()
    ## read datasets
    train_dataset, test_dataset = read_dataset(spark)
    ## train
    model = train(spark, train_dataset, **params)
    ## transform
    train_result = transform(spark, model, train_dataset)
    test_result = transform(spark, model, test_dataset)
    ## evaluate
    train_marital_stat_auc, train_income_50k_auc = evaluate(spark, train_result)
    test_marital_stat_auc, test_income_50k_auc = evaluate(spark, test_result)
    print('Debug -- Train marital_stat AUC: ', train_marital_stat_auc, ', Train income_50k AUC: ', train_income_50k_auc)
    print('Debug -- Test marital_stat AUC: ', test_marital_stat_auc, ', Test income_50k AUC: ', test_income_50k_auc)
    stop_spark(spark)