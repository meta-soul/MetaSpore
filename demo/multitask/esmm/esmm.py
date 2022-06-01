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

sys.path.append('../../../') 
from python.esmm_net import ESMM
from python.training import ESMMAgent

def load_config(path):
    params=dict()
    with open(path,'r') as stream:
        params=yaml.load(stream,Loader=yaml.FullLoader)
        print('Debug--load config:',params)
    return params

def init_spark():
    subprocess.run(['zip', '-r', 'demo/multitask/esmm/python.zip', 'python'], cwd='../../../')
    spark_confs={
        "spark.network.timeout":"500",
        "spark.submit.pyFiles":"python.zip",
        "spark.ui.showConsoleProgress": "true",
        "spark.kubernetes.executor.deleteOnTermination":"true",
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
    train_dataset=spark.read.parquet(train_path)
    test_dataset=spark.read.parquet(test_path)
    print('Debug -- match train dataset sample:')
    train_dataset.show(10)
    print('Debug--match train test dataset sample:')
    test_dataset.show(10)
    print('Debug -- train dataset positive count:', train_dataset[train_dataset['label']=='1'].count())
    print('Debug -- train dataset negative count:', train_dataset[train_dataset['label']=='0'].count())
    print('Debug -- test dataset positive count:', test_dataset[test_dataset['label']=='1'].count())
    print('Debug -- test dataset negative count:', test_dataset[test_dataset['label']=='0'].count())
    test_dataset = test_dataset.limit(10000000)
    return train_dataset, test_dataset
    
def train(spark, trian_dataset, **model_params):
    ## init wide and deep model
    module = ESMM(embedding_dim=embedding_dim,
                  column_name_path=column_name_path,
                  combine_schema_path=combine_schema_path,
                  sparse_init_var=sparse_init_var,
                  dnn_hidden_units=dnn_hidden_units,
                  dnn_hidden_activations=dnn_hidden_activations,
                  use_bias=use_bias,
                  batch_norm=batch_norm,
                  net_dropout=net_dropout,
                  net_regularizer=net_regularizer,
                  ftrl_l1=ftrl_l1,
                  ftrl_l2=ftrl_l2,
                  ftrl_alpha=ftrl_alpha,
                  ftrl_beta=ftrl_beta)
    
    print('Debug -- ESMM module:', module)
    
    estimator = ms.PyTorchEstimator(module=module,
                                    agent_class=ESMMAgent,
                                    output_ctr_label_column_name='ctr_label',
                                    output_cvr_label_column_name='cvr_label',
                                    output_ctr_prediction_column_name='rawCTRPrediction',
                                    output_cvr_prediction_column_name='rawCVRPrediction',
                                    **model_params)
    model = estimator.fit(trian_dataset)
     ## dnn learning rate
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
    cvr_test_result = test_result.filter(test_result['ctr_label'] == '1')
    print('Debug -- evaluate total sample count:', test_result.count())
    print('Debug -- evaluate cvr sample count:', cvr_test_result.count())

    ctcvr_evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator()
    ctr_evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator(rawPredictionCol='rawCTRPrediction', labelCol='ctr_label')
    cvr_evaluator = pyspark.ml.evaluation.BinaryClassificationEvaluator(rawPredictionCol='rawCVRPrediction', labelCol='cvr_label')
    ctcvr_auc = ctcvr_evaluator.evaluate(test_result)
    ctr_auc = ctr_evaluator.evaluate(test_result)
    cvr_auc = cvr_evaluator.evaluate(cvr_test_result)
    return ctcvr_auc, ctr_auc, cvr_auc

if __name__=="__main__":
    print('Debug -- Multi-task Demo ESMM')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    params = load_config(args.conf)
    locals().update(params)
    spark = init_spark()
    ## read datasets
    train_dataset, test_dataset = read_dataset(spark)
    ## train
    model = train(spark, train_dataset, **params)
    ## transform
    # train_result = transform(spark, model, train_dataset)
    test_result = transform(spark, model, test_dataset)
    ## evaluate
    # train_auc, train_ctr_auc, train_cvr_auc = evaluate(spark, train_result)
    test_auc, test_ctr_auc, test_cvr_auc = evaluate(spark, test_result)
    # print('Debug -- Train AUC: ', train_auc,'Train CTR AUC:', train_ctr_auc, 'Train CVR AUC:', train_cvr_auc)
    print('Debug -- Test AUC: ', test_auc, 'Test CTR AUC:', test_ctr_auc, 'Test CVR AUC:', test_cvr_auc)
    stop_spark(spark)
