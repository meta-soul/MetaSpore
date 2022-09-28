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

import random
import subprocess
import os
import datetime
import copy
import yaml
import argparse
import traceback

import metaspore as ms
import numpy as np


class BaseTuner(object):
    def __init__(self, config):
        self._config = config
        self._num_experiment = config['num_experiment']

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._s3_result_path = os.path.join(config['result_path'], timestamp)
        local_result_path = config.get('local_result_path', '/home/spark/work/tunner_output')
        self._local_result_path = os.path.join(local_result_path, config['model_name'], timestamp)
        subprocess.run(['mkdir', '-p', self._local_result_path])

        self._dataset = {}

    def get_freezed_config(self):
        freezed_hyper_param = {}
        for param_name, param_space in self._config['hyper_param'].items():
            if isinstance(param_space, list):
                freezed_hyper_param[param_name] = random.choice(param_space)
            elif isinstance(param_space, str) and param_space.startswith('eval@'):
                freezed_hyper_param[param_name] = eval(param_space.split('@')[1])
            else:
                freezed_hyper_param[param_name] = param_space

        freezed_config = copy.deepcopy(self._config)
        freezed_config['hyper_param'] = freezed_hyper_param

        # freezed_config = {**freezed_config, **self._config['common_param']}
        print('freezed_hyper_param: ', freezed_hyper_param)
        return freezed_config

    def init_spark_session(self, config):
        app_name = config.get('app_name', 'Demo')
        py_files = config.get('py_files')

        local = config['common_param'].get('local', False)
        worker_count = config['common_param'].get('worker_count', 2)
        server_count = config['common_param'].get('server_count', 2)
        worker_memory = config['common_param'].get('worker_memory', '10G')
        server_memory = config['common_param'].get('server_memory', '10G')
        coordinator_memory = config['common_param'].get('coordinator_memory', '10G')

        batch_size = config['hyper_param'].get('batch_size', 256)
        spark_confs = {}
        spark_confs['spark.network.timeout'] = '500'
        if py_files is not None:
            spark_confs['spark.submit.pyFiles'] = py_files

        self._ss = ms.spark.get_session(local = local,
                                        app_name = app_name,
                                        batch_size = batch_size,
                                        worker_count = worker_count,
                                        server_count = server_count,
                                        worker_memory = worker_memory,
                                        server_memory = server_memory,
                                        coordinator_memory = coordinator_memory,
                                        spark_confs = spark_confs)
        self._sc = self._ss.sparkContext
        print('spark version: ', self._sc.version)
        print('spark url:', self._sc.uiWebUrl)
        print('spark app id: ', self._sc.applicationId)

    def init_dataset(self, config):
        dataset_path_dict = config['dataset']
        for k, v in config['dataset'].items():
            self._dataset[k] = self._ss.read.parquet(v)
        '''
        print('Debug - train count: ', self._dataset['train'].count())
        print('Debug - test count: ', self._dataset['test'].count())
        print('Debug - item count: ', self._dataset['item'].count())
        '''

    def get_estimator(self, config):
        raise NotImplementedError

    def evaluate(self, model):
        raise NotImplementedError

    def export_experiment(self, i, config):
        if 'item_dataset' in config['common_param']:
            del config['common_param']['item_dataset']
        file_name = 'round_' + str(i) + '.yaml'
        self.save_yaml_file(config, file_name)

    def export_summary(self, experiments):
        summary = []
        for exp in experiments:
            summary.append({key: exp[key] for key in ['hyper_param', 'result']})
        if self._experiments_order_by:
            summary.sort(key = lambda x: x['result'][self._experiments_order_by], reverse=self._experiments_order_reverse)
        print('=========================================')
        print('Summary: ', summary)
        print('=========================================')
        file_name = 'summary.yaml'
        self.save_yaml_file(summary, file_name)

    def save_yaml_file(self, content, file_name):
        local_file_full_path = os.path.join(self._local_result_path, file_name)
        s3_file_full_path = os.path.join(self._s3_result_path, file_name)
        with open(local_file_full_path, 'w') as stream:
            yaml.dump(content, stream, default_flow_style=False)
        subprocess.run(['aws', 's3', 'cp', local_file_full_path, s3_file_full_path])

    def run(self):
        if self._experiments_order_by is None:
            raise AssertionError
        if self._experiments_order_reverse is None:
            self._experiments_order_reverse = True
        print('Debug - self._experiments_order_by: ', self._experiments_order_by, \
                     ' self._experiments_order_reverse:', self._experiments_order_reverse)

        experiments = []
        for i in range(self._num_experiment):
            freezed_config = self.get_freezed_config()
            self.init_spark_session(freezed_config)
            self.init_dataset(freezed_config)
            estimator = self.get_estimator(freezed_config)
            print('Debug - estimator: ', estimator)
            try:
                model = estimator.fit(self._dataset['train'])
                metrics = self.evaluate(model)
                freezed_config['result'] = metrics
                self.export_experiment(i, freezed_config)
                experiments.append(freezed_config)
            except Exception as e:
                print('Debug -- catch tuning excepion: ', e)
                traceback.print_exc()
            finally:
                if self._sc is not None:
                    self._sc.stop()
        self.export_summary(experiments)

'''
if __name__ == '__main__':
    subprocess.run(['rm', '-r', '__pycache__/'])
    subprocess.run(['zip', '-r', 'python.zip', '.'])

    parser = argparse.ArgumentParser(description='Tuner information')
    parser.add_argument('-c', dest='config', type=str, help='Path of config file')
    args = parser.parse_args()
    print(args.config)

    #config_path='../config/dssm/dssm_tune.yaml'
    config_path = args.config
    config = dict()
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    tuner = BaseTuner(config)
    tuner.run()
'''
