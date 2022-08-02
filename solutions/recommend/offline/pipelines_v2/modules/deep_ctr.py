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
import logging
import cattrs

from typing import Dict
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from ..utils import get_class

logger = logging.getLogger(__name__)

    
class DeepCTRModule:
    def __init__(self, conf: dict):
        self.model_params, self.estimator_params = DeepCTRModule.validate(conf)
        self.deep_ctr_model_class = conf['deep_ctr_model_class']
        self.model = None
        
    @staticmethod
    def validate(conf: dict):
        if not 'deep_ctr_model_class' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'deep_ctr_model_class' !")
        if not 'model_params' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'model_params' !")
        if not 'estimator_params' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'estimator_params' !")
            
        module_name = conf['deep_ctr_model_class'].get('module_name', '')
        class_name = conf['deep_ctr_model_class'].get('class_name', '')
        mn_list = module_name.split('.')
        mn_list.insert(len(mn_list)-1, 'config')
        
        mn_list[-1] = mn_list[-1].replace('_net', '_config')
        module_name = '.'.join(mn_list)
        class_name = class_name + 'Config'
        model_config_class = get_class(module_name=module_name, class_name=class_name)
        
        mn_list[-1] = 'deep_ctr_estimator_config'
        module_name = '.'.join(mn_list)
        class_name = 'DeepCTREstimatorConfig'
        estimator_config_class = get_class(module_name=module_name, class_name=class_name)

        model_params = cattrs.structure(conf['model_params'], model_config_class)
        print('Debug - model_params: ', model_params)

        estimator_params = cattrs.structure(conf['estimator_params'], estimator_config_class)
        print('Debug - estimator_params: ', estimator_params)
        
        return model_params, estimator_params
        
    def train(self, train_dataset, worker_count, server_count):
        deepCTRModel = get_class(**self.deep_ctr_model_class)
        module = deepCTRModel(**cattrs.unstructure(self.model_params))
        estimator = ms.PyTorchEstimator(module = module,
                                        worker_count = worker_count,
                                        server_count = server_count,
                                        **cattrs.unstructure(self.estimator_params))
        ## model train
        estimator.updater = ms.AdamTensorUpdater(self.estimator_params.adam_learning_rate)
        self.model = estimator.fit(train_dataset)
        
        logger.info('DeepCTR - training: done')
    
    def evaluate(self, train_result, test_result):
        train_evaluator = BinaryClassificationEvaluator()
        test_evaluator = BinaryClassificationEvaluator()
        
        metric_dict = {}
        metric_dict['train_auc'] = train_evaluator.evaluate(train_result)
        metric_dict['test_auc'] = test_evaluator.evaluate(test_result)
        print('Debug -- metric_dict: ', metric_dict)
        
        logger.info('DeepCTR - evaluation: done')
        return metric_dict
    
    def run(self, train_dataset, test_dataset, worker_count, server_count) -> Dict[str, float]:
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        if not isinstance(test_dataset, DataFrame):
            raise ValueError("Type of test_dataset must be DataFrame.")
        
        # 1. create estimator and fit model
        self.train(train_dataset, worker_count, server_count)
        
        # 2. transform train and test data using self.model
        train_result = self.model.transform(train_dataset)
        test_result = self.model.transform(test_dataset)
        logger.info('DeepCTR - inference: done')
        
        # 3. get metric dictionary (metric name -> metric value)
        metric_dict = self.evaluate(train_result, test_result)
        
        return metric_dict