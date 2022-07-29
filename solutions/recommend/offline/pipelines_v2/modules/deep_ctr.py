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

from logging import Logger

import metaspore as ms
from ..utils import get_class
import attrs
from typing import Dict, Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator

@attrs.frozen
class DeepCTRConfig:
    deep_ctr_model_class = attrs.field(validator=attrs.validators.instance_of(Dict))
    model_params = attrs.field(validator=attrs.validators.instance_of(Dict))
    estimator_params = attrs.field(validator=attrs.validators.instance_of(Dict))
    
class DeepCTRModule():
    def __init__(self, conf: DeepCTRConfig, logger: Logger):
        self.conf = conf
        self.logger = logger
        self.model = None
        
    def train(self, train_dataset, worker_count, server_count):
        deepCTRModel = get_class(**self.conf.deep_ctr_model_class)
        module = deepCTRModel(**self.conf.model_params)
        estimator = ms.PyTorchEstimator(module = module,
                                        worker_count = worker_count,
                                        server_count = server_count,
                                        **self.conf.estimator_params)
        ## model train
        estimator.updater = ms.AdamTensorUpdater(self.conf.estimator_params['adam_learning_rate'])
        self.model = estimator.fit(train_dataset)
        
        self.logger.info('DeepCTR - training: done')
    
    def evaluate(self, test_result):
        train_evaluator = BinaryClassificationEvaluator()
        test_evaluator = BinaryClassificationEvaluator()
        
        metric_dict = {}
        metric_dict['train_auc'] = train_evaluator.evaluate(train_result)
        metric_dict['test_auc'] = test_evaluator.evaluate(test_result)
        print('Debug -- metric_dict: ', metric_dict)
        
        self.logger.info('DeepCTR - evaluation: done')
        return metric_dict
    
    def run(self, train_dataset, test_dataset, worker_count, server_count) -> :
        if not isinstance(train_dataset, DataFrame):
            raise ValueError("Type of train_dataset must be DataFrame.")
        if not isinstance(test_dataset, DataFrame):
            raise ValueError("Type of test_dataset must be DataFrame.")
        
        # 1. create estimator and fit model
        self.train(train_dataset, worker_count, server_count)
        
        # 2. transform train and test data using self.model
        train_result = model.transform(train_dataset)
        test_result = model.transform(test_dataset)
        self.logger.info('DeepCTR - inference: done')
        
        # 3. get metric dictionary (metric name -> metric value)
        metric_dict = self.evaluate(train_result, test_result)
        
        return metric_dict