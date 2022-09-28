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
import attrs
import cattrs

from typing import Dict
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from attrs import frozen, field
from attrs.validators import optional, instance_of, gt

from .utils import get_class, remove_none_value
from .utils.constants import *
from .common_validators import array_index_validator, learning_rate_validator, dim_validator, hidden_units_validator, prob_like_validator


logger = logging.getLogger(__name__)

@attrs.frozen
class DeepCTRConfig:
    deep_ctr_model_class: object
    model_config_class: object
    model_params: object
    estimator_config_class: object
    estimator_params: object

class DeepCTRModule:
    def __init__(self, conf):
        if isinstance(conf, dict):
            self.conf = DeepCTRModule.convert(conf)
        elif isinstance(conf, DeepCTRConfig):
            self.conf = conf
        else:
            raise TypeError("Type of 'conf' must be dict or DeepCTRConfig. Current type is {}".format(type(conf)))
        self.model = None

    @staticmethod
    def convert(conf: dict) -> DeepCTRConfig:
        if not 'deep_ctr_model_class' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'deep_ctr_model_class' !")
        if not 'model_params' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'model_params' !")
        if not 'estimator_params' in conf:
            raise ValueError("Dict of DeepCTRModule must have key 'estimator_params' !")

        deep_ctr_model_class = get_class(conf['deep_ctr_model_class'])
        model_config_class = get_class(conf['model_config_class'])
        model_params = cattrs.structure(conf['model_params'], model_config_class)
        estimator_config_class = get_class(conf['estimator_config_class'])
        estimator_params = cattrs.structure(conf['estimator_params'], estimator_config_class)

        return DeepCTRConfig(deep_ctr_model_class, model_config_class, model_params, estimator_config_class, estimator_params)

    def train(self, train_dataset, worker_count, server_count):
        module = self.conf.deep_ctr_model_class(**remove_none_value(cattrs.unstructure(self.conf.model_params)))
        estimator = ms.PyTorchEstimator(module = module,
                                        worker_count = worker_count,
                                        server_count = server_count,
                                        **remove_none_value(cattrs.unstructure(self.conf.estimator_params)))
        ## model train
        estimator.updater = ms.AdamTensorUpdater(self.conf.estimator_params.adam_learning_rate)
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

        # 1. create estimator and fit model
        self.train(train_dataset, worker_count, server_count)

        metric_dict = {}
        if test_dataset:
            if not isinstance(test_dataset, DataFrame):
             raise ValueError("Type of test_dataset must be DataFrame.")
            # 2. transform train and test data using self.model
            train_result = self.model.transform(train_dataset)
            test_result = self.model.transform(test_dataset)
            logger.info('DeepCTR - inference: done')
            # 3. get metric dictionary (metric name -> metric value)
            metric_dict = self.evaluate(train_result, test_result)

        return metric_dict

@frozen(kw_only=True)
class DeepCTREstimatorConfig:
    model_in_path = field(default=None, validator=optional(instance_of(str)))
    model_out_path = field(validator=instance_of(str))
    model_export_path = field(default=None, validator=optional(instance_of(str)))
    model_version = field(validator=instance_of(str))
    experiment_name = field(validator=instance_of(str))
    input_label_column_index = field(validator=array_index_validator)
    metric_update_interval = field(validator=[instance_of(int), gt(0)])
    adam_learning_rate = field(validator=learning_rate_validator)
    training_epoches = field(validator=[instance_of(int), gt(0)])
    shuffle_training_dataset = field(validator=instance_of(bool))

@frozen(kw_only=True)
class WideDeepConfig:
    use_wide = field(validator=instance_of(bool))
    wide_embedding_dim = field(validator=dim_validator)
    deep_embedding_dim = field(validator=dim_validator)
    wide_column_name_path = field(default=None, validator=optional(instance_of(str)))
    wide_combine_schema_path = field(validator=instance_of(str))
    deep_column_name_path = field(default=None, validator=optional(instance_of(str)))
    deep_combine_schema_path = field(validator=instance_of(str))
    dnn_hidden_units = field(validator=hidden_units_validator)
    dnn_hidden_activations = field(validator=instance_of(str))
    use_bias = field(validator=instance_of(bool))
    net_dropout = field(validator=prob_like_validator)
    batch_norm = field(validator=instance_of(bool))
    embedding_regularizer = field(default=None, validator=optional(instance_of(str)))
    net_regularizer = field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = field(validator=instance_of(float))
    ftrl_l2 = field(validator=instance_of(float))
    ftrl_alpha = field(validator=instance_of(float))
    ftrl_beta = field(validator=instance_of(float))

@frozen(kw_only=True)
class DeepFMConfig:
    use_wide = field(validator=instance_of(bool))
    use_dnn = field(validator=instance_of(bool))
    use_fm = field(validator=instance_of(bool))
    wide_embedding_dim = field(validator=dim_validator)
    deep_embedding_dim = field(validator=dim_validator)
    wide_column_name_path = field(default=None, validator=optional(instance_of(str)))
    wide_combine_schema_path = field(default=None, validator=optional(instance_of(str)))
    wide_combine_schema_source = field(default=None, validator=optional(instance_of(str)))
    deep_column_name_path = field(default=None, validator=optional(instance_of(str)))
    deep_combine_schema_path = field(default=None, validator=optional(instance_of(str)))
    deep_combine_schema_source = field(default=None, validator=optional(instance_of(str)))
    sparse_init_var = field(validator=instance_of(float))
    dnn_hidden_units = field(validator=hidden_units_validator)
    dnn_hidden_activations = field(validator=instance_of(str))
    use_bias = field(validator=instance_of(bool))
    net_dropout = field(validator=prob_like_validator)
    batch_norm = field(validator=instance_of(bool))
    embedding_regularizer = field(default=None, validator=optional(instance_of(str)))
    net_regularizer = field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = field(validator=instance_of(float))
    ftrl_l2 = field(validator=instance_of(float))
    ftrl_alpha = field(validator=instance_of(float))
    ftrl_beta = field(validator=instance_of(float))
