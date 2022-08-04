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
from .deepctr_config import *
from ..widedeep_net import WideDeep
from ..deepfm_net import DeepFM

ESTIMATOR_CONFIG = {
    WideDeep: DeepCTREstimatorConfig,
    DeepFM: DeepCTREstimatorConfig
}

MODULE_CONFIG = {
    WideDeep: WideDeepConfig,
    DeepFM: DeepFMConfig
}

def get_estimator_config_class(algo_module_class):
    estimator_config_class = ESTIMATOR_CONFIG.get(algo_module_class, None)
    return estimator_config_class

def get_module_config_class(algo_module_class):
    module_config_class = MODULE_CONFIG.get(algo_module_class, None)
    return module_config_class
