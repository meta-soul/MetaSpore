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
from metaspore import SwingEstimator

from .deepctr_config import *
from .i2i_retrieval_config import *
from .twotower_retrieval_config import *

from ..widedeep_net import WideDeep
from ..deepfm_net import DeepFM
from ..item_cf_retrieval import ItemCFEstimator
from ..twotower.dssm import UserModule as DSSMUserModule, ItemModule as DSSMItemModule, SimilarityModule as DSSMSimlarityModule

ESTIMATOR_CONFIG = {
    # Deep CTR
    WideDeep: DeepCTREstimatorConfig,
    DeepFM: DeepCTREstimatorConfig,
    # I2I
    SwingEstimator: SwingEstimatorConfig,
    ItemCFEstimator: ItemCFEstimatorConfig,
    # TwoTowers
    DSSMUserModule: TwoTowerEstimatorConfig,
    DSSMItemModule: TwoTowerEstimatorConfig,
    DSSMSimlarityModule: TwoTowerEstimatorConfig
}

MODEL_CONFIG = {
    # Deep CTR
    WideDeep: WideDeepConfig,
    DeepFM: DeepFMConfig,
    # TwoTowers
    DSSMUserModule: DSSMModelConfig,
    DSSMItemModule: DSSMModelConfig,
    DSSMSimlarityModule: DSSMModelConfig
}

def get_estimator_config_class(algo_estimator_class):
    estimator_config_class = ESTIMATOR_CONFIG.get(algo_estimator_class, None)
    return estimator_config_class

def get_model_config_class(algo_model_class):
    model_config_class = MODEL_CONFIG.get(algo_model_class, None)
    return model_config_class
