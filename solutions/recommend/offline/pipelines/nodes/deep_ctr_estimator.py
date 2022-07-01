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

from .node import PipelineNode
import metaspore as ms
from ..utils import get_class

class DeepCTREstimatorNode(PipelineNode):   
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        training_conf = conf[self._node_conf]
        spark_session_conf = payload['conf']['spark']['session_confs']
        train_dataset = payload['train_dataset']
        test_dataset = payload['test_dataset']
        
        DeepCTRModel = get_class(**training_conf['deep_ctr_model_class'])
        
        module = DeepCTRModel(**training_conf)
        
        estimator = ms.PyTorchEstimator(module = module,
                                        worker_count = spark_session_conf['worker_count'],
                                        server_count = spark_session_conf['server_count'],
                                        **training_conf)
        
        ## model train
        estimator.updater = ms.AdamTensorUpdater(training_conf['adam_learning_rate'])
        model = estimator.fit(train_dataset)
        
        payload['train_result'] = model.transform(train_dataset)
        payload['test_result'] = model.transform(test_dataset)
        
        return payload