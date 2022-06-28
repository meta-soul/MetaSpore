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

class TwoTowersEstimatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        training_conf = payload['conf']['training']
        spark_session_conf = payload['conf']['spark']['session_confs']
        
        UserModule = get_class(**training_conf['user_module_class'])
        ItemModule = get_class(**training_conf['item_module_class'])
        SimilarityModule = get_class(**training_conf['similarity_module_class'])
        TwoTowersRetrievalModule = get_class(**training_conf['two_towers_retrieval_module_class'])
        TwoTowersAgentModule = get_class(**training_conf['two_towers_agent_class'])
        TwoTowersEstimatorModule = get_class(**training_conf['two_towers_estimator_class'])

        ## init user module, item module, similarity module
        user_module = UserModule(training_conf['user_column_name'], \
                                 training_conf['user_combine_schema'], \
                                 emb_size = training_conf['vector_embedding_size'], \
                                 alpha = training_conf['ftrl_learning_rate'], \
                                 beta = training_conf['ftrl_smothing_rate'], \
                                 l1 = training_conf['ftrl_l1_regularization'], \
                                 l2 = training_conf['ftrl_l2_regularization'], \
                                 dense_structure = training_conf['dense_structure'])
        item_module = ItemModule(training_conf['item_column_name'], \
                                 training_conf['item_combine_schema'], \
                                 emb_size = training_conf['vector_embedding_size'], \
                                 alpha = training_conf['ftrl_learning_rate'], \
                                 beta = training_conf['ftrl_smothing_rate'], \
                                 l1 = training_conf['ftrl_l1_regularization'], \
                                 l2 = training_conf['ftrl_l2_regularization'], \
                                 dense_structure = training_conf['dense_structure'])
        similarity_module = SimilarityModule(training_conf['tau'])
        module = TwoTowersRetrievalModule(user_module, item_module, similarity_module)

        estimator = TwoTowersEstimatorModule(module = module,
                                     worker_count = spark_session_conf['worker_count'],
                                     server_count = spark_session_conf['server_count'],
                                     item_dataset = payload['item_dataset'],
                                     agent_class = TwoTowersAgentModule,
                                     **training_conf)
        ## dnn learning rate
        estimator.updater = ms.AdamTensorUpdater(training_conf['adam_learning_rate'])
        ## model train
        model = estimator.fit(payload['train_dataset'])
        
        payload['test_result'] = model.transform(payload['test_dataset'])
        
        return payload