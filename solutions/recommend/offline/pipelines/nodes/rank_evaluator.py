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

class RankEvaluatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        train_result = payload['train_result']
        test_result = payload['test_result']
        
        from pyspark.ml.evaluation import BinaryClassificationEvaluator
        train_evaluator = BinaryClassificationEvaluator()
        test_evaluator = BinaryClassificationEvaluator()
        
        train_auc = train_evaluator.evaluate(train_result)
        test_auc = test_evaluator.evaluate(test_result)
        
        print('Debug -- Train AUC: ', train_auc)
        print('Debug -- Test AUC: ', test_auc)
        
        return payload