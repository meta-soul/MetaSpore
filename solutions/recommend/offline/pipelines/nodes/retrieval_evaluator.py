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

class RetrievalEvaluatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        test_result = payload['test_result']
        user_id = conf['dataset']['user_id']
        item_id = conf['dataset']['item_id']
        
        from pyspark.sql import functions as F
        print('Debug -- test sample:')
        test_result.select(user_id, (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)
        
        user_id_to_verify = test_result.head(1).collect()[0][user_id]

        test_result[test_result[user_id]==user_id_to_verify]\
                    .select(user_id, (F.posexplode('rec_info').alias('pos', 'rec_info'))).show(60)

        ## evaluation
        from pyspark.mllib.evaluation import RankingMetrics
        prediction_label_rdd = test_result.rdd.map(lambda x:(\
                                                [xx.name for xx in x.rec_info] if x.rec_info is not None else [], \
                                                [getattr(x, item_id)]))

        recall_metrics = RankingMetrics(prediction_label_rdd)

        print("Debug -- Precision@20: ", recall_metrics.precisionAt(20))
        print("Debug -- Recall@20: ", recall_metrics.recallAt(20))
        print("Debug -- MAP@20: ", recall_metrics.meanAveragePrecisionAt(20))
        
        return payload