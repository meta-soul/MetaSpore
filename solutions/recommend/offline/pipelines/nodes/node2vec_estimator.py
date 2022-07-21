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

class Node2VecEstimatorNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        training_conf = conf[self._node_conf]
        user_id = conf['dataset']['user_id_column']
        friend_id = conf['dataset']['item_id_column']
        label = conf['dataset']['label_column']
        label_value = conf['dataset']['label_value']
        train_dataset = payload['train_dataset']
        test_dataset = payload['test_dataset']
        
        Node2VecEstimator = get_class(**training_conf['node2vec_estimator_class'])
        
        estimator = Node2VecEstimator(source_vertex_column_name=user_id,
                              destination_vertex_column_name=friend_id,
                              trigger_vertex_column_name=friend_id,
                              behavior_column_name = label,
                              behavior_filter_value = label_value,
                              random_walk_p = 0.5,
                              random_walk_q = 1.0,
                              debug=False)
        
        ## model train
        model = estimator.fit(train_dataset)
        
        # model.df.write.parquet(training_conf['model_out_path'], mode="overwrite")
        payload['df_to_mongodb'] = model.df
        
        ## transform test dataset
        test_result = model.transform(test_dataset)
        
        from pyspark.sql import functions as F 
        str_schema = 'array<struct<name:string,_2:double>>'
        test_result = test_result.withColumn('rec_info', F.col('value').cast(str_schema))
        
        payload['test_result'] = test_result
        
        return payload