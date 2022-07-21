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
        random_walk_p = training_conf['random_walk_p']
        random_walk_q = training_conf['random_walk_q']
        random_walk_Z = training_conf['random_walk_Z']
        random_walk_steps = training_conf['random_walk_steps']
        w2v_vector_size = training_conf['w2v_vector_size']
        w2v_window_size = training_conf['w2v_window_size']
        w2v_min_count = training_conf['w2v_min_count']
        w2v_max_iter = training_conf['w2v_max_iter']
        w2v_num_partitions = training_conf['w2v_num_partitions']
        euclid_bucket_length = training_conf['euclid_bucket_length']
        euclid_distance_threshold = training_conf['euclid_distance_threshold']
        train_dataset = payload['train_dataset']
        test_dataset = payload['test_dataset']
        
        Node2VecEstimator = get_class(**training_conf['node2vec_estimator_class'])
        
        estimator = Node2VecEstimator(source_vertex_column_name=user_id,
                                        destination_vertex_column_name=friend_id,
                                        trigger_vertex_column_name=friend_id,
                                        behavior_column_name=label,
                                        behavior_filter_value=label_value,
                                        random_walk_p=random_walk_p,
                                        random_walk_q=random_walk_q,
                                        random_walk_Z=random_walk_Z,
                                        random_walk_steps = random_walk_steps,
                                        w2v_vector_size=w2v_vector_size,
                                        w2v_window_size=w2v_window_size,
                                        w2v_min_count=w2v_min_count,
                                        w2v_max_iter=w2v_max_iter,
                                        w2v_num_partitions=w2v_num_partitions,
                                        euclid_bucket_length=euclid_bucket_length,
                                        euclid_distance_threshold=euclid_distance_threshold,
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