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

class I2IEstimatorNode(PipelineNode):
    def preprocess(self, **payload) -> dict:
        from pyspark.sql import functions as F 
        test_dataset = payload['test_dataset']
        conf = payload['conf']
        item_id = conf['dataset']['item_id_column']
        last_item_id = conf['dataset']['last_item_id_column']
        test_dataset = test_dataset.withColumn(last_item_id, F.col(item_id))
        payload['test_dataset'] = test_dataset
        return payload
    
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        training_conf = conf['training']
        user_id = conf['dataset']['user_id_column']
        item_id = conf['dataset']['item_id_column']
        label = conf['dataset']['label_column']
        label_value = conf['dataset']['label_value']
        last_item_id = conf['dataset']['last_item_id_column']
        train_dataset = payload['train_dataset']
        test_dataset = payload['test_dataset']
        
        I2IEstimatorModule = get_class(training_conf['i2i_estimator_class'])
        
        estimator = I2IEstimatorModule(user_id_column_name = user_id,
                                       item_id_column_name = item_id,
                                       behavior_column_name = label,
                                       behavior_filter_value = label_value,
                                       max_recommendation_count = training_conf['max_recommendation_count'])
        
        ## model train
        model = estimator.fit(train_dataset)
        
        model.df.write.parquet(training_conf['model_out_path'], mode="overwrite")
        
        ## prepare trigger item id 
        original_item_id ='original_item_id'
        test_df = test_dataset.withColumnRenamed(item_id, original_item_id)
        test_df = test_df.withColumnRenamed(last_item_id, item_id)
        
        ## transform test dataset
        test_result = model.transform(test_df)
        
        ## revert original item id
        test_result = test_result.withColumnRenamed(item_id, last_item_id)
        test_result = test_result.withColumnRenamed(original_item_id, item_id)
        
        from pyspark.sql import functions as F 
        str_schema = "array<struct<name:string,_2:double>>"
        test_result = test_result.withColumn('rec_info', F.col("value").cast(str_schema))
        
        payload['test_result'] = test_result
        
        return payload