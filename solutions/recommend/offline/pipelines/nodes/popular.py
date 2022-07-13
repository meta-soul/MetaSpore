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
from ..utils import start_logging
from pyspark.sql import Window, functions as F

class PopularNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        recall_conf = conf[self._node_conf]
        logger = start_logging(**conf['logging'])
        recall_nums = recall_conf['recall_nums']
        group_nums = recall_conf['group_nums']
        user_id = conf['dataset']['user_id_column']
        item_id = conf['dataset']['item_id_column']
        label = conf['dataset']['label_column']
        label_value = conf['dataset']['label_value']
        train_dataset = payload['train_dataset']
        test_dataset = payload.get('test_dataset', None)
        
        ## calculate the recall result
        recall_result = self.calculate(train_dataset, label, label_value, item_id, user_id, group_nums, recall_nums)
        logger.info('recall_result: {}'\
                     .format(recall_result.show(10)))
        payload['df_to_mongodb'] = recall_result
        
        ## transfrom the test_dataset
        if test_dataset:
            recall_result = recall_result.filter(F.col('key')==0)        
            test_result = test_dataset.join(recall_result.select('value_list'), None, 'full') 
            str_schema = 'array<struct<name:string,_2:double>>'
            test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
            payload['test_result'] = test_result
        return payload
    
    def calculate(self, train_dataset, label, label_value, item_id, user_id, group_nums, recall_nums):
        recall_result = train_dataset.filter(F.col(label)==label_value) \
                            .groupBy(item_id)\
                            .agg(F.countDistinct('user_id'))\
                            .sort(F.col('count('+user_id+')').desc())\
                            .limit(group_nums * recall_nums) 
        recall_result = recall_result.withColumn('key', F.floor(F.rand() * group_nums))
        
        ## sort according to count value in each group
        recall_result = recall_result.withColumn('rank', F.dense_rank().over(
                            Window.partitionBy('key').orderBy(F.col('count('+user_id+')'))))
        
        ## calculate the score
        recall_result = recall_result.withColumn('score', 1 / (1 + F.col('rank')))\
                            .drop(F.col('rank'))\
                            .drop(F.col('count('+user_id+')'))        
        recall_result = recall_result.withColumn('value', F.struct(item_id, 'score'))\
                            .drop(F.col(item_id))\
                            .drop(F.col('score'))
        recall_result = recall_result.groupBy('key').agg(F.collect_list('value').alias('value_list'))
        return recall_result
        