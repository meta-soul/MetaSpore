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
from pyspark.ml.feature import CountVectorizer, MinHashLSH

class FriendsOfFriendsNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        recall_conf = conf[self._node_conf]
        logger = start_logging(**conf['logging'])
        user_id = conf['dataset']['user_id_column']
        item_id = conf['dataset']['item_id_column']
        time = conf['dataset'].get('time_column')
        time_format = conf['dataset'].get('time_format')
        label = conf['dataset']['label_column']
        label_value = conf['dataset']['label_value']
        train_dataset = payload['train_dataset']
        test_dataset = payload.get('test_dataset', None)
        max_recommendation_count = recall_conf['max_recommendation_count']
        decay_max = recall_conf['decay_max']        
        ## calculate the recall result
        u2f_table = self.u2f_table(train_dataset, user_id, item_id, label, label_value, time, time_format, decay_max)
        recall_result = self.u2fof_table(u2f_table, user_id, item_id, max_recommendation_count)
        logger.info('recall_result: {}'\
                     .format(recall_result.show(10)))
        payload['df_to_mongodb'] = recall_result        
        ## transfrom the test_dataset
        if test_dataset:
            test_result = self.test_transform(test_dataset, recall_result, user_id)
            payload['test_result'] = test_result
        return payload
    
    def u2f_table(self, dataset, user_id, item_id, label, label_value, \
                  last_act_time, time_format='yyyy-MM-dd HH:mm:ss',decay_max=10000,\
                  decay_factor=-0.3):
        dataset = dataset.filter(F.col(label)==label_value)
        item_count = dataset.groupBy(F.col(user_id))\
                            .agg(F.countDistinct(item_id).alias('friend_count'))
        rel_score = dataset.alias('ta').join(item_count.alias('tb'), F.col('ta.'+user_id)==F.col('tb.'+user_id),\
                                             how='inner')\
                                       .select('ta.*', 'tb.friend_count')\
                                       .withColumn('rel_score', F.lit(1)/F.sqrt(F.col('tb.friend_count')))
        if last_act_time:
            rel_score = rel_score.withColumn('day_diff', F.datediff(F.to_date(F.current_timestamp()),\
                                                                    F.to_date(last_act_time, time_format)))
            rel_score = rel_score.withColumn('day_diff',  F.when(F.col('day_diff')>F.lit(decay_max),\
                                                                 F.lit(decay_max)).otherwise(F.col('day_diff')))
            rel_score = rel_score.withColumn('rel_score', F.col('rel_score') * F.pow(F.col('day_diff'),decay_factor))
        return rel_score

    def u2fof_table(self, u2f_table, user_id, item_id, walk_length=1, filter_u2f=True, max_recommendation_count=100):
        u2fof_table = u2f_table
        for i in range(0, walk_length):
            u2fof_table = u2fof_table.alias('ta').join(u2f_table.alias('tb'), F.col('ta.'+item_id)==F.col('tb.'+user_id),\
                                                       how='inner') \
                                                 .filter(F.col('ta.'+user_id)!=F.col('tb.'+item_id)) \
                                                 .groupBy(F.col('ta.'+user_id), F.col('tb.'+item_id)) \
                                                 .agg(F.sum(F.col('ta.rel_score') * F.col('tb.rel_score')).alias('rel_score'))
        if filter_u2f:
            on_cond = (F.col('ta.'+user_id)==F.col('tb.'+user_id))&(F.col('ta.'+item_id)==F.col('tb.'+item_id))
            u2fof_table = u2fof_table.alias('ta').join(u2f_table.alias('tb'), on=on_cond, how='left_outer') \
                                     .select(F.col('ta.*'))\
                                     .filter(F.col('tb.'+item_id).isNull())

        u2fof_table =u2fof_table.withColumn('rn',F.row_number().over\
                                 (Window.partitionBy(user_id).orderBy(F.desc('rel_score')))) \
                                 .filter(f'rn <= %d'%max_recommendation_count)  \
                                 .groupBy(user_id) \
                                 .agg(F.collect_list(F.struct(F.col(item_id), F.col('rel_score'))).alias('value_list')) \
                                 .withColumnRenamed(user_id, 'key')
        return u2fof_table
    
    def test_transform(self, test_dataset, recall_result, user_id):
        ## friend_id is the trigger item
        cond = test_dataset[user_id]==recall_result['key']
        test_result = test_dataset.join(recall_result, on=cond, how='left')
        str_schema = 'array<struct<name:string,_2:double>>'
        test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
        return test_result
