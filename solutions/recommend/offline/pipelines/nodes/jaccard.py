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

class JaccardNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        conf = payload['conf']
        recall_conf = conf[self._node_conf]
        logger = start_logging(**conf['logging'])
        user_id = conf['dataset']['user_id_column']
        item_id = conf['dataset']['item_id_column']
        label = conf['dataset']['label_column']
        label_value = conf['dataset']['label_value']
        train_dataset = payload['train_dataset']
        test_dataset = payload.get('test_dataset', None)
        max_recommendation_count = recall_conf['max_recommendation_count']
        
        ## calculate the recall result
        recall_result = self.calculate(train_dataset, user_id, item_id,  label, label_value, max_recommendation_count)
        logger.info('recall_result: {}'\
                     .format(recall_result.show(10)))
        payload['df_to_mongodb'] = recall_result
        
        ## transfrom the test_dataset
        if test_dataset:
            ## friend_id is the trigger item
            cond = [test_dataset[item_id]==recall_result['friend_A']]
            test_result = test_dataset.join(recall_result, on=cond, how='left')
            str_schema = 'array<struct<name:string,_2:double>>'
            test_result = test_result.withColumn('rec_info', F.col('value_list').cast(str_schema))
            payload['test_result'] = test_result
        return payload
    
    def calculate(self, relationship_data, user_id, item_id, label, label_value, max_recommendation_count=20):
        relationship_data = relationship_data.filter(F.col(label)==label_value)\
                                .groupBy(item_id)\
                                .agg(F.collect_list(user_id)\
                                .alias('user_list'))
        ## 'user_list' column must be array<string> type
        cv = CountVectorizer(inputCol='user_list', outputCol='features')
        model_cv = cv.fit(relationship_data)
        cv_result = model_cv.transform(relationship_data)
        mh = MinHashLSH(inputCol='features', outputCol='hashes')
        model_mh = mh.fit(cv_result)
        jaccard_dist_table = model_mh.approxSimilarityJoin(cv_result, cv_result, 1, distCol='jaccard_dist')\
                                                .select(F.col('datasetA.'+item_id).alias('friend_A'),\
                                                        F.col('datasetB.'+item_id).alias('friend_B'),\
                                                        F.col('jaccard_dist'))
        jaccard_sim_table = jaccard_dist_table.withColumn('jaccard_sim', 1-F.col('jaccard_dist')).drop('jaccard_dist')
        jaccard_sim_table = jaccard_sim_table.filter(F.col('jaccard_sim') != 0)
        jaccard_sim_table = jaccard_sim_table.filter(F.col('friend_A') != F.col('friend_B'))
        jaccard_sim_table = jaccard_sim_table.withColumn('value', F.struct('friend_B', 'jaccard_sim'))
        w = Window.partitionBy('friend_A').orderBy(F.desc('jaccard_sim'))
        recall_result = jaccard_sim_table.withColumn('rn',F.row_number()\
                            .over(w))\
                            .filter(f'rn <= %d' % max_recommendation_count)\
                            .groupby('friend_A')\
                            .agg(F.collect_list('value').alias('value_list'))
        return recall_result
