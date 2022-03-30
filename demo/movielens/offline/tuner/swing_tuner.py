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

import metaspore as ms
import subprocess
import yaml
import argparse
import sys
import pyspark
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics

sys.path.append("../../../../")
from python.algos.tuner.base_tuner import BaseTuner

class SwingTuner(BaseTuner):
    def __init__(self, config):
        super().__init__(config)
        self._experiments_order_by = 'recall_at_20'
        self._experiments_order_reverse = True
        
    def get_estimator(self, config):
        param_dict = {**config['common_param'], **config['hyper_param'], **self._dataset}
        estimator = ms.SwingEstimator(user_id_column_name=param_dict['user_id_column_name'],
                                item_id_column_name=param_dict['item_id_column_name'],
                                behavior_column_name=param_dict['behavior_column_name'],
                                behavior_filter_value=param_dict['behavior_filter_value'],
                                key_column_name=param_dict['key_column_name'],
                                value_column_name=param_dict['value_column_name'],
                                use_plain_weight=param_dict['use_plain_weight'],
                                smoothing_coefficient=param_dict['smoothing_coefficient'],
                                max_recommendation_count=param_dict['max_recommendation_count'])
        ## we need transform test dataframe
        self._test_df = self._dataset['test']
        self._test_df = self._test_df.select(param_dict['user_id_column_name'], param_dict['last_item_col_name'], param_dict['item_id_column_name'])\
                                     .groupBy(param_dict['user_id_column_name'], param_dict['last_item_col_name'])\
                                     .agg(F.collect_set(param_dict['item_id_column_name']).alias('label_items'))
        self._test_df = self._test_df.withColumnRenamed(param_dict['last_item_col_name'], param_dict['item_id_column_name'])
        
        return estimator
    
    def evaluate(self, model):
        prediction_df = model.transform(self._test_df)
        prediction_df = prediction_df.withColumnRenamed('value', 'rec_info')
        prediction_label_rdd = prediction_df.rdd.map(lambda x:(\
                                            [xx._1 for xx in x.rec_info] if x.rec_info is not None else [], \
                                            x.label_items))
        metrics = RankingMetrics(prediction_label_rdd)
        precisionAt20 = metrics.precisionAt(20)
        recallAt20 = metrics.recallAt(20)
        meanAveragePrecisionAt20 = metrics.meanAveragePrecisionAt(20)
        print('=========================================')
        print('Precision@20: ', precisionAt20)
        print('Recall@20: ', recallAt20)
        print('MAP@20: ', meanAveragePrecisionAt20)
        print('=========================================')
        return {'precision_at_20': precisionAt20, \
                'recall_at_20': recallAt20, \
                'map_at_20': meanAveragePrecisionAt20}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuner information')
    parser.add_argument('-conf', dest='config', type=str, help='Path of config file')
    args = parser.parse_args()
    print(args.config)

    config_path = args.config
    config = dict()
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    subprocess.run(['zip', '-r', 'demo/movielens/offline/tuner/python.zip', 'python'], cwd='../../../../')
    tuner = SwingTuner(config)
    tuner.run()
