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

import sys
import logging
import argparse
import yaml
import cattrs

from pipelines_v2.modules import InitSparkModule, InitSparkConfig, DataLoaderModule, DataLoaderConfig, TwoTowersRetrievalModule
from pipelines_v2 import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    
    spec = dict()
    with open(args.conf, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
        spec = yaml_dict['spec']
        
    setup_logging(**spec['logging'])
    
    # 1. init spark
    initSparkModule = InitSparkModule(cattrs.structure(spec['spark'], InitSparkConfig))    
    spark, worker_count, server_count = initSparkModule.run()
    
    # 2. load dataset
    dataLoaderModule = DataLoaderModule(cattrs.structure(spec['dataset'], DataLoaderConfig), spark)
    dataset_dict = dataLoaderModule.run()
   
    # 3. train, predict and evaluate
    user_id_column = dataset_dict.get('user_id_column')
    item_id_column = dataset_dict.get('item_id_column')
    metric_dict = TwoTowersRetrievalModule(spec['training']).\
        run(dataset_dict['train'], dataset_dict['test'], dataset_dict['item'], worker_count, server_count,
            user_id_column=user_id_column, item_id_column=item_id_column)
    
    # 4. stop spark session
    spark.stop()
