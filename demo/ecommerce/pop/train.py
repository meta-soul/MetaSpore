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

sys.path.append('../../../') 
from python.algos.pipeline import InitSparkModule, InitSparkConfig
from python.algos.pipeline import DataLoaderModule, DataLoaderConfig
from python.algos.pipeline import PopularRetrievalModule
from python.algos.pipeline import DumpToMongoDBModule, DumpToMongoDBConfig
from python.algos.pipeline import setup_logging

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
    df_to_mongodb, metric_dict = PopularRetrievalModule(spec['training']).run(
        dataset_dict['train'], 
        dataset_dict.get('test')
    )
    # 4. dump to mongo_db
    dumpToMongoDBModule = DumpToMongoDBModule(cattrs.structure(spec['mongodb'], DumpToMongoDBConfig))
    dumpToMongoDBModule.run(df_to_mongodb)
    # 5. stop spark session
    spark.stop()
