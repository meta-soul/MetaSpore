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
    metric_dict = TwoTowersRetrievalModule(spec['training']).\
        run(dataset_dict['train'], dataset_dict['test'], dataset_dict['item'], worker_count, server_count,
            user_id_column='user_id', item_id_column='friend_id')
    
    # 4. stop spark session
    spark.stop()