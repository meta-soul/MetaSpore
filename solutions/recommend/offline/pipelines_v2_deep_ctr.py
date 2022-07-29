import sys
import logging
import argparse
import yaml
from pipelines_v2.utils import start_logging
import cattrs
from pipelines_v2.modules import InitSparkModule, InitSparkConfig, DataLoaderModule, DataLoaderConfig, DeepCTRModule, DeepCTRConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    
    spec = dict()
    with open(args.conf, 'r') as stream:
        yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
        spec = yaml_dict['spec']

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger = start_logging(**spec['logging'])
    
    # 1. init spark
    initSparkModule = InitSparkModule(cattrs.structure(spec['spark'], InitSparkConfig), logger)
    print('Debug: ', initSparkModule)
    spark, worker_count, server_count = initSparkModule.run()
    
    # 2. load dataset
    dataLoaderModule = DataLoaderModule(cattrs.structure(spec['dataset'], DataLoaderConfig), spark, logger)
    dataset_dict = dataLoaderModule.run()
    
    # 3. train, predict and evaluate
    deepCTRModule = DeepCTRModule(cattrs.structure(spec['training'], DeepCTRConfig), logger)
    metric_dict = deepCTRModule.run(dataset_dict['train'], dataset_dict['test'], worker_count, server_count)
    
    # 5. stop spark session
    spark.stop()