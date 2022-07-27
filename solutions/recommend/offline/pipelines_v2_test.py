import sys
import logging
import argparse
import yaml
from pipelines_v2.utils import start_logging
import cattrs
from pipelines_v2.modules import DataLoaderModule, DataLoaderConfig, InitSparkModule, InitSparkConfig, I2IRetrievalModule, I2IRetrievalConfig, DumpToMongoDBModule, DumpToMongoDBConfig

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
    spark = initSparkModule.run()
    
    # 2. load dataset
    dataLoaderModule = DataLoaderModule(cattrs.structure(spec['dataset'], DataLoaderConfig), spark, logger)
    dataset_dict = dataLoaderModule.run()
    import pyspark.sql.functions as F
    for key in dataset_dict:
        df = dataset_dict[key]
        df = df.withColumnRenamed('friend_id', 'item_id')
        df = df.withColumn('last_item_id', F.col('item_id'))
        dataset_dict[key] = df
    
    # 
    spark.stop()