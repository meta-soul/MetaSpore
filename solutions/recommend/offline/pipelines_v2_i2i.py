import sys
import logging
import argparse
import yaml
import cattrs

from pipelines_v2.modules import InitSparkModule, InitSparkConfig, DataLoaderModule, DataLoaderConfig, I2IRetrievalModule, I2IRetrievalConfig, DumpToMongoDBModule, DumpToMongoDBConfig
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
    spark, _, _ = initSparkModule.run()
    
    # 2. load dataset
    dataLoaderModule = DataLoaderModule(cattrs.structure(spec['dataset'], DataLoaderConfig), spark)
    dataset_dict = dataLoaderModule.run()
    # the logic below will be removed when FG generate the dataset using some 
    # conventional column names such as 'label', 'user_id', 'last_item_id', etc.
    import pyspark.sql.functions as F
    for key in dataset_dict:
        df = dataset_dict[key]
        df = df.withColumnRenamed('friend_id', 'item_id')
        df = df.withColumn('last_item_id', F.col('item_id'))
        dataset_dict[key] = df
    
    # 3. train, predict and evaluate
    i2IRetrievalModule = I2IRetrievalModule(cattrs.structure(spec['training'], I2IRetrievalConfig))
    model_df_to_mongo, metric_dict = i2IRetrievalModule.run(dataset_dict['train'], dataset_dict['test'])
    
    # 4. dump to mongo_db
    dumpToMongoDBModule = DumpToMongoDBModule(cattrs.structure(spec['mongodb'], DumpToMongoDBConfig))
    dumpToMongoDBModule.run(model_df_to_mongo)
    
    # 5. stop spark session
    spark.stop()