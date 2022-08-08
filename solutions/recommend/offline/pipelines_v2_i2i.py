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
    # ---start---
    import pyspark.sql.functions as F
    columns = dataset_dict['train'].columns
    if 'friend_id' in columns: # pokec
        for key in dataset_dict:
            df = dataset_dict[key]
            df = df.withColumnRenamed('friend_id', 'item_id')
            df = df.withColumn('last_item_id', F.col('item_id'))
            dataset_dict[key] = df
    elif 'movie_id' in columns: # movielens
        for key in dataset_dict:
            df = dataset_dict[key]
            df = df.withColumnRenamed('movie_id', 'item_id')
            df = df.withColumnRenamed('last_movie', 'last_item_id')
            dataset_dict[key] = df
    elif '205' in columns: # aliccp
        for key in dataset_dict:
            df = dataset_dict[key]
            df = df.withColumn('label', F.lit('1'))
            df = df.withColumnRenamed('101', 'user_id')
            df = df.withColumnRenamed('205', 'item_id')
            df = df.withColumn('last_item_id', F.col('item_id'))
            dataset_dict[key] = df
    dataset_dict['train'].show()
    # ---end---
    
    # 3. train, predict and evaluate
    i2IRetrievalModule = I2IRetrievalModule(spec['training'])
    model_df_to_mongo, metric_dict = i2IRetrievalModule.run(dataset_dict['train'], dataset_dict['test'])
    
    # 4. dump to mongo_db
    dumpToMongoDBModule = DumpToMongoDBModule(cattrs.structure(spec['mongodb'], DumpToMongoDBConfig))
    dumpToMongoDBModule.run(model_df_to_mongo)
    
    # 5. stop spark session
    spark.stop()