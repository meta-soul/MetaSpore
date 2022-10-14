import sys
import pandas as pd
import metaspore as ms
import argparse
import subprocess
import yaml

from pyspark.sql import Window, functions as F
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def load_config(path):
    params = dict()
    with open(path, 'r') as stream:
        params = yaml.load(stream, Loader=yaml.FullLoader)
        print('Debug -- load config: ', params)
    return params

def init_spark(conf):
    session_conf = conf['session_confs']
    extended_conf = conf.get('extended_confs') or {}
    if conf.get('pyzip'):
        pyzip_conf = conf['pyzip']
        cwd_path = pyzip_conf['cwd_path']
        zip_file_path = os.getcwd() + '/python.zip'
        subprocess.run(['zip', '-r', zip_file_path, 'python'], cwd=cwd_path)
        extended_conf['spark.submit.pyFiles'] = 'python.zip'
    spark = ms.spark.get_session(
        local=session_conf['local'],
        app_name=session_conf['app_name'] or 'metaspore',
        batch_size=session_conf['batch_size'] or 100,
        worker_count=session_conf['worker_count'] or 1,
        server_count=session_conf['server_count'] or 1,
        worker_cpu=session_conf.get('worker_cpu') or 1,
        server_cpu=session_conf.get('server_cpu') or 1,
        worker_memory=session_conf['worker_memory'] or '5G',
        server_memory=session_conf['server_memory'] or '5G',
        coordinator_memory=session_conf['coordinator_memory'] or '5G',
        spark_confs=extended_conf)
    sc = spark.sparkContext
    print('Debug -- spark init')
    print('Debug -- version:', sc.version)   
    print('Debug -- applicaitonId:', sc.applicationId)
    print('Debug -- uiWebUrl:', sc.uiWebUrl)
    return spark


def load_dataset(spark, profile_path, relationship_path, profile_limit, **kwargs):
    profile_colunm_names = ['user_id', 'public', 'completion_percentage', 'gender', 'region', 'last_login', 'registration',
                    'AGE', 'body', 'I_am_working_in_field', 'spoken_languages', 'hobbies', 'I_most_enjoy_good_food',
                    'pets', 'body_type', 'my_eyesight', 'eye_color', 'hair_color', 'hair_type', 'completed_level_of_education',
                    'favourite_color', 'relation_to_smoking', 'relation_to_alcohol', 'sign_in_zodiac',
                    'on_pokec_i_am_looking_for', 'love_is_for_me', 'relation_to_casual_sex', 'my_partner_should_be',
                    'marital_status', 'children', 'relation_to_children', 'I_like_movies', 'I_like_watching_movie',
                    'I_like_music', 'I_mostly_like_listening_to_music', 'the_idea_of_good_evening', 'I_like_specialties_from_kitchen',
                    'fun', 'I_am_going_to_concerts', 'my_active_sports', 'my_passive_sports', 'profession', 'I_like_books',
                    'life_style', 'music', 'cars', 'politics', 'relationships', 'art_culture', 'hobbies_interests',
                    'science_technologies', 'computers_internet', 'education', 'sport', 'movies', 'travelling', 'health',
                    'companies_brands', 'more']
    relationship_colunm_names = ['user_id', 'friend_id']

    profile_schema = StructType([StructField(cn, StringType(), True) for cn in profile_colunm_names])
    relationship_schema = StructType([StructField(cn, LongType(), True) for cn in relationship_colunm_names])

    profile_dataset = spark.read.csv(profile_path, sep='\t', schema=profile_schema, header=False, inferSchema=False)
    relationship_dataset = spark.read.csv(relationship_path, sep='\t', schema=relationship_schema, header=False, inferSchema=False)

    profile_dataset = profile_dataset.withColumn('user_id', F.col('user_id').cast(LongType()))
    profile_dataset = profile_dataset.orderBy(F.col('user_id')).limit(profile_limit)
    max_user_id = profile_dataset.agg({"user_id": "max"}).collect()[0]['max(user_id)']
    relationship_dataset = relationship_dataset.filter((F.col('user_id') <= max_user_id) & (F.col('friend_id') <= max_user_id))

    profile_dataset = profile_dataset.withColumn('user_id', F.col('user_id').cast(StringType()))
    relationship_dataset = relationship_dataset.withColumn('user_id', F.col('user_id').cast(StringType()))
    relationship_dataset = relationship_dataset.withColumn('friend_id', F.col('friend_id').cast(StringType()))

    return profile_dataset, relationship_dataset

def split_train_test(profile_dataset, relationship_dataset, test_ratio, random_seed, **kwargs):
    relationship_df = relationship_dataset.alias('t1').join(profile_dataset.alias('t2'), on=F.col('t1.user_id')==F.col('t2.user_id'), how='leftouter')\
                    .select(F.col('t1.*'),
                            F.col('t2.gender').alias('user_gender'),
                            F.col('t2.AGE').alias('user_age'),
                            F.col('t2.completion_percentage').alias('user_completion_percentage'))

    relationship_df = relationship_df.alias('t1').join(profile_dataset.alias('t2'), on=F.col('t1.friend_id')==F.col('t2.user_id'), how='leftouter')\
                    .select(F.col('t1.*'),
                            F.col('t2.gender').alias('friend_gender'),
                            F.col('t2.AGE').alias('friend_age'),
                            F.col('t2.completion_percentage').alias('friend_completion_percentage'))
    
    relationship_df = relationship_df.select(F.lit('1').alias('label'), '*')
    
    splits = relationship_df.randomSplit([1-test_ratio, test_ratio], random_seed)
    train_dataset, test_dataset = splits[0], splits[1]
    print('train dataset count: ', train_dataset.count())
    print('test dataset count: ', test_dataset.count())
    
    item_dataset = (
        relationship_df
        .withColumn('rn', F.row_number().over(
            Window.partitionBy('friend_id').orderBy(F.col('user_id'))
        ))
        .filter('rn == 1')
        .drop(F.col('rn'))
    )
    
    return train_dataset, test_dataset, item_dataset

def save_dataset(train_dataset, test_dataset, item_dataset, profile_dataset, relationship_dataset, 
                 train_path, test_path, item_path, profile_path, relationship_path, **kwargs):
    train_dataset.write.parquet(train_path, mode="overwrite")
    test_dataset.write.parquet(test_path, mode="overwrite")
    item_dataset.write.parquet(item_path, mode="overwrite")
    profile_dataset.write.parquet(profile_path, mode="overwrite")
    relationship_dataset.write.parquet(relationship_path, mode="overwrite")

if __name__=="__main__":
    print('Debug -- Pokec dataset preprocessing')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, action='store', default='', help='config file path')
    args = parser.parse_args()
    print('Debug -- conf:', args.conf)
    params = load_config(args.conf)
    # init spark
    spark = init_spark(params['spark'])
    # load datasets
    profile_dataset, relationship_dataset = load_dataset(spark,  **params['load_dataset'])
    # split train test
    train_dataset, test_dataset, item_dataset = split_train_test(profile_dataset, relationship_dataset, **params['split_train_test'])
    # save dataset
    save_dataset(train_dataset, test_dataset, item_dataset, profile_dataset, relationship_dataset, **params['save_dataset'])
    spark.stop()