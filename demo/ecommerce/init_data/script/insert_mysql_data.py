import os
try:
    import findspark
    findspark.init()
except:
    pass
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

if __name__ == '__main__':
    conf = SparkConf().set("spark.jars", "/opt/script/mysql-connector-java-5.1.49.jar").setAppName('insert-mysql')
    sc = SparkContext('local', 'init_mysql_data', conf=conf)
    spark = SparkSession(sc)

    prop = {}
    prop['user'] = 'root'
    prop['password'] = 'test_mysql_123456'
    prop['driver'] = 'com.mysql.jdbc.Driver'

    user_parquet_path = '/opt/script/amazon_fashion_user.base.parquet'
    item_parquet_path = '/opt/script/amazon_fashion_item.base.parquet'
    interantion_parquet_path = '/opt/script/amazon_fashion_interaction.base.parquet'
    item_df = spark.read.parquet(item_parquet_path)
    interaction_df = spark.read.parquet(interantion_parquet_path)
    user_df = spark.read.parquet(user_parquet_path)
    user_df.write.jdbc("jdbc:mysql://localhost:3306/metaspore_offline_flow?useSSL=false", 'user', 'overwrite', prop)
    item_df.write.jdbc("jdbc:mysql://localhost:3306/metaspore_offline_flow?useSSL=false", 'item', 'overwrite', prop)
    interaction_df.write.jdbc("jdbc:mysql://localhost:3306/metaspore_offline_flow?useSSL=false", 'interaction', 'overwrite', prop)
