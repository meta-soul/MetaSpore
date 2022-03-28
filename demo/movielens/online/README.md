# movielens



## Mongodb Query Test

自动生成 Feature Query 教程: [**教程链接**](https://github.com/meta-soul/auto-feature-plugin/blob/main/README.md)

注：本地测试使用 ssh 转发端口
```
ssh -i ~/.ssh/YOUR_RSA YOUR_USER@120.92.42.86 -L 27017:127.0.0.1:27017 
```
## 灌 Mongo	

1. pyspark 灌 mongodb 脚本：write_mongo.py

```python
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession

if __name__ == '__main__':
    table_name = sys.argv[1]
    mongodb_uri = "mongodb://jpa:Dmetasoul_123456@172.31.37.47:27017/jpa." + table_name
    data_path = "s3://dmetasoul-bucket/demo/movielens/mango/" + table_name + "s.parquet/*"
    print("mongodb_uri: "+mongodb_uri)
    print("data_path: "+data_path)

    spark = SparkSession \
        .builder \
        .master('local') \
        .config("spark.mongodb.input.uri", mongodb_uri) \
        .config("spark.mongodb.output.uri", mongodb_uri) \
        .getOrCreate()

    ori_query_col = ""
    if table_name == "item":
        ori_query_col = "movie_id"
    if table_name == "user":
        ori_query_col = "user_id"

    if ori_query_col != "":
        read_df = spark.read.parquet(data_path)
        format_df = read_df.withColumn("queryid", read_df[ori_query_col].cast(StringType()))
        format_df.write.format('mongo').mode("overwrite").save()

```



2. 执行脚本

根据传入参数不同，灌不同的 mongodb 表。例如 item，user

```shel
spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
/home/spark/work/write_mongo.py item
```

