import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession \
    .builder \
    .appName("mongodbtest1") \
    .master('local')\
    .config("spark.mongodb.input.uri", "mongodb://root:Dmetasoul_123456@127.0.0.1:27018/jpa") \
    .config("spark.mongodb.output.uri", "mongodb://root:Dmetasoul_123456@127.0.0.1:27018/jpa") \
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
    .getOrCreate()

studentDf = spark.createDataFrame([
    Row(id=1,name='vijay',marks=67),
    Row(id=2,name='Ajay',marks=88),
    Row(id=3,name='jay',marks=79),
    Row(id=4,name='binny',marks=99),
    Row(id=5,name='omair',marks=99),
    Row(id=6,name='divya',marks=98),
])
studentDf.show()
studentDf.select("id","name","marks").write\
    .format('com.mongodb.spark.sql.DefaultSource')\
    .option( "uri", "mongodb://root:Dmetasoul_123456@127.0.0.1:27018/jpa.test?authSource=admin") \
    .save()