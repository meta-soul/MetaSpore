#!/bin/sh
spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin items --dest item --queryid movie_id


spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin users --dest user --queryid user_id

spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin item_feature --dest item_feature --queryid movie_id

spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin swing --dest swing --queryid key

spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin itemcf --dest itemcf --queryid key

spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./write_mongo.py --origin milvus_item_id --dest milvus_item_id --queryid milvus_id

