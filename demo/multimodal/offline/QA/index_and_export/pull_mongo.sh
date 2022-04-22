export PYTHONPATH="${PYTHONPATH}:./src"
source ./env.sh

spark-submit \
--master local \
--name read_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./src/indexing/mongodb/pull.py \
--mongo-uri mongodb://${MY_MONGO_DB}:${MY_MONGO_USERNAME}_${MY_MONGO_PASSWORD}@${MY_MONGO_HOST}:${MY_MONGO_PORT} \
--mongo-table jpa.baike_qa_demo \
