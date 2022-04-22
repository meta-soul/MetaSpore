export PYTHONPATH="${PYTHONPATH}:./src"
source ./env.sh

spark-submit \
--master local \
--name write_mongo \
--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
./src/indexing/mongodb/push.py \
--mongo-uri mongodb://${MY_MONGO_DB}:${MY_MONGO_USERNAME}_${MY_MONGO_PASSWORD}@${MY_MONGO_HOST}:${MY_MONGO_PORT} \
--mongo-table jpa.baike_qa_demo \
--id-field id \
--index-file data/baike/baike_qa_train.doc.index.json
