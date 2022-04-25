#
# Copyright 2022 DMetaSoul
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
