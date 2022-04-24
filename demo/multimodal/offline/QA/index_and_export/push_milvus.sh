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

host=${MY_MILVUS_HOST}
port=${MY_MILVUS_PORT}
collection=baike_qa_demo
# 删除索引库
#python src/indexing/milvus/drop.py --host ${host} --port ${port} --collection-name ${collection}

# 构建索引库
#python src/indexing/milvus/push.py --host ${host} --port ${port} --collection-name ${collection} --index-field question_emb  --index-file data/baike/baike_qa_1w.doc.index.json
nohup python src/indexing/milvus/push.py --host ${host} --port ${port} --collection-name ${collection} --index-field question_emb  --index-file data/baike/baike_qa_train.doc.index.json > push_milvus.log 2>&1 &
