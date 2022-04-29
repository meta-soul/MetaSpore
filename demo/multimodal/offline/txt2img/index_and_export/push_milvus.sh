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
collection=txt_to_img_demo
index_field=image_emb
index_file=data/unsplash-25k/unsplash_25k.index.json
# delete collection
#python src/indexing/milvus/drop.py --host ${host} --port ${port} --collection-name ${collection}

# build indexing
nohup python src/indexing/milvus/push.py --host ${host} --port ${port} --collection-name ${collection} --index-field ${index_field} --index-file ${index_file} > push_milvus.log 2>&1 &
