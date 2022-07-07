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

python -m jobs.user_model.user_offline_embed \
    --job-name user-embedding \
    --scene-id ml-cb-100 \
    --action-type rating \
    --action-data ../data/ml-1m-schema/action.train \
    --item-data ../data/ml-1m-dump/item.emb \
    --dump-data ../data/ml-1m-dump/user.emb \
    --action-agg-func avg \
    --action-max-len 3 \
    --action-sortby-key action_time

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_user_tag \
    --data ../data/ml-1m-dump/user.tag \
    --fields user_id,tags,tag_weights \
    --index-fields user_id \
    --write-mode overwrite

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_user_emb \
    --data ../data/ml-1m-dump/user.emb \
    --fields user_id,user_emb \
    --index-fields user_id \
    --write-mode overwrite
