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

python -m jobs.item_model.item_offline_attr \
    --job-name item-attr \
    --item-data ../data/ml-1m-schema/item.train \
    --action-data ../data/ml-1m-schema/action.train \
    --dump-attr-data ../data/ml-1m-dump/item.attr \
    --dump-tag-data ../data/ml-1m-dump/item.rtag \
    --scene-id ml-cb-100 \
    --action-type rating \
    --tag-max-len 5000

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_item_attr \
    --data ../data/ml-1m-dump/item.attr \
    --fields item_id,tags,weight,popularity \
    --index-fields item_id \
    --write-mode overwrite

python -m jobs.tools.push_mongo --mongo-uri mongodb://172.31.37.47:27017 \
    --mongo-database jpa \
    --mongo-collection movielens_cb_demo_item_rtag \
    --data ../data/ml-1m-dump/item.rtag \
    --fields tag,item_ids \
    --index-fields tag \
    --write-mode overwrite
