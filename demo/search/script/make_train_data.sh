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

# raw json to tsv data
cat data/train/train.json | python src/preprocess/json2tsv.py > data/train/train.tsv

# [(query, pos),...]
cut -f1,3 data/train/train.tsv > data/train/train.pos.tsv

# [(query, pos, 1), (query, neg, 0),...]
cat data/train/train.pos.tsv | python src/preprocess/negative_rand_sample.py 5 pair > data/train/train.rand.neg.pair.tsv

# [(query, pos, neg),...]
cat data/train/train.pos.tsv | python src/preprocess/negative_rand_sample.py 1 triplet > data/train/train.rand.neg.triplet.tsv
