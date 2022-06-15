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

# 1. make pair format hard negative dataset (you can use that to train cross-encoder model)
python src/preprocess/negative_hard_sample.py \
    data/train/train.pos.tsv \
    data/output/train.recall.top50 \
    data/passage-collection/part-00,data/passage-collection/part-01,data/passage-collection/part-02,data/passage-collection/part-03 \
    1 pair data/train/train.hard.neg1.pair.tsv \

# 2. make pair format hard negative dataset (you can use that to train dual-encoder model with in-batch loss)
python src/preprocess/negative_hard_sample.py \
    data/train/train.pos.tsv \
    data/output/train.recall.top50 \
    data/passage-collection/part-00,data/passage-collection/part-01,data/passage-collection/part-02,data/passage-collection/part-03 \
    1 triplet data/train/train.hard.neg1.triplet.tsv \
