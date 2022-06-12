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

export PYTHONPATH="$PYTHONPATH:$PWD/src"

mkdir -p ./logs

nohup python -u src/train/train_cross_encoder.py --name train_ce_multiclass \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --train-file data/train/train.hard.neg1.pair.tsv \
    --train-kind multiclass \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 32 \
    --num-labels 2 \
    --save-steps 2000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    > logs/train_cross_encoder-1.log 2>&1 &
