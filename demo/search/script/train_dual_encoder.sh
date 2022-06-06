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

# online sampling negatives
python -u src/train/train_dual_encoder.py --name train_de_loss_contrastive_in_batch \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss contrastive_in_batch \
    --train-file data/train/train.pos.tsv \
    --train-kind pair \
    --train-text-index 0,1 \
    --train-label-index -1 \
    --train-batch-size 64 \
    --save-steps 2000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    --debug \
    > logs/train_dual_encoder-1.log 2>&1

python -u src/train/train_dual_encoder.py --name train_de_loss_contrastive_in_batch_with_neg \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss contrastive_in_batch \
    --train-file data/train/train.rand.neg.triplet.tsv \
    --train-kind triplet \
    --train-text-index 0,1,2 \
    --train-label-index -1 \
    --train-batch-size 32 \
    --save-steps 10000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    --debug \
    > logs/train_dual_encoder-2.log 2>&1

# offline sampling negatives
python -u src/train/train_dual_encoder.py --name train_de_loss_triplet \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss triplet \
    --train-file data/train/train.rand.neg.triplet.tsv \
    --train-kind triplet \
    --train-text-index 0,1,2 \
    --train-label-index -1 \
    --train-batch-size 32 \
    --save-steps 10000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    --debug \
    > logs/train_dual_encoder-3.log 2>&1

python -u src/train/train_dual_encoder.py --name train_de_loss_cosine \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss cosine \
    --train-file data/train/train.rand.neg.pair.tsv \
    --train-kind pair_with_label \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 64 \
    --save-steps 10000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    --debug \
    > logs/train_dual_encoder-4.log 2>&1

python -u src/train/train_dual_encoder.py --name train_de_loss_contrastive \
    --model DMetaSoul/sbert-chinese-general-v2 \
    --dual-model DMetaSoul/sbert-chinese-general-v2 \
    --num-epochs 2 \
    --lr 3e-05 \
    --loss contrastive \
    --train-file data/train/train.rand.neg.pair.tsv \
    --train-kind pair_with_label \
    --train-text-index 0,1 \
    --train-label-index 2 \
    --train-batch-size 64 \
    --save-steps 10000 \
    --eval-qid-file data/dev/dev4eval.qid.tsv \
    --eval-pid-file data/dev/dev4eval.pid.tsv \
    --eval-rel-file data/dev/dev4eval.rel.tsv \
    --debug \
    > logs/train_dual_encoder-5.log 2>&1
