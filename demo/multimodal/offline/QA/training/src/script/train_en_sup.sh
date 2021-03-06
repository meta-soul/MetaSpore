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

###############
# english supervised
###############
log_dir=../logs
output_dir=../output
dataset_dir=../datasets/processed
mkdir -p ${log_dir}
mkdir -p ${output_dir}

:<<EOF
# 1.1 train with sts
exp_name=sts_benchmark
task_type=sts
loss_type=default
nohup python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_sts_benchmark-sts \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 1.2 train with sts + CircleLoss
exp_name=sts_benchmark
task_type=sts
loss_type=circle
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_sts_benchmark-sts-circle \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.1 train with xnli/mnli + softmax loss
exp_name=allnli
task_type=nli
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_allnli-nli \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.2 train with xnli/mnli + ranking loss
exp_name=allnli_ranking
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_allnli_ranking-nli_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 2.3 train with xnli/mnli + circle loss
exp_name=allnli
task_type=nli
loss_type=circle
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_allnli-nli-circle \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 3.1 continue train sts based nli model
exp_name=sts_benchmark_nli
task_type=sts
loss_type=default
model_path=./archives/training_allnli-nli/2022-02-22_14-24-23
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_sts_benchmark_nli-sts \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 3.2 continue train sts based nli model based ranking loss
exp_name=sts_benchmark_nli_ranking
task_type=sts
loss_type=default
model_path=${output_dir}/training_allnli_ranking-nli_ranking
nohup python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_sts_benchmark_nli_ranking-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1 &

# 3.3 continue train sts based nli model based circle loss
exp_name=sts_benchmark_nli
task_type=sts
loss_type=circle
model_path=${output_dir}/training_allnli-nli-circle
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/sts_benchmark/train.tsv \
    --dev-file ${dataset_dir}/sts_benchmark/dev.tsv \
    --test-file ${dataset_dir}/sts_benchmark/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_sts_benchmark_nli_circle-sts-circle \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1
EOF

# Evaluation
python eval.py --eval-mode sts --eval-list sts_test#${dataset_dir}/sts_benchmark/test.tsv \
    --model-list sts#${output_dir}/training_sts_benchmark-sts,sts_v2#${output_dir}/training_sts_benchmark-sts-circle,allnli#${output_dir}/training_allnli-nli,allnli_v2#${output_dir}/training_allnli_ranking-nli_ranking,allnli_v3#${output_dir}/training_allnli-nli-circle,sts_allnli#${output_dir}/training_sts_benchmark_nli-sts,sts_allnli_v2#${output_dir}/training_sts_benchmark_nli_ranking-sts_default,sts_allnli_v3#${output_dir}/training_sts_benchmark_nli_circle-sts-circle,mpnet#all-mpnet-base-v2 \
    > ${log_dir}/eval.log-en_sup.log 2>&1
