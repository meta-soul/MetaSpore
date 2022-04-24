#######################
# 中文表示学习
# - 基于 SimCLUE 语义数据集
#######################
log_dir=../logs
output_dir=../output
dataset_dir=../datasets/processed
mkdir -p ${log_dir}
mkdir -p ${output_dir}

# 中文语义相似数据集benchmark
eval_list=csts_dev#${dataset_dir}/Chinese-STS-B/dev.tsv,csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv

:<<EOF
exp_name=simclue
task_type=triplet
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.triplet.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 128 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-triplet_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=triplet
loss_type=triplet
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.triplet.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 16 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-triplet_triplet \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-qmc_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=qmc
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-qmc_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=qmc
loss_type=logistic
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-qmc_logistic \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=simclue
task_type=qmc
loss_type=cosine
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/simclue_public/train.tsv \
    --dev-file ${dataset_dir}/simclue_public/dev.tsv \
    --test-file ${dataset_dir}/simclue_public/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_simclue-qmc_cosine \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1
EOF

# SimCLUE
log_file=${log_dir}/eval.log-zh_sup_general_v3.log
model_list=csts_base#${output_dir}/training_csts_benchmark-sts,csts_best#${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default,simclue_v1#${output_dir}/training_simclue-qmc_default,simclue_v2#${output_dir}/training_simclue-qmc_ranking,simclue_v3#${output_dir}/training_simclue-triplet_default,simclue_v4#${output_dir}/training_simclue-triplet_triplet
eval_list_simclue=${eval_list},simclue_dev#${dataset_dir}/simclue_public/dev.tsv,simclue_test#${dataset_dir}/simclue_public/test.tsv
nohup python eval.py --model-list ${model_list} --eval-list ${eval_list_simclue} --device cuda:0 --batch-size 64 > ${log_file} 2>&1 &

