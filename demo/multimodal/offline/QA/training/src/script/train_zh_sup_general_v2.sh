###############
# 中文表示学习
# - 多任务监督组合学习
###############
log_dir=../logs
output_dir=../output
dataset_dir=../datasets/processed
mkdir -p ${log_dir}
mkdir -p ${output_dir}

# 中文语义相似数据集benchmark
eval_list=csts_dev#${dataset_dir}/Chinese-STS-B/dev.tsv,csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv

# 1. 从难到易串行训练
:<<EOF
model_path=${output_dir}/training-pawsx-qmc-default
exp_name=pkuparaph_pawsx
task_type=qmc
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/PKU-Paraphrase-Bank/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_pkuparaph_pawsx-qmc_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_pkuparaph_pawsx-qmc_ranking
exp_name=allnlizh_pkuparaph_pawsx
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli_zh/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_allnlizh_pkuparaph_pawsx-nli_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_allnlizh_pkuparaph_pawsx-nli_ranking
exp_name=csts_allnlizh_pkuparaph_pawsx
task_type=sts
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Eval 通用模型-从难到易
log_file=${log_dir}/eval.log-zh_sup_general_v2_1.log
model_list=csts#${output_dir}/training_csts_benchmark-sts,pawsx#${output_dir}/training_pawsx-qmc-default,pkuparaph_pawsx#${output_dir}/training_pkuparaph_pawsx-qmc_ranking,allnlizh_pkuparaph_pawsx#${output_dir}/training_allnlizh_pkuparaph_pawsx-nli_ranking,csts_allnlizh_pkuparaph_pawsx#${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default
python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 64 > ${log_file} 2>&1
EOF

# 2. 从易到难训练
:<<EOF
exp_name=allnlizh
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/allnli_zh/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_allnlizh-nli_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_allnlizh-nli_ranking
exp_name=pkuparaph_allnlizh
task_type=qmc
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/PKU-Paraphrase-Bank/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_pkuparaph_allnlizh-qmc_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_pkuparaph_allnlizh-qmc_ranking
exp_name=pawsx_pkuparaph_allnlizh
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/paws-x-zh/train.tsv \
    --dev-type qmc --dev-file ${dataset_dir}/paws-x-zh/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_pawsx_pkuparaph_allnlizh-qmc_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_pawsx_pkuparaph_allnlizh-qmc_default
exp_name=csts_pawsx_pkuparaph_allnlizh
task_type=sts
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_csts_pawsx_pkuparaph_allnlizh-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Eval 通用模型-从易到难
log_file=${log_dir}/eval.log-zh_sup_general_v2_2.log
model_list=csts#${output_dir}/training_csts_benchmark-sts,allnlizh#${output_dir}/training_allnlizh-nli_ranking,pkuparaph_allnlizh#${output_dir}/training_pkuparaph_allnlizh-qmc_ranking,pawsx_pkuparaph_allnlizh#${output_dir}/training_pawsx_pkuparaph_allnlizh-qmc_default,csts_pawsx_pkuparaph_allnlizh#${output_dir}/training_csts_pawsx_pkuparaph_allnlizh-sts_default
python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 64 > ${log_file} 2>&1
EOF

# 3. 仅使用NLI数据训练
:<<EOF
model_path=${output_dir}/training_allnlizh-nli_ranking
exp_name=csts_allnlizh
task_type=sts
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_csts_allnlizh-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1
EOF

# Eval 通用模型-仅保留NLI
log_file=${log_dir}/eval.log-zh_sup_general_v2_3.log
model_list=csts#${output_dir}/training_csts_benchmark-sts,allnlizh#${output_dir}/training_allnlizh-nli_ranking,csts_allnlizh#${output_dir}/training_csts_allnlizh-sts_default
python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 64 > ${log_file} 2>&1

# 4. 全部正样本对数据集
exp_name=pospair_zh
task_type=qmc
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/pos_pairs_zh/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_pospair_zh-qmc_ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

model_path=${output_dir}/training_pospair_zh-qmc_ranking
exp_name=csts_pospair_zh
task_type=sts
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_csts_pospair_zh-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Eval 通用模型-混合正样本
log_file=${log_dir}/eval.log-zh_sup_general_v2_4.log
model_list=csts#${output_dir}/training_csts_benchmark-sts,allpospair#${output_dir}/training_pospair_zh-qmc_ranking,csts_allpospair#${output_dir}/training_csts_pospair_zh-sts_default
nohup python eval.py --model-list ${model_list} --eval-list ${eval_list} > ${log_file} 2>&1 &
