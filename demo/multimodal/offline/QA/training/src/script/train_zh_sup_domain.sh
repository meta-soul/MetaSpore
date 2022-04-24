#######################
# 中文表示学习
# - 领域迁移学习
#######################
log_dir=../logs
output_dir=../output
dataset_dir=../datasets/processed
mkdir -p ${log_dir}
mkdir -p ${output_dir}

# 中文语义相似数据集benchmark
eval_list=csts_dev#${dataset_dir}/Chinese-STS-B/dev.tsv,csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv

# 开放领域问题匹配
:<<EOF
# v1
model_path=${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default
exp_name=qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/lcqmc/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/lcqmc/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_qmc_adaptive_domain-qmc_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v2
model_path=${output_dir}/training_allnlizh-nli_ranking
exp_name=qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/lcqmc/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/lcqmc/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_qmc_adaptive_domain-qmc_default_v2 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v3
model_path=${output_dir}/training_simclue-qmc_default
exp_name=qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/lcqmc/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/lcqmc/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_qmc_adaptive_domain-qmc_default_v3 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 专用模型-问题匹配
# domain_v1: ${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default, domain_v2: ${output_dir}/training_allnlizh-nli_ranking, domain_v3: ${output_dir}/training_simclue-qmc_default
#log_file=${log_dir}/eval.log-zh_qmc.log
#model_list=csts_base#${output_dir}/training_csts_benchmark-sts,csts_best#${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default,qmc_base#${output_dir}/training_lcqmc-qmc,qmc_domain_v1#${output_dir}/training_qmc_adaptive_domain-qmc_default,qmc_domain_v2#${output_dir}/training_qmc_adaptive_domain-qmc_default_v2,qmc_domain_v3#${output_dir}/training_qmc_adaptive_domain-qmc_default_v3
#nohup python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 16 > ${log_file} 2>&1 &
EOF

# 开放领域对话匹配
:<<EOF
# v1
model_path=${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default
exp_name=dtm_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/oppo-xiaobu/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/oppo-xiaobu/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_dtm_adaptive_domain-qmc_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v2
model_path=${output_dir}/training_allnlizh-nli_ranking
exp_name=dtm_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/oppo-xiaobu/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/oppo-xiaobu/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_dtm_adaptive_domain-qmc_default_v2 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v3
model_path=${output_dir}/training_simclue-qmc_default
exp_name=dtm_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/oppo-xiaobu/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/oppo-xiaobu/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_dtm_adaptive_domain-qmc_default_v3 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 专用模型-对话匹配
# domain_v1: ${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default, domain_v2: ${output_dir}/training_allnlizh-nli_ranking, domain_v3: ${output_dir}/training_simclue-qmc_default
#log_file=${log_dir}/eval.log-zh_dtm.log
#model_list=csts_base#${output_dir}/training_csts_benchmark-sts,csts_best#${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default,dtm_base#${output_dir}/training_xiaobu-qmc,dtm_domain#${output_dir}/training_dtm_adaptive_domain-qmc_default,dtm_domain_v2#${output_dir}/training_dtm_adaptive_domain-qmc_default_v2,dtm_domain_v3#${output_dir}/training_dtm_adaptive_domain-qmc_default_v3
#nohup python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 16 > ${log_file} 2>&1 &
EOF

# 金融领域问题匹配
:<<EOF
# v1
model_path=${output_dir}/training_qmc_adaptive_domain-qmc_default
exp_name=fin_qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/bq_corpus/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/bq_corpus/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_fin_qmc_adaptive_domain-qmc_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v2
model_path=${output_dir}/training_qmc_adaptive_domain-qmc_default_v2
exp_name=fin_qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/bq_corpus/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/bq_corpus/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_fin_qmc_adaptive_domain-qmc_default_v2 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# v3
model_path=${output_dir}/training_qmc_adaptive_domain-qmc_default_v3
exp_name=fin_qmc_adaptive_domain
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/bq_corpus/train.tsv \
    --dev-type sts --dev-file ${dataset_dir}/bq_corpus/dev.tsv \
    --test-type sts --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model ${model_path} --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training_fin_qmc_adaptive_domain-qmc_default_v3 \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# 专用模型-金融领域问题匹配
# domain_v1: ${output_dir}/training_qmc_adaptive_domain-qmc_default, domain_v2: ${output_dir}/training_qmc_adaptive_domain-qmc_default_v2, domain_v3: ${output_dir}/training_qmc_adaptive_domain-qmc_default_v3
#log_file=${log_dir}/eval.log-zh_finance.log
#model_list=csts_base#${output_dir}/training_csts_benchmark-sts,csts_best#${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default,fin_base#${output_dir}/training_bqcorpus-qmc,fin_domain#${output_dir}/training_fin_qmc_adaptive_domain-qmc_default,fin_domain_v2#${output_dir}/training_fin_qmc_adaptive_domain-qmc_default_v2,fin_domain_v3#${output_dir}/training_fin_qmc_adaptive_domain-qmc_default_v3
#nohup python eval.py --model-list ${model_list} --eval-list ${eval_list} --device cuda:0 --batch-size 16 > ${log_file} 2>&1 &
EOF

