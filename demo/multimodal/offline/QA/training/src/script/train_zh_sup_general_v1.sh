###############
# 中文表示学习
# - 各个任务监督学习
###############
log_dir=../logs
output_dir=../output
dataset_dir=../datasets/processed
mkdir -p ${log_dir}
mkdir -p ${output_dir}

exp_name=csts_benchmark
task_type=sts
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/Chinese-STS-B/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-csts_benchmark-sts_default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=ocnli
task_type=nli
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/ocnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-ocnli-nli-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=ocnli_rank
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/ocnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-ocnli_rank-nli-ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=cmnli
task_type=nli
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/cmnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-cmnli-nli-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=cmnli_rank
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/cmnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-cmnli_rank-nli-ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=csnli
task_type=nli
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/csnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-csnli-nli-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=csnli_rank
task_type=nli
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/csnli_public/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-csnli_rank-nli-ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=pkuparaph
task_type=qmc
loss_type=ranking
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/PKU-Paraphrase-Bank/train.tsv \
    --dev-file ${dataset_dir}/Chinese-STS-B/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 32 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-pkuparaph-qmc-ranking \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=afqmc
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/afqmc_public/train.tsv \
    --dev-file ${dataset_dir}/afqmc_public/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-afqmc-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=lcqmc
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/lcqmc/train.tsv \
    --dev-file ${dataset_dir}/lcqmc/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-lcqmc-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=bqcorpus
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/bq_corpus/train.tsv \
    --dev-file ${dataset_dir}/bq_corpus/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-bqcorpus-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=pawsx
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/paws-x-zh/train.tsv \
    --dev-file ${dataset_dir}/paws-x-zh/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-pawsx-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=xiaobu
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/oppo-xiaobu/train.tsv \
    --dev-file ${dataset_dir}/oppo-xiaobu/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-xiaobu-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

exp_name=qbqtc
task_type=qmc
loss_type=default
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${dataset_dir}/QBQTC/train.tsv \
    --dev-file ${dataset_dir}/QBQTC/dev.tsv \
    --test-file ${dataset_dir}/Chinese-STS-B/test.tsv \
    --model bert-base-chinese --max-seq-len 256 --device cuda:0 \
    --num-epochs 4 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${output_dir}/training-qbqtc-qmc-default \
    > ${log_dir}/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Eval 各任务之间zero-shot能力
model_list=csts#${output_dir}/training-csts_benchmark-sts_default,ocnli#${output_dir}/training-ocnli-nli-default,ocnli_rank#${output_dir}/training-ocnli_rank-nli-ranking,cmnli#${output_dir}/training-cmnli-nli-default,cmnli_rank#${output_dir}/training-cmnli_rank-nli-ranking,csnli#${output_dir}/training-csnli-nli-default,csnli_rank#${output_dir}/training-csnli_rank-nli-ranking,pku#${output_dir}/training-pkuparaph-qmc-ranking,afqmc#${output_dir}/training-afqmc-qmc-default,lcqmc#${output_dir}/training-lcqmc-qmc-default,bqcorpus#${output_dir}/training-bqcorpus-qmc-default,pawsx#${output_dir}/training-pawsx-qmc-default,xiaobu#${output_dir}/training-xiaobu-qmc-default,qbqtc#${output_dir}/training-qbqtc-qmc-default
eval_list=csts_test#${dataset_dir}/Chinese-STS-B/test.tsv,ocnli_dev#${dataset_dir}/ocnli_public/dev.tsv,afqmc_dev#${dataset_dir}/afqmc_public/dev.tsv,lcqmc_dev#${dataset_dir}/lcqmc/dev.tsv,bqcorpus_dev#${dataset_dir}/bq_corpus/dev.tsv,pawsx_dev#${dataset_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${dataset_dir}/oppo-xiaobu/dev.tsv,cmnli_dev#${dataset_dir}/cmnli_public/dev.tsv,csnli_dev#${dataset_dir}/csnli_public/dev.tsv

python eval.py --model-list ${model_list} --eval-list ${eval_list} > ${log_dir}/eval.log-zh_sup_general_v1 2>&1
