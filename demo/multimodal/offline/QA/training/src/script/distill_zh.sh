#device=cuda:0
device=cpu
batch_size=1
data_dir=../datasets/processed
log_dir=../logs
output_dir=../output
eval_list=csts_dev#${data_dir}/Chinese-STS-B/dev.tsv,csts_test#${data_dir}/Chinese-STS-B/test.tsv,afqmc_dev#${data_dir}/afqmc_public/dev.tsv,lcqmc_dev#${data_dir}/lcqmc/dev.tsv,bqcorpus_dev#${data_dir}/bq_corpus/dev.tsv,pawsx_dev#${data_dir}/paws-x-zh/dev.tsv,xiaobu_dev#${data_dir}/oppo-xiaobu/dev.tsv
bench_file=${data_dir}/sentences_zh/dev.tsv

###############
# 中文蒸馏轻量模型
###############
# 1. general-best
# 1.1 Distill
exp_name="general_distill_zh"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_csts_allnlizh_pkuparaph_pawsx-sts_default
s_model_path=${output_dir}/training_general_distill_zh
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_zh/train.tsv \
   --dev-file ${data_dir}/sentences_zh/dev.tsv \
   --test-file ${data_dir}/Chinese-STS-B/test.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 128 --learning-rate 1e-04 \
   --model-save-dir ${s_model_path} \
   > ${log_file} 2>&1 

# 1.2 Evaluation
python eval.py --eval-mode sts \
    --eval-list ${eval_list} \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# 1.3 Performance
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1

# 2. simclue-best
# 2.1 Train
exp_name="simclue_distill_zh"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_simclue-qmc_default
s_model_path=${output_dir}/training_simclue_distill_zh
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_zh/train.tsv \
   --dev-file ${data_dir}/sentences_zh/dev.tsv \
   --test-file ${data_dir}/simclue_public/test.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 128 --learning-rate 1e-04 \
   --model-save-path ${s_model_path} \
   > ${log_file} 2>&1 

# 2.2 Evaluation
python eval.py --eval-mode sts \
    --eval-list ${eval_list} \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# 2.3 Performance
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1

# 3. qmc_distill_zh
# 3.1 Train
exp_name="qmc_distill_zh"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_qmc_adaptive_domain-qmc_default_v3
s_model_path=${output_dir}/training_qmc_distill_zh
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_zh/train.tsv \
   --dev-file ${data_dir}/sentences_zh/dev.tsv \
   --test-file ${data_dir}/simclue_public/test.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 128 --learning-rate 1e-04 \
   --model-save-path ${s_model_path} \
   > ${log_file} 2>&1 

# 3.2 Evaluation
python eval.py --eval-mode sts \
    --eval-list ${eval_list} \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# 3.3 Performance
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1

# 4. dtm_distill_zh
# 4.1 Train
exp_name="dtm_distill_zh"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_dtm_adaptive_domain-qmc_default_v3
s_model_path=${output_dir}/training_dtm_distill_zh
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_zh/train.tsv \
   --dev-file ${data_dir}/sentences_zh/dev.tsv \
   --test-file ${data_dir}/simclue_public/test.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 128 --learning-rate 1e-04 \
   --model-save-path ${s_model_path} \
   > ${log_file} 2>&1 

# 4.2 Evaluation
python eval.py --eval-mode sts \
    --eval-list ${eval_list} \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# 4.3 Performance
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1

# 5. fin_qmc_distill_zh
# 5.1 Train
exp_name="fin_qmc_distill_zh"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_fin_qmc_adaptive_domain-qmc_default_v3
s_model_path=${output_dir}/training_fin_qmc_distill_zh
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_zh/train.tsv \
   --dev-file ${data_dir}/sentences_zh/dev.tsv \
   --test-file ${data_dir}/simclue_public/test.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 128 --learning-rate 1e-04 \
   --model-save-path ${s_model_path} \
   > ${log_file} 2>&1 

# 5.2 Evaluation
python eval.py --eval-mode sts \
    --eval-list ${eval_list} \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# 5.3 Performance
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
