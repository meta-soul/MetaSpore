#device=cuda:0
device=cpu
batch_size=1
log_dir=../logs
output_dir=../output
data_dir=./datasets/processed

###############
# 英文蒸馏轻量模型
##############
exp_name="sts_distill_en"
log_file=${log_dir}/train.log-${exp_name}
t_model_path=${output_dir}/training_sts_benchmark_nli_ranking-sts_default
s_model_path=${output_dir}/training_sts_distill_en
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_en/train.tsv \
   --dev-file ${data_dir}/sentences_en/dev.tsv \
   --test-file ${data_dir}/sts_benchmark/dev.tsv \
   --teacher-model ${t_model_path} \
   --student-keep-layers 1,4,7,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 256 \
   --model-save-dir ${s_model_path} \
   > ${log_file} 2>&1

# Evaluation
python eval.py --eval-mode sts --eval-list sts_test#${data_dir}/sts_benchmark/test.tsv \
    --model-list teacher#${t_model_path},student#${s_model_path} \
    --batch-size 32 --device cuda:0 --max-seq-len 256 \
    >> ${log_file} 2>&1

# Performance
bench_file=${data_dir}/sentences_en/dev.tsv
python encode.py --input-file ${bench_file} \
    --model ${t_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
python encode.py --input-file ${bench_file} \
    --model ${s_model_path} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    >> ${log_file} 2>&1
