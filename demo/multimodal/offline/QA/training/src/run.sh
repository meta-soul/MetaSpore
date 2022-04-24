# 单脚本启动完整训练流程：train -> train-eval -> train-bench -> distill -> distill-eval -> distill-bench -> export

data_dir=../datasets/processed
output_dir=../output/sts-b
mkdir -p ${output_dir}

# Train
exp_name=sts_benchmark
task_type=sts
loss_type=default
model_save_dir=${output_dir}/training_pipline_train
mkdir -p ${model_save_dir}/logs
python train.py --exp-name $exp_name --task-type ${task_type} --loss-type ${loss_type} \
    --train-file ${data_dir}/sts_benchmark/train.tsv \
    --dev-file ${data_dir}/sts_benchmark/dev.tsv \
    --test-file ${data_dir}/sts_benchmark/test.tsv \
    --model bert-base-uncased --max-seq-len 256 --device cuda:0 \
    --num-epochs 1 --train-batch-size 64 --eval-batch-size 32 --learning-rate 2e-05 \
    --model-save-dir ${model_save_dir} \
    > ${model_save_dir}/logs/train.log-${exp_name}-${task_type}-${loss_type} 2>&1

# Train-Evaluation
python eval.py --eval-mode sts --eval-list sts_test#${data_dir}/sts_benchmark/test.tsv \
    --model-list sts#${model_save_dir} \
    > ${model_save_dir}/logs/eval.log-${exp_name}-${task_type}-${loss_type} 2> /dev/null

# Train-Bench
batch_size=1
device=cpu
bench_file=${data_dir}/sentences_en/dev.tsv
python encode.py --input-file ${bench_file} \
    --model ${model_save_dir} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    > ${model_save_dir}/logs/bench.log-${exp_name}-${task_type}-${loss_type} 2> /dev/null

# Distill 
teacher_model_path=${output_dir}/training_pipline_train
model_save_dir=${output_dir}/training_pipline_distill
mkdir -p ${model_save_dir}/logs
exp_name="sts_distill_en"
python distill.py --exp-name $exp_name \
   --train-file ${data_dir}/sentences_en/train.tsv \
   --dev-file ${data_dir}/sentences_en/dev.tsv \
   --test-file ${data_dir}/sts_benchmark/dev.tsv \
   --teacher-model ${teacher_model_path} \
   --student-keep-layers 1,10 \
   --max-seq-len 256 --device cuda:0 \
   --num-epochs 1 --train-batch-size 256 \
   --model-save-dir ${model_save_dir} \
   > ${model_save_dir}/logs/train.log 2>&1

# Distill-Evaluation
python eval.py --eval-mode sts --eval-list sts_test#${data_dir}/sts_benchmark/test.tsv \
    --model-list sts#${model_save_dir} \
    > ${model_save_dir}/logs/eval.log 2> /dev/null

# Distill-Bench
batch_size=1
device=cpu
bench_file=${data_dir}/sentences_en/dev.tsv
python encode.py --input-file ${bench_file} \
    --model ${model_save_dir} \
    --batch-size ${batch_size} --device ${device} --max-seq-len 256 --disable-progress \
    > ${model_save_dir}/logs/bench.log 2> /dev/null

# Export
export_path=${output_dir}/training_pipline_export
python infer/modeling.py --model-name ${model_save_dir} --onnx-path ${export_path} > ${model_save_dir}/logs/export.log 2>&1
python infer/bench.py --model-name ${model_save_dir} --onnx-path ${export_path} > ${model_save_dir}/logs/bench_onnx.log 2>&1
