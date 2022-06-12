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

set -x

query_model=$1
passage_model=$2
query_file=$3
passage_data_dir=$4
topk=$5
split_name=$6
job_name=$7

index_mode=faiss:FlatIP
num_parts=4
part_size=`cat ${passage_data_dir}/part-00 | wc -l`
query_embs=./data/output/${split_name}.query.embs
passage_embs=./data/output/passage.embs
result_file=./data/output/${split_name}.recall.top${topk}

# encode query
echo "Encode query starting..."
#[ "$job_name" == "index" ] && sh script/infer_dual_encoder.sh cuda:0 "query" ${query_model} ${query_file} ${query_embs}
sh script/infer_dual_encoder.sh cuda:0 "query" ${query_model} ${query_file} ${query_embs}
echo "Encode query done!"

# encode passage
echo "Encode passage starting..."
for (( part_id=0; part_id<$num_parts; part_id++ ))
do

    device="cuda:${part_id}"
    part_id=`printf "%02d" ${part_id}`
    input_file=${passage_data_dir}/part-${part_id}
    output_file=${passage_embs}.part-${part_id}

    # multi-gpu parallel
    #nohup sh script/infer_dual_encoder.sh "cuda:0" "${part_id}" ${passage_model} ${input_file} ${output_file} &

    [ "$job_name" == "index" ] && sh script/infer_dual_encoder.sh "cuda:0" "${part_id}" ${passage_model} ${input_file} ${output_file}
done
wait
echo "Encode passage done!"

# index
echo "Index starting..."
for (( part_id=0; part_id<$num_parts; part_id++ ))
do
    part_id=`printf "%02d" ${part_id}`
    input_file=${passage_embs}.part-${part_id}.npy
    output_file=${passage_embs}.part-${part_id}.faiss
    if [ -f "${input_file}" ]; then
        #nohup python src/search/index_build.py --emb-file ${input_file} --index-file ${output_file} --index-mode ${index_mode} && rm -rf ${input_file} &
        python src/search/index_build.py --emb-file ${input_file} --index-file ${output_file} --index-mode ${index_mode} && rm -rf ${input_file}
    fi
done
wait
echo "Index done!"

#if [ "$job_name" == "index" ]; then
#    exit
#fi

# search
echo "Search starting..."
shift_idx=0
recall_batch=500
for (( part_id=0; part_id<$num_parts; part_id++ ))
do
    part_id=`printf "%02d" ${part_id}`
    input_file=${passage_embs}.part-${part_id}.faiss
    output_file=./data/output/${split_name}.recall.top${topk}-${part_id}
    if [ -f "${input_file}" ]; then
        # parallel
        #nohup python src/search/index_search.py ${query_file} ${query_embs}.npy ${input_file} ${output_file} ${topk} ${recall_batch} &
        python src/search/index_search.py ${query_file} ${query_embs}.npy ${shift_idx} ${input_file} ${output_file} ${topk} ${recall_batch}
    fi
    doc_file=${passage_data_dir}/part-${part_id}
    n=`wc -l ${doc_file} | cut -d' ' -f1`
    shift_idx=$(( $shift_idx + $n ))
done
wait
echo "Search done!"

# merge search results
echo "Merge starting..."
recall_results=''
for (( part_id=0; part_id<$num_parts; part_id++ ))
do
    part_id=`printf "%02d" ${part_id}`
    input_file=./data/output/${split_name}.recall.top${topk}-${part_id}
    if [ ! -f "${input_file}" ]; then
        continue
    fi
    if [ -z "$recall_results" ]; then
        recall_results="${input_file}"
    else
        recall_results="${recall_results},${input_file}"
    fi
done
python src/search/merge.py ${topk} ${num_parts} ${recall_results} ${result_file}
echo "Results merge done!"
