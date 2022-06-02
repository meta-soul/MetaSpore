set -x

query_model=$1
passage_model=$2
query_file=$3
passage_data_dir=$4
topk=$5

index_mode=faiss:FlatIP
num_parts=4
part_size=`cat ${passage_data_dir}/part-00 | wc -l`
split_name=dev
query_embs=./data/output/query.embs
passage_embs=./data/output/passage.embs
result_file=./data/output/${split_name}.recall.top${topk}

## encode query
#echo "Encode query starting..."
#sh script/infer_dual_encoder.sh cuda:0 "query" ${query_model} ${query_file} ${query_embs}
#echo "Encode query done!"

# encode passage
echo "Encode passage starting..."
for (( part_id=0; part_id<$num_parts; part_id++ ))
do
    if [ "${part_id}" == "2" ]; then
        continue
    fi
    if [ "${part_id}" == "3" ]; then
        continue
    fi

    device="cuda:${part_id}"
    part_id=`printf "%02d" ${part_id}`
    input_file=${passage_data_dir}/part-${part_id}
    output_file=${passage_embs}.part-${part_id}

    # multi-gpu parallel
    #nohup sh script/infer_dual_encoder.sh "cuda:0" "${part_id}" ${passage_model} ${input_file} ${output_file} &

    sh script/infer_dual_encoder.sh "cuda:0" "${part_id}" ${passage_model} ${input_file} ${output_file}
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
