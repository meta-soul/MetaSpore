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

export PYTHONPATH="$PYTHONPATH:$PWD/src"

mkdir -p ./data/output

#device=cpu
#model_name=DMetaSoul/sbert-chinese-general-v2
#input_file=./data/dev/dev.q.format
#output_file=./data/output/dev.q.embs
#text_indices=0
#batch_size=16
#output_format=numpy

device=$1
part=$2
model_name=$3
input_file=$4
output_file=$5
index_mode=faiss:FlatIP

if [ $part == "query" ]; then
    text_indices=0  # the column index for query
    text_max_len=16
    batch_size=1024
    output_format=numpy
    output_file=${output_file}.npy
else
    text_indices=2  # the column index for passage
    text_max_len=360
    batch_size=1280
    #output_format=faiss:FlatIP
    #output_file=${output_file}.faiss
    output_format=numpy
    output_file=${output_file}.npy
fi

# encoder inference
num_batches=-1
# using a small batches for testing
#num_batches=10
python src/infer/dual_encoder_infer.py --model ${model_name} \
    --input-file ${input_file} \
    --output-file ${output_file} \
    --text-indices ${text_indices} \
    --text-max-len ${text_max_len} \
    --output-format ${output_format} \
    --device ${device} --batch-size ${batch_size} \
    --num-batches ${num_batches}

# embedding index
#if [ $part != "query" ]; then
#    filename=`ls ${output_file} | rev | cut -d'.' -f2 | rev`
#    extname=`ls ${output_file} | rev | cut -d'.' -f1 | rev`
#    if [ "$extname" == "npy" ] && [ -f "${output_file}" ]; then
#        index_file=${filename}.faiss
#        python src/search/index_build.py --emb-file ${output_file} --index-file ${index_file} --index-mode ${index_mode}
#        if [ $? == 0 ]; then
#            rm -rf ${output_file}
#        fi
#    fi
#fi
