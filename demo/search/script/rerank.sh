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

model=$1
pair_file=$2
score_file=$3

echo "Cross score start..."
python src/infer/cross_encoder_infer.py \
    --model ${model} \
    --input-file ${pair_file} \
    --input-q-i 2 --input-p-i 3 \
    --output-file ${score_file} \
    --task-type multiclass --num-labels 2 \
    --batch-size 512 --device cuda:0 
echo "Cross score done!"
