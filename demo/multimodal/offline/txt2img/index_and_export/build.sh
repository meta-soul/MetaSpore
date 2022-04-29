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

export PYTHONPATH="${PYTHONPATH}:./src"

nohup python -u src/indexing/image_build.py --model openai/clip-vit-base-patch32 --doc-file data/unsplash-25k/unsplash_25k.json --index-file data/unsplash-25k/unsplash_25k.index.json --batch-size 256 --shard-size 2048 --doc-key-index image:image_emb --doc-key-values id,name,url > build.log 2>&1 &
