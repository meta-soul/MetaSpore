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

#python src/indexing/text_build.py --model DMetaSoul/sbert-chinese-qmc-domain-v1 --doc-file data/baike/baike_qa_1w.doc.json --index-file data/baike/baike_qa_1w.doc.index.json --batch-size 16

nohup python src/indexing/text_build.py --model DMetaSoul/sbert-chinese-qmc-domain-v1 --doc-file data/baike/baike_qa_train.doc.json --index-file data/baike/baike_qa_train.doc.index.json --batch-size 128 --device cuda:0 > build.log 2>&1 &
