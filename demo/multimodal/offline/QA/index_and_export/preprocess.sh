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

#python src/preprocess/make_baike_qa_data.py --input-jsonline data/baike/baike_qa_1w.json --output-jsonline data/baike/baike_qa_1w.doc.json --start-id 0
nohup python src/preprocess/make_baike_qa_data.py --input-jsonline data/baike/baike_qa_train.json --output-jsonline data/baike/baike_qa_train.doc.json --start-id 0 > preprocess.log 2>&1 &
