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

img_dir=$1
img_base_url=$2

mkdir -p data/unsplash-25k

python src/preprocess/make_imgdocs_from_dir.py --img-dir ${img_dir} --output-jsonline data/unsplash-25k/unsplash_25k.json --base-url ${img_base_url}
