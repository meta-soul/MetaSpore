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

def download_dataset():
    import glob
    import subprocess
    GLOB_PATTERN = 'data/**/*.csv'
    NUM_FILES = 24 + 24
    if len(glob.glob(GLOB_PATTERN)) == NUM_FILES:
        print('MetaSpore demo dataset already downloaded')
        return
    string = "rm -rf data && "
    string += "mkdir -p data/train && "
    string += "cd data/train && "
    string += "curl -L -O https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/demo/train/day_{$(seq -s ',' 0 23)}_0.001_train.csv && "
    string += "cd ../.. && "
    string += "mkdir -p data/test && "
    string += "cd data/test && "
    string += "curl -L -O https://ks3-cn-beijing.ksyuncs.com/dmetasoul-bucket/demo/test/day_{$(seq -s ',' 0 23)}_0.001_test.csv && "
    string += "cd ../.. && "
    string += "echo OK: criteo"
    args = string,
    subprocess.check_call(args, shell=True, stderr=subprocess.PIPE)
    if len(glob.glob(GLOB_PATTERN)) == NUM_FILES:
        print('MetaSpore demo dataset downloaded')
    else:
        message = "Failed to download the MetaSpore demo dataset."
        raise RuntimeError(message)
