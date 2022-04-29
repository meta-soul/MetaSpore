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

from indexing.milvus.utils import get_base_parser, get_collection, drop_collection

def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    drop_collection(args.host, args.port, args.collection_name)

if __name__ == '__main__':
    results = main()
