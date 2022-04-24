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

from indexing.milvus.utils import get_base_parser, get_collection


def parse_args():
    parser = get_base_parser()
    parser.add_argument(
        "--index-field", type=str, required=True
    )
    parser.add_argument(
        "--vector", type=str, required=True, help="The embedding vector string split by comma"
    )
    parser.add_argument(
        "--limit", type=int, default=10
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    collection = get_collection(args.host, args.port, args.collection_name)
    assert collection is not None, "Collection {} not exists!".format(args.collection_name)

    search_params = {
        "metric_type": args.ann_metric_type, 
        "params": {"nprobe": args.ann_param_nprobe}
    }

    results = collection.search(
        data=[[float(v) for v in args.vector.split(',')]],
        anns_field=args.index_field,
        param=search_params, 
        limit=args.limit,
        expr=None,
        consistency_level="Strong"
    )
    return results

if __name__ == '__main__':
    results = main()
    print(results)
