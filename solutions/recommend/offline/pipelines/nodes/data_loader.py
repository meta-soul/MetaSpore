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

from .node import PipelineNode

class DataLoaderNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        dataset = payload['conf']['dataset']
        spark = payload['spark']
        
        payload['train_dataset'] = spark.read.parquet(dataset['train_path'])
        payload['test_dataset'] = spark.read.parquet(dataset['test_path'])
        payload['item_dataset']  = spark.read.parquet(dataset['item_path'])

        return payload