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

from abc import ABC, abstractmethod

class PipelineNode(ABC):
    def __init__(self, node_conf=None, **kwargs):
        self._node_conf = node_conf
    
    def preprocess(self, **payload) -> dict:
        return payload
    
    def postprocess(self, **payload) -> dict:
        return payload
    
    @abstractmethod
    def __call__(self, **payload) -> dict:
        pass