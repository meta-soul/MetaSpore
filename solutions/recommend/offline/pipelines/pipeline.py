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

import yaml
from .nodes import PipelineNode

class Pipeline(object):
    def __init__(self, conf_path):
        self._nodes = []
        self._conf = dict()
        with open(conf_path, 'r') as stream:
            self._conf = yaml.load(stream, Loader=yaml.FullLoader)
            print('Debug -- load config: ', self._conf)
    
    def add_node(self, node):
        if not isinstance(node, PipelineNode):
            raise TypeError(f"node must be PipelineNode; {node!r} is invalid")
        self._nodes.append(node)
        
    def run(self):
        payload = {'conf': self._conf}
        for node in self._nodes:
            payload = node.preprocess(**payload)
            payload = node(**payload)
            payload = node.postprocess(**payload)

