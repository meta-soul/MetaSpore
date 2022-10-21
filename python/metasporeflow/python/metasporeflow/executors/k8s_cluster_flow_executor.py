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

from .flow_executor import FlowExecutor

class K8sClusterFlowExecutor(FlowExecutor):
    def __init__(self, resources):
        super(K8sClusterFlowExecutor, self).__init__(resources)

    async def execute_up(self):
        print('k8s cluster flow up')

    async def execute_down(self):
        print('k8s cluster flow down')

    async def execute_status(self):
        print('k8s cluster flow status')

    async def execute_reload(self):
        print('k8s cluster flow reload')
