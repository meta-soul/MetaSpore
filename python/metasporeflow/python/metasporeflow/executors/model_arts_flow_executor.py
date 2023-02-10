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
from metasporeflow.online.model_arts_online_flow_executor import ModelArtsOnlineFlowExecutor
from metasporeflow.offline.model_arts_offline_flow_executor import ModelArtsOfflineFlowExecutor

class ModelArtsFlowExecutor(FlowExecutor):
    def __init__(self, resources):
        super(ModelArtsFlowExecutor, self).__init__(resources)
        self.online_executor = ModelArtsOnlineFlowExecutor(self._resources)
        self.offline_executor = ModelArtsOfflineFlowExecutor(self._resources)

    async def execute_up(self):
        print(self._resources)
        print('-------------------------------')
        # For ModelArts, ``online_executor.execute_up`` will be called by ``offline_executor``.
        self.offline_executor.execute_up()
        print('-------------------------------')
        print('ModelArts flow up')

    async def execute_down(self):
        print('-------------------------------')
        self.offline_executor.execute_down()
        print('offline modelarts flow down')
        print('-------------------------------')
        self.online_executor.execute_down()
        print('online modelarts flow down')
        print('-------------------------------')
        print('modelarts flow down')

    async def execute_status(self):
        import json
        print('modelarts flow status:')
        print('-------------------------------')
        status = {
            "online": self.online_executor.execute_status(),
            "offline": self.offline_executor.execute_status(),
        }
        print(json.dumps(status, separators=(',', ': '), indent=4))
        return status

    async def execute_reload(self):
        print('-------------------------------')
        # The implementation of ``reload`` is the same as ``up`` for the moment.
        self.offline_executor.execute_up()
        print('offline modelarts flow reload')
        print('-------------------------------')
        print('modelarts flow reload')

    @staticmethod
    async def execute_update(resource):
        print(resource)
        from metasporeflow.resources.resource_manager import ResourceManager
        resource_manager = ResourceManager()
        resource_manager.add_resource("online_local_flow", "update_content", resource)
        online_executor = ModelArtsOnlineFlowExecutor(resource_manager)
        return online_executor.execute_update(resource)
