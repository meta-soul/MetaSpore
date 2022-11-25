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
from metasporeflow.online.sagemaker_executor import SageMakerExecutor
from metasporeflow.offline.sage_maker_offline_executor import SageMakerOfflineFlowExecutor


class SageMakerFlowExecutor(FlowExecutor):
    def __init__(self, resources):
        super(SageMakerFlowExecutor, self).__init__(resources)
        self.online_executor = SageMakerExecutor(self._resources)
        self.offline_executor = SageMakerOfflineFlowExecutor(self._resources)

    async def execute_up(self):
        print(self._resources)
        print('-------------------------------')
        # For SageMaker, ``online_executor.execute_up`` will be called by ``offline_executor``.
        self.offline_executor.execute_up()
        print('-------------------------------')
        print('sagemaker flow up')

    async def execute_down(self):
        print('-------------------------------')
        self.offline_executor.execute_down()
        print('offline sagemaker flow down')
        print('-------------------------------')
        self.online_executor.execute_down()
        print('online sagemaker flow down')
        print('-------------------------------')
        print('sagemaker flow down')

    async def execute_status(self):
        print('sagemaker flow status:')
        print('-------------------------------')
        return {
            "online": self.online_executor.execute_status(),
            "offline": self.offline_executor.execute_status(),
        }

    async def execute_reload(self):
        print('-------------------------------')
        # The implementation of ``reload`` is the same as ``up`` for the moment.
        self.offline_executor.execute_up()
        print('offline sagemaker flow reload')
        print('-------------------------------')
        print('sagemaker flow reload')

    @staticmethod
    async def execute_update(resource):
        print(resource)
        from metasporeflow.resources.resource_manager import ResourceManager
        resource_manager = ResourceManager()
        resource_manager.add_resource("online_local_flow", "update_content", resource)
        online_executor = SageMakerExecutor(resource_manager)
        return online_executor.execute_update(resource)
