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
from metasporeflow.online.online_k8s_executor import OnlineK8sExecutor
from metasporeflow.offline.k8s_cluster_offline_executor import K8sClusterOfflineFlowExecutor


class K8sClusterFlowExecutor(FlowExecutor):
    def __init__(self, resources):
        super(K8sClusterFlowExecutor, self).__init__(resources)
        self.online_executor = OnlineK8sExecutor(self._resources)
        self.offline_executor = K8sClusterOfflineFlowExecutor(self._resources)

    async def execute_up(self):
        print(self._resources)
        print('-------------------------------')
        self.online_executor.execute_up()
        print('online k8s cluster flow up')
        print('-------------------------------')
        self.offline_executor.execute_up()
        print('offline k8s cluster flow up')
        print('-------------------------------')
        print('k8s cluster flow up')

    async def execute_down(self):
        print('-------------------------------')
        self.online_executor.execute_down()
        print('online k8s cluster flow down')
        print('-------------------------------')
        self.offline_executor.execute_down()
        print('offline k8s cluster flow down')
        print('-------------------------------')
        print('k8s cluster flow down')

    async def execute_status(self):
        print('k8s cluster flow status:')
        print('-------------------------------')
        return {
            "online": self.online_executor.execute_status(),
            "offline": self.offline_executor.execute_status(),
        }

    async def execute_reload(self):
        print('-------------------------------')
        self.online_executor.execute_reload()
        print('online k8s cluster flow reload')
        print('-------------------------------')
        self.offline_executor.execute_reload()
        print('offline k8s cluster flow reload')
        print('-------------------------------')
        print('k8s cluster flow reload')

    @staticmethod
    async def execute_update(resource):
        print(resource)
        return OnlineK8sExecutor.execute_update(resource)
