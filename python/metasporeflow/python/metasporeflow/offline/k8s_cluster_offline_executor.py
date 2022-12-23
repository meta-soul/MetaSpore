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

from typing import Dict, Tuple

from metasporeflow.flows.metaspore_offline_flow import OfflineTask
from metasporeflow.offline.task.task import Task
from metasporeflow.resources.resource import Resource


class K8sClusterOfflineFlowExecutor():
    def __init__(self, resources):
        from .scheduler_manager import SchedulerManager
        self._tasks_conf: Tuple[Resource] = resources.find_all(OfflineTask)
        self._tasks: Dict[str, Task] = self._get_tasks(self._tasks_conf)
        self._schedulers: SchedulerManager = SchedulerManager(resources, self._tasks)

    def execute_up(self):
        self._schedulers.start()

    def execute_down(self):
        self._schedulers.stop()

    def execute_status(self):
        print('offline status is not implemented yet')

    def execute_reload(self):
        print('offline reload is not implemented yet')

    def _get_tasks(self, tasks_conf) -> Dict[str, Task]:
        from .task_manager import TaskManager
        return TaskManager(tasks_conf).get_tasks()
