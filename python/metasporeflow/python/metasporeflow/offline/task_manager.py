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

from typing import Dict

from .task.offline_python_task import OfflinePythonTask
from .task.task import Task

_TASK_TYPES = {'OfflinePythonTask': OfflinePythonTask}

class TaskManager:

    def __init__(self, tasks_conf):
        # todo: validate tasks_conf: check duplicate task name
        self._tasks_conf = tasks_conf

    def get_tasks(self) -> Dict[str, Task]:
        return self._create_tasks()

    def _create_tasks(self) -> Dict[str, Task]:
        tasks: Dict[str, Task] = {}
        for task in self._tasks_conf:
            task_name = task.name
            task_type = task.kind
            task_data = task.data
            task = self._create_task(task_name,
                                     task_type,
                                     task_data)
            tasks[task_name] = task
        return tasks

    def _create_task(self,
                     name,
                     type,
                     data):
        try:
            return _TASK_TYPES[type](name,
                                     type,
                                     data)
        except Exception:
            raise Exception('Invalid scheduler type')
