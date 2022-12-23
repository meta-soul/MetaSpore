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

import os
import subprocess
import time
from typing import Dict, Tuple

from metasporeflow.flows.metaspore_offline_flow import OfflineTask
from metasporeflow.offline.task.task import Task
from metasporeflow.resources.resource import Resource


class LocalOfflineFlowExecutor():
    def __init__(self, resources):
        from .scheduler_manager import LocalDockerSchedulerManager
        self.offline_local_image = resources.find_by_name(
            "offline_local_flow").data.offlineLocalImage
        self.offline_local_container_name = resources.find_by_name(
            "offline_local_flow").data.offlineLocalContainerName
        self.shared_volume_in_container = resources.find_by_name(
            "demo_metaspore_flow").data.sharedVolumeInContainer
        self._tasks_conf: Tuple[Resource] = resources.find_all(OfflineTask)
        self._tasks: Dict[str, Task] = self._get_tasks(self._tasks_conf)
        self._schedulers: LocalDockerSchedulerManager = LocalDockerSchedulerManager(resources, self._tasks)

    def execute_up(self):
        self._init_local_container()
        self._schedulers.start()

    def execute_down(self):
        self._stop_local_container()
        self._remove_local_container()

    def _get_tasks(self, tasks_conf) -> Dict[str, Task]:
        from .task_manager import TaskManager
        return TaskManager(tasks_conf).get_tasks()

    def _init_local_container(self):
        if not self._is_local_offline_container_active():
            volume = "%s/volumes:%s" % (os.getcwd(),
                                        self.shared_volume_in_container)
            create_container_cmd = ['docker', 'run', '--volume', volume, '--network=host', '-itd',
                                    '--name', self.offline_local_container_name, self.offline_local_image]
            msg = "start create offline container: %s\n" % self.offline_local_container_name + \
                "cmd: %s" % " ".join(create_container_cmd)
            print(msg)
            subprocess.run(create_container_cmd)
            n = 10
            while not self._is_local_offline_container_active():
                print("Waiting for container to start...")
                n -= 1
                if n == 0:
                    raise Exception(
                        "[failed]: %s Container failed to start" % self.offline_local_container_name)
                time.sleep(3)
            print("[success]: %s has been started successfully" %
                  self.offline_local_container_name)
        else:
            print("Offline-Container: [%s] is already active" %
                  self.offline_local_container_name)

    def _is_local_offline_container_active(self):
        cmd = "echo $( docker container inspect -f '{{.State.Running}}' %s )" % self.offline_local_container_name
        res = subprocess.run(cmd, shell=True, check=True,
                             capture_output=True, text=True)
        return res.stdout.strip() == "true"

    def _stop_local_container(self):
        cmd = "docker stop %s" % self.offline_local_container_name
        subprocess.run(cmd, shell=True)

    def _remove_local_container(self):
        if self._is_local_offline_container_active():
            self._stop_local_container()
        cmd = "docker rm %s" % self.offline_local_container_name
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
