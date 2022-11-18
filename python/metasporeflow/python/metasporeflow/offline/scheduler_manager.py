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

import subprocess
from typing import Dict
from ..resources.resource import Resource
from .scheduler.scheduler_type import SchedulerType
from .scheduler.scheduler import Scheduler
from ..flows.metaspore_offline_flow import OfflineScheduler
from .scheduler.offline_crontab_scheduler import OfflineCrontabScheduler
from .scheduler.offline_k8s_cronjob_scheduler import OfflineK8sCronjobScheduler
from .scheduler.offline_sage_maker_scheduler import OfflineSageMakerScheduler

_SCHEDULER_TYPES = {
    'OfflineCrontabScheduler': OfflineCrontabScheduler,
    'OfflineK8sCronjobScheduler': OfflineK8sCronjobScheduler,
    'OfflineSageMakerScheduler': OfflineSageMakerScheduler,
}

class SchedulerManager:

    def __init__(self, resources, tasks):
        self._resources = resources
        self._schedulers_conf: Tuple[Resource] = resources.find_all(OfflineScheduler)
        self._tasks = tasks
        self._schedulers: Dict[str, Scheduler] = None

    def start(self):
        for scheduler in self._get_schedulers().values():
            scheduler.publish()

    def stop(self):
        for scheduler in self._get_schedulers().values():
            scheduler.destroy()

    def _get_schedulers(self) -> Dict[str, Scheduler]:
        if self._schedulers is not None:
            return self._schedulers
        schedulers: Dict[str, Scheduler] = {}
        for scheduler_conf in self._schedulers_conf:
            scheduler_name = scheduler_conf.name
            scheduler_type = scheduler_conf.kind
            scheduler = None
            if scheduler_type == SchedulerType.OfflineCrontabScheduler.value:
                scheduler = OfflineCrontabScheduler(
                    scheduler_conf, self._tasks, self._offline_local_container_name)
            elif scheduler_type == SchedulerType.OfflineK8sCronjobScheduler.value:
                scheduler = OfflineK8sCronjobScheduler(
                    scheduler_conf, self._tasks)
            elif scheduler_type == SchedulerType.OfflineSageMakerScheduler.value:
                scheduler = OfflineSageMakerScheduler(
                    scheduler_conf, self._tasks)
            else:
                message = f"Invalid scheduler type: {scheduler_type}"
                raise Exception(message)
            schedulers[scheduler_name] = scheduler
        self._schedulers = schedulers
        return schedulers

    def _create_scheduler(self, scheduler_type, schedulers_conf):
        try:
            return _SCHEDULER_TYPES[scheduler_type](schedulers_conf, self._tasks)
        except Exception:
            raise Exception('Invalid scheduler type')


class LocalDockerSchedulerManager(SchedulerManager):

    def __init__(self, resources, tasks):
        super().__init__(resources, tasks)
        self._offline_local_container_name = self._resources.find_by_name(
            "offline_local_flow").data.offlineLocalContainerName

    def start(self):
        self._init_crontab_env()
        super().start()

    def _init_crontab_env(self):
        clear_crontab_history = 'crontab -r'
        clear_crontab_history_cmd = ['docker', 'exec', '-i', self._offline_local_container_name,
                                     '/bin/bash', '-c', clear_crontab_history]
        subprocess.run(clear_crontab_history_cmd)
