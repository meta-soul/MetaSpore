from typing import Dict, Tuple
import subprocess
import os
import time
from metasporeflow.flows.metaspore_oflline_flow import OfflineScheduler, OfflineTask
from metasporeflow.offline.scheduler.offline_crontab_scheduler import OfflineCrontabScheduler
from metasporeflow.offline.scheduler.scheduler import Scheduler
from metasporeflow.offline.task.offline_python_task import OfflinePythonTask
from metasporeflow.offline.task.task import Task
from metasporeflow.resources.resource import Resource
from metasporeflow.offline.scheduler.scheduler_type import SchedulerTpye

_SCHEDULER_TYPES = {'OfflineCrontabScheduler': OfflineCrontabScheduler}
_TASK_TYPES = {'OfflinePythonTask': OfflinePythonTask}


class LocalOfflineFlowExecutor():
    def __init__(self, resources):
        self.offline_local_image = resources.find_by_name(
            "offline_local_flow").data.offlineLocalImage
        self.offline_local_container_name = resources.find_by_name(
            "offline_local_flow").data.offlineLocalContainerName
        self.shared_volume_in_container = resources.find_by_name(
            "demo_metaspore_flow").data.sharedVolumeInContainer
        self._tasks_conf: Tuple[Resource] = resources.find_all(OfflineTask)
        self._tasks: Dict[str, Task] = self._get_tasks(self._tasks_conf)
        self._schedulers: LocalOfflineFlowExecutor.Schedulers = self.Schedulers(resources,
                                                                                self._tasks)

    def execute_up(self):
        self._init_local_container()
        self._schedulers.start()

    def execute_down(self):
        self._stop_local_container()
        self._remove_local_container()

    def _get_tasks(self, tasks_conf) -> Dict[str, Task]:
        return self.Tasks(tasks_conf).get_tasks()

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

    class Schedulers:

        def __init__(self, resources, tasks):
            self._offline_local_container_name = resources.find_by_name(
                "offline_local_flow").data.offlineLocalContainerName
            self._schedulers_conf: Tuple[Resource] = resources.find_all(
                OfflineScheduler)
            self._tasks = tasks
            self._schedulers: Dict[str, Scheduler] = self._get_schedulers()

        def start(self):
            self._init_crontab_env()
            for scheduler in self._schedulers.values():
                scheduler.publish()

        def _get_schedulers(self) -> Dict[str, Scheduler]:
            schedulers: Dict[str, Scheduler] = {}
            for scheduler_conf in self._schedulers_conf:
                scheduler_name = scheduler_conf.name
                scheduler_type = scheduler_conf.kind
                scheduler = None
                if scheduler_type == SchedulerTpye.OfflineCrontabScheduler.value:
                    scheduler = OfflineCrontabScheduler(
                        scheduler_conf, self._tasks, self._offline_local_container_name)
                schedulers[scheduler_name] = scheduler
            return schedulers

        def _create_scheduler(self, scheduler_type, schedulers_conf):
            try:
                return _SCHEDULER_TYPES[scheduler_type](schedulers_conf, self._tasks)
            except Exception:
                raise Exception('Invalid scheduler type')

        def _init_crontab_env(self):
            clear_crontab_history = 'crontab -r'
            clear_crontab_history_cmd = ['docker', 'exec', '-i', self._offline_local_container_name,
                                         '/bin/bash', '-c', clear_crontab_history]
            subprocess.run(clear_crontab_history_cmd)

    class Tasks:

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
