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
from .scheduler import Scheduler
# TODO: generate sagemaker entrypoint
#from .k8s_job_config_generator import K8sJobConfigGenerator
from ..utils.file_util import FileUtil

class OfflineSageMakerScheduler(Scheduler):
    def __init__(self, scheduler_conf, tasks):
        super().__init__(scheduler_conf, tasks)

    def publish(self):
        # TODO: cf: use cronjob instead of batch job
        #self._create_offline_cronjob()
        #self._run_offline_batch_job_once()
        print('sagemaker offline flow published')

    def destroy(self):
        #self._delete_offline_cronjob()
        #self._delete_offline_batch_job()
        print('sagemaker offline flow destroyed')

    def _get_job_command(self):
        task_commands = map(lambda x: x.execute, self._dag_tasks)
        job_command = ' && '.join(task_commands)
        return job_command

    def _generate_job_config(self, for_cronjob=False):
        job_command = self._get_job_command()
        generator = K8sJobConfigGenerator(self._scheduler_conf, job_command)
        text = generator.generate_job_config(for_cronjob)
        return text

    def _log_job_config(self, title, text):
        print(title)
        print(f'\033[38;5;240m{text}\033[m')

    def _create_offline_cronjob(self):
        text = self._generate_job_config(for_cronjob=True)
        self._log_job_config('Create offline cronjob:', text)
        try:
            args = ['kubectl', 'apply', '-f', '-']
            subprocess.run(args, input=text.encode('utf-8'), check=True)
            print('Create offline cronjob succeeded.')
        except subprocess.CalledProcessError:
            self._log_job_config('Create offline cronjob failed:', text)
            raise

    def _delete_offline_cronjob(self):
        text = self._generate_job_config(for_cronjob=True)
        self._log_job_config('Delete offline cronjob:', text)
        try:
            args = ['kubectl', 'delete', '-f', '-']
            subprocess.run(args, input=text.encode('utf-8'))
            print('Delete offline cronjob succeeded.')
        except subprocess.CalledProcessError:
            self._log_job_config('Delete offline cronjob failed:', text)
            raise

    def _run_offline_batch_job_once(self):
        text = self._generate_job_config(for_cronjob=False)
        self._log_job_config('Run offline batch job once:', text)
        try:
            args = ['kubectl', 'apply', '-f', '-']
            subprocess.run(args, input=text.encode('utf-8'), check=True)
            print('Run offline batch job succeeded.')
        except subprocess.CalledProcessError:
            self._log_job_config('Run offline batch job failed:', text)
            raise
        # TODO: cf: wait batch job completed

    def _delete_offline_batch_job(self):
        text = self._generate_job_config(for_cronjob=False)
        self._log_job_config('Delete offline batch job:', text)
        try:
            args = ['kubectl', 'delete', '-f', '-']
            subprocess.run(args, input=text.encode('utf-8'))
            print('Delete offline batch job succeeded.')
        except subprocess.CalledProcessError:
            self._log_job_config('Delete offline batch job failed:', text)
            raise
