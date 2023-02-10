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
from .sage_maker_entrypoint_generator import SageMakerEntrypointGenerator
from ..utils.file_util import FileUtil

class OfflineModelArtsScheduler(Scheduler):
    def __init__(self, resources, scheduler_conf, tasks):
        super().__init__(resources, scheduler_conf, tasks)

    def publish(self):
        self._set_obs_ak_sk()
        self._upload_config()
        self._save_flow_config()
        self._install_crontab()
        self._run_job_once()

    def destroy(self):
        self._set_obs_ak_sk()
        self._uninstall_crontab()
        self._clear_flow_config()
        self._clear_config()

    def get_status(self):
        self._set_obs_ak_sk()
        recent_training_jobs = self._get_training_jobs()
        if recent_training_jobs:
            last_training_job_name = recent_training_jobs[0]['name']
            last_training_job_status = self._get_training_job_status(last_training_job_name)
            last_training_job_url = self._get_training_job_url(last_training_job_name)
        else:
            last_training_job_name = None
            last_training_job_status = None
            last_training_job_url = None
        if not self._installed_in_crontab():
            status = 'DOWN'
        elif last_training_job_status is None:
            status = 'UP'
        elif last_training_job_status == 'InProgress':
            status = 'TRAINING'
        elif last_training_job_status == 'Failed':
            status = 'FAIL'
        else:
            assert last_training_job_status in ('Completed', 'Stopping', 'Stopped')
            status = 'UP'
        info = {
            'status': status,
            'last_training_job_name': last_training_job_name,
            'last_training_job_status': last_training_job_status,
            'last_training_job_url': last_training_job_url,
            'recent_training_jobs': recent_training_jobs,
        }
        return info

    @property
    def _scene_name(self):
        from metasporeflow.flows.metaspore_flow import MetaSporeFlow
        flow_resource = self._resources.find_by_type(MetaSporeFlow)
        scene_name = flow_resource.name
        return scene_name

    @property
    def _training_job_name_prefix(self):
        import re
        scene_name = re.sub('[^A-Za-z0-9]', '-', self._scene_name)
        name_prefix = '%s-' % scene_name
        return name_prefix

    @property
    def _model_arts_config(self):
        from metasporeflow.flows.model_arts_config import ModelArtsConfig
        model_arts_resource = self._resources.find_by_type(ModelArtsConfig)
        model_arts_config = model_arts_resource.data
        return model_arts_config

    @property
    def _s3_access_key_id(self):
        model_arts_config = self._model_arts_config
        access_key_id = model_arts_config.obsAccessKeyId
        return access_key_id

    @property
    def _s3_secret_access_key(self):
        model_arts_config = self._model_arts_config
        secret_access_key = model_arts_config.obsSecretAccessKey
        return secret_access_key

    @property
    def _s3_endpoint(self):
        model_arts_config = self._model_arts_config
        obs_endpoint = model_arts_config.obsEndpoint
        if obs_endpoint.startswith('http://') or obs_endpoint.startswith('https://'):
            s3_endpoint = obs_endpoint
        else:
            s3_endpoint = 'http://' + obs_endpoint
        return s3_endpoint

    @property
    def _s3_work_dir(self):
        model_arts_config = self._model_arts_config
        obs_work_dir = model_arts_config.obsWorkDir
        s3_work_dir = obs_work_dir.replace('obs://', 's3://')
        return s3_work_dir

    @property
    def _crontab_expr(self):
        crontab_expr = self._scheduler_conf.data.cronExpr
        return crontab_expr

    @property
    def _crontab_command(self):
        import sys
        module_name = 'metasporeflow.runners.crontab_sage_maker_runner'
        scene_name = self._scene_name
        python = sys.executable
        crontab_command = '%s -m %s --scene %s --redirect-stdio' % (python, module_name, scene_name)
        return crontab_command

    @property
    def _crontab_entry(self):
        crontab_expr = self._crontab_expr
        crontab_command = self._crontab_command
        crontab_entry = '%s %s' % (crontab_expr, crontab_command)
        return crontab_entry

    @property
    def _local_config_dir_path(self):
        import os
        config_dir = self._scheduler_conf.data.configDir
        if config_dir is None:
            from metasporeflow.flows.metaspore_flow import MetaSporeFlow
            flow_resource = self._resources.find_by_type(MetaSporeFlow)
            resource_path = flow_resource.path
            resource_dir = os.path.dirname(resource_path)
            config_dir = os.path.join(resource_dir, 'volumes')
        return config_dir

    @property
    def _s3_config_dir_path(self):
        import os
        scene_name = self._scene_name
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        config_dir = os.path.join(flow_dir, 'scene', scene_name, 'config')
        return config_dir

    @property
    def _flow_config_path(self):
        import os
        home_dir = os.path.expanduser('~')
        flow_dir = os.path.join(home_dir, '.metaspore', 'flow')
        scene_dir = os.path.join(flow_dir, 'scene', self._scene_name)
        config_path = os.path.join(scene_dir, 'metaspore-flow.dat')
        return config_path

    def _ensure_trailing_slash(self, path):
        if path.endswith('/'):
            return path
        else:
            return path + '/'

    def _set_obs_ak_sk(self):
        import os
        os.environ['AWS_ACCESS_KEY_ID'] = self._s3_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = self._s3_secret_access_key

    def _generate_entrypoint(self, s3_config_dir_path):
        import os
        import subprocess
        generator = SageMakerEntrypointGenerator(self._dag_tasks)
        text = generator.generate_entrypoint()
        s3_path = os.path.join(s3_config_dir_path, 'custom_entrypoint.sh')
        print('Generate SageMaker entrypoint to %s ...' % s3_path)
        args = ['aws', '--endpoint-url', self._s3_endpoint, 's3', 'cp', '-', s3_path]
        subprocess.run(args, input=text.encode('utf-8'), check=True)

    def _upload_config(self):
        import subprocess
        local_path = self._ensure_trailing_slash(self._local_config_dir_path)
        s3_path = self._ensure_trailing_slash(self._s3_config_dir_path)
        print('Upload algorithm config to %s ...' % s3_path)
        args = ['aws', '--endpoint-url', self._s3_endpoint, 's3', 'sync', '--delete', local_path, s3_path]
        subprocess.check_call(args)
        self._generate_entrypoint(s3_path)

    def _clear_config(self):
        s3_path = self._ensure_trailing_slash(self._s3_config_dir_path)
        print('Clear algorithm config %s ...' % s3_path)
        args = ['aws', '--endpoint-url', self._s3_endpoint, 's3', 'rm', '--recursive', s3_path]
        subprocess.check_call(args)

    def _save_flow_config(self):
        import io
        import os
        config_path = self._flow_config_path
        config_dir = os.path.dirname(config_path)
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        print('Save MetaSpore flow config to %s ...' % config_path)
        self._resources.save(config_path)

    def _clear_flow_config(self):
        import os
        config_path = self._flow_config_path
        config_dir = os.path.dirname(config_path)
        print('Clear MetaSpore flow config %s ...' % config_path)
        if os.path.isfile(config_path):
            os.remove(config_path)
        if os.path.isdir(config_dir) and not os.listdir(config_dir):
            os.rmdir(config_dir)

    def _get_old_crontab_spec(self):
        import subprocess
        try:
            args = ['crontab', '-l']
            output = subprocess.check_output(args)
            old_spec = output.decode('utf-8')
            return old_spec
        except subprocess.CalledProcessError:
            return ''

    def _make_new_crontab_spec(self, old_spec):
        lines = old_spec.splitlines()
        command = self._crontab_command
        lines = [line for line in lines if line and not line.endswith(command)]
        crontab_entry = self._crontab_entry
        lines.append(crontab_entry)
        text = '\n'.join(lines)
        new_spec = text + '\n'
        return new_spec

    def _filter_crontab_spec(self, old_spec):
        lines = old_spec.splitlines()
        command = self._crontab_command
        lines = [line for line in lines if line and not line.endswith(command)]
        if not lines:
            return ''
        text = '\n'.join(lines)
        new_spec = text + '\n'
        return new_spec

    def _installed_in_crontab(self):
        old_spec = self._get_old_crontab_spec()
        lines = old_spec.splitlines()
        command = self._crontab_command
        for line in lines:
            if line.endswith(command):
                return True
        return False

    def _update_crontab(self, crontab_spec):
        args = ['crontab', '-']
        subprocess.run(args, input=crontab_spec.encode('utf-8'), check=True)

    def _install_crontab(self):
        old_spec = self._get_old_crontab_spec()
        new_spec = self._make_new_crontab_spec(old_spec)
        print('Install crontab entry %r ...' % self._crontab_entry)
        # TODO: cf: check this later
        #self._update_crontab(new_spec)

    def _uninstall_crontab(self):
        old_spec = self._get_old_crontab_spec()
        new_spec = self._filter_crontab_spec(old_spec)
        print('Uninstall crontab command %r ...' % self._crontab_command)
        self._update_crontab(new_spec)

    def _run_job_once(self):
        import shlex
        import subprocess
        command = self._crontab_command
        args = shlex.split(command)
        assert args and args[-1] == '--redirect-stdio'
        args.pop() # pop --redirect-stdio
        # TODO: cf: check this later
        #subprocess.check_call(args)

    def _get_boto3_client_config(self):
        from botocore.config import Config
        config = Config(connect_timeout=5, read_timeout=60, retries={'max_attempts': 20})
        return config

    def _get_training_jobs(self):
        import boto3
        training_jobs = []
        max_training_jobs = 10
        config = self._get_boto3_client_config()
        sagemaker_client = boto3.client('sagemaker', self._aws_region, config=config)
        response = sagemaker_client.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            NameContains=self._training_job_name_prefix,
            MaxResults=10,
        )
        for summary in response['TrainingJobSummaries']:
            job_name = summary['TrainingJobName']
            job_status = summary['TrainingJobStatus']
            training_job = dict(name=job_name, status=job_status)
            training_jobs.append(training_job)
        next_token = response.get('NextToken')
        while len(training_jobs) < max_training_jobs and next_token is not None:
            response = sagemaker_client.list_training_jobs(
                SortBy='CreationTime',
                SortOrder='Descending',
                NameContains=self._training_job_name_prefix,
                MaxResults=10,
                NextToken=next_token,
            )
            for summary in response['TrainingJobSummaries']:
                job_name = summary['TrainingJobName']
                job_status = summary['TrainingJobStatus']
                training_job = dict(name=job_name, status=job_status)
                training_jobs.append(training_job)
                if len(training_jobs) >= max_training_jobs:
                    break
            next_token = response.get('NextToken')
        return training_jobs

    def _get_training_job_status(self, job_name):
        import boto3
        import botocore
        config = self._get_boto3_client_config()
        sagemaker_client = boto3.client('sagemaker', self._aws_region, config=config)
        try:
            response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        except botocore.exceptions.ClientError as ex:
            message = "training job %r not found" % job_name
            raise RuntimeError(message) from ex
        status = response['TrainingJobStatus']
        return status

    def _get_training_job_url(self, job_name):
        url = 'https://%s.console.amazonaws.cn' % self._aws_region
        url += '/sagemaker/home?region=%s' % self._aws_region
        url += '#/jobs/%s' % job_name
        return url
