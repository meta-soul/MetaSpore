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

class CrontabModelArtsRunner(object):
    def __init__(self):
        self._scene_name = None
        self._resources = None
        self._model_arts_config = None
        self._model_version = None
        self._training_job_name = None
        self._redirect_logs = None
        self._pid_fout = None

    @property
    def _scene_dir(self):
        import os
        home_dir = os.path.expanduser('~')
        flow_dir = os.path.join(home_dir, '.metaspore', 'flow')
        scene_dir = os.path.join(flow_dir, 'scene', self._scene_name)
        return scene_dir

    @property
    def _pid_path(self):
        import os
        pid_path = os.path.join(self._scene_dir, 'metaspore-flow.pid')
        return pid_path

    @property
    def _flow_config_path(self):
        import os
        config_path = os.path.join(self._scene_dir, 'metaspore-flow.dat')
        return config_path

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
    def _huawei_cloud_region(self):
        import re
        pattern = r'https?://obs\.([A-Za-z0-9\-]+?)\.myhuaweicloud\.com$'
        s3_endpoint = self._s3_endpoint
        match = re.match(pattern, s3_endpoint)
        if match is None:
            message = 'invalid s3 endpoint %r' % s3_endpoint
            raise RuntimeError(message)
        hwc_region = match.group(1)
        return hwc_region

    @property
    def _model_arts_project_id(self):
        model_arts_config = self._model_arts_config
        project_id = model_arts_config.projectId
        return project_id

    @property
    def _resource_pool_id(self):
        model_arts_config = self._model_arts_config
        resource_pool_id = model_arts_config.resourcePoolId
        return resource_pool_id

    @property
    def _access_key_id(self):
        model_arts_config = self._model_arts_config
        access_key_id = model_arts_config.accessKeyId
        return access_key_id

    @property
    def _secret_access_key(self):
        model_arts_config = self._model_arts_config
        secret_access_key = model_arts_config.secretAccessKey
        return secret_access_key

    @property
    def _s3_config_dir(self):
        import os
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        config_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'config')
        return config_dir

    @property
    def _s3_dummy_input_dir(self):
        import os
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        dummy_input_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'dummy', 'input')
        return dummy_input_dir

    @property
    def _s3_model_dir(self):
        import os
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        model_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'model')
        model_dir = os.path.join(model_dir, 'export', self._model_version)
        return model_dir

    @property
    def _s3_logs_dir(self):
        import os
        s3_work_dir = self._s3_work_dir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        logs_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'logs')
        logs_dir = os.path.join(logs_dir, self._model_version)
        return logs_dir

    @property
    def _logs_dir(self):
        import os
        logs_dir_path = os.path.join(self._scene_dir, 'logs')
        return logs_dir_path

    @property
    def _stdout_path(self):
        import os
        stdout_file_name = '%s.stdout' % self._training_job_name
        stdout_path = os.path.join(self._logs_dir, stdout_file_name)
        return stdout_path

    @property
    def _stderr_path(self):
        import os
        stderr_file_name = '%s.stderr' % self._training_job_name
        stderr_path = os.path.join(self._logs_dir, stderr_file_name)
        return stderr_path

    def _load_flow_config(self):
        import os
        from metasporeflow.resources.resource_manager import ResourceManager
        from metasporeflow.flows.model_arts_config import ModelArtsConfig
        config_path = self._flow_config_path
        if not os.path.isfile(config_path):
            message = 'MetaSpore flow config of scene %r not found' % self._scene_name
            print(message)
            raise SystemExit(1)
        self._resources = ResourceManager.load(config_path)
        self._model_arts_config = self._resources.find_by_type(ModelArtsConfig).data

    def _set_model_version(self):
        import datetime
        self._model_version = datetime.datetime.now().strftime('%Y%m%d-%H%M')

    def _set_training_job_name(self):
        import re
        scene_name = re.sub('[^A-Za-z0-9]', '-', self._scene_name)
        training_job_name = '%s-%s-train' % (scene_name, self._model_version)
        self._training_job_name = training_job_name

    def _parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(description='runner for MetaSpore Flow crontab ModelArts entry')
        parser.add_argument('-s', '--scene', type=str, required=True,
            help='scene name')
        parser.add_argument('-r', '--redirect-stdio', action='store_true',
            help='redirect stdout and stderr to logs dir')
        args = parser.parse_args()
        self._scene_name = args.scene
        self._redirect_logs = args.redirect_stdio
        self._load_flow_config()
        self._set_model_version()
        self._set_training_job_name()

    def _check_unique(self):
        import os
        import fcntl
        pid = os.getpid()
        fd = os.open(self._pid_path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print('Job is already running.')
            raise SystemExit
        self._pid_fout = os.fdopen(fd, 'w')
        print(pid, file=self._pid_fout)
        self._pid_fout.flush()

    def _prune_logs(self):
        import os
        import time
        if not os.path.isdir(self._logs_dir):
            return
        cut = time.time() - 7 * 24 * 3600
        for file_name in os.listdir(self._logs_dir):
            file_path = os.path.join(self._logs_dir, file_name)
            mtime = os.path.getmtime(file_path)
            if mtime < cut:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _redirect_stdio(self):
        import io
        import os
        import sys
        if not self._redirect_logs:
            return
        if not os.path.isdir(self._logs_dir):
            os.makedirs(self._logs_dir)
        # Pass ``buffering=1`` for line buffering.
        sys.stdout = io.open(self._stdout_path, 'w', buffering=1)
        sys.stderr = io.open(self._stderr_path, 'w', buffering=1)

    def __enter__(self):
        self._check_unique()
        self._prune_logs()
        self._redirect_stdio()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        import os
        import sys
        self._pid_fout.close()
        try:
            os.remove(self._pid_path)
        except OSError:
            pass
        sys.stdout.flush()
        sys.stderr.flush()

    def _get_model_arts_session(self):
        from modelarts.session import Session
        session = Session(access_key=self._access_key_id,
                          secret_key=self._secret_access_key,
                          project_id=self._model_arts_project_id,
                          region_name=self._huawei_cloud_region)
        return session

    def _create_model_arts_estimator(self, session):
        import os
        from modelarts.train_params import TrainingFiles
        from modelarts.train_params import OutputData
        from modelarts.estimatorV2 import Estimator
        s3_config_dir = self._s3_config_dir.rstrip('/') + '/'
        s3_output_path = self._s3_model_dir.rstrip('/') + '/'
        s3_logs_dir = self._s3_logs_dir.rstrip('/') + '/'
        obs_config_dir = s3_config_dir.replace('s3://', 'obs://')
        obs_output_path = s3_output_path.replace('s3://', 'obs://')
        obs_logs_dir = s3_logs_dir.replace('s3://', 'obs://')
        # ModelArts boot file (entrypoint) must be named '*.py'.
        training_boot_file = 'custom_entrypoint.py'
        output_data_name = 'output_dir'
        # TODO: cf: check this later
        user_image_url = 'dmetasoul-repo/metaspore-modelarts-training:v0.2-test'
        user_command = 'python %s' % training_boot_file
        train_instance_count = 1
        training_files = TrainingFiles(code_dir=obs_config_dir, boot_file=training_boot_file)
        output_data = OutputData(obs_path=obs_output_path, name=output_data_name)
        env_variables = dict(PYTHONUNBUFFERED='1',
                             AWS_ENDPOINT=self._s3_endpoint,
                             AWS_ACCESS_KEY_ID=self._access_key_id,
                             AWS_SECRET_ACCESS_KEY=self._secret_access_key,
                             METASPORE_FLOW_S3_WORK_DIR=self._s3_work_dir,
                             METASPORE_FLOW_SCENE_NAME=self._scene_name,
                             # NOTE: only one NN model is supported for the moment
                             METASPORE_FLOW_MODEL_NAME='widedeep',
                             METASPORE_FLOW_MODEL_VERSION=self._model_version,
                             # NOTE: incremental training is not supported for the moment
                             METASPORE_FLOW_LAST_MODEL_VERSION='',
                            )
        local_code_dir = '/home/ma-user/modelarts/user-job-dir'
        working_dir = '%s/%s' % (local_code_dir, os.path.basename(s3_config_dir.rstrip('/')))
        job_description = 'This is MetaSpore training job %s.' % self._training_job_name
        estimator = Estimator(session=session,
                              training_files=training_files,
                              outputs=[output_data],
                              parameters=[],
                              user_image_url=user_image_url,
                              user_command=user_command,
                              train_instance_count=train_instance_count,
                              log_url=obs_logs_dir,
                              env_variables=env_variables,
                              local_code_dir=local_code_dir,
                              working_dir=working_dir,
                              job_description=job_description,
                              pool_id=self._resource_pool_id,
                             )
        return estimator

    def _create_dummy_input_data(self):
        from modelarts.train_params import InputData
        s3_dummy_input_dir = self._s3_dummy_input_dir.rstrip('/') + '/'
        obs_dummy_input_dir = s3_dummy_input_dir.replace('s3://', 'obs://')
        input_data_name = 'data_url'
        input_data = InputData(obs_path=obs_dummy_input_dir, name=input_data_name)
        return input_data

    def _get_model_paths(self, obs):
        from urllib.parse import urlparse
        results = urlparse(self._s3_model_dir, allow_fragments=False)
        bucket = results.netloc
        prefix = results.path.strip('/') + '/'
        model_paths = dict()
        objects = obs.listObjects(bucketName=bucket, prefix=prefix, delimiter='/').body.commonPrefixs
        if objects is not None:
            for obj in objects:
                dir_name = obj.prefix[len(prefix):].strip('/')
                dir_url = 's3://%s/%s%s/' % (bucket, prefix, dir_name)
                model_paths[dir_name] = dir_url
        return model_paths

    def _update_online_service(self, session):
        import pprint
        from metasporeflow.online.model_arts_online_flow_executor import ModelArtsOnlineFlowExecutor
        obs = session.get_obs_client()
        model_paths = self._get_model_paths(obs)
        print('models:')
        pprint.pprint(model_paths)
        if len(model_paths) != 1:
            # NOTE: only one NN model is supported for the moment
            message = "expect one model; found %d" % (len(model_paths))
            raise RuntimeError(message)
        executor = ModelArtsOnlineFlowExecutor(self._resources)
        executor.execute_reload(models=model_paths)

    def _create_training_job(self):
        session = self._get_model_arts_session()
        estimator = self._create_model_arts_estimator(session)
        input_data = self._create_dummy_input_data()
        job_instance = estimator.fit(inputs=[input_data],
                                     job_name=self._training_job_name,
                                     wait=True,
                                     show_log=True)
        job_info = job_instance.get_job_info()
        print('job_info: %s' % job_info)
        status = job_info['status']['phase']
        print('status: %s' % status)
        if status == 'Completed':
            self._update_online_service(session)

    def run(self):
        self._parse_args()
        with self:
            self._create_training_job()

def main():
    runner = CrontabModelArtsRunner()
    runner.run()

if __name__ == '__main__':
    main()
