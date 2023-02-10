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
    def _aws_region(self):
        import re
        pattern = r's3\.([A-Za-z0-9\-]+?)\.amazonaws\.com(\.cn)?$'
        s3_endpoint = self._sage_maker_config.s3Endpoint
        match = re.match(pattern, s3_endpoint)
        if match is None:
            message = 'invalid s3 endpoint %r' % s3_endpoint
            raise RuntimeError(message)
        aws_region = match.group(1)
        return aws_region

    @property
    def _s3_config_dir(self):
        import os
        s3_work_dir = self._sage_maker_config.s3WorkDir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        config_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'config')
        return config_dir

    @property
    def _s3_model_dir(self):
        import os
        s3_work_dir = self._sage_maker_config.s3WorkDir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        model_dir = os.path.join(flow_dir, 'scene', self._scene_name, 'model')
        model_dir = os.path.join(model_dir, 'export', self._model_version)
        return model_dir

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

    def _create_training_job_config(self):
        # NOTE: Default offline training docker image
        repo_url = '132825542956.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dmetasoul-repo'
        docker_image = '%s/metaspore-spark-training-release:v1.1.2-sagemaker-entrypoint' % repo_url
        role_arn = self._sage_maker_config.roleArn
        security_groups = self._sage_maker_config.securityGroups
        subnets = self._sage_maker_config.subnets
        s3_endpoint = self._sage_maker_config.s3Endpoint
        s3_work_dir = self._sage_maker_config.s3WorkDir
        s3_config_dir = self._s3_config_dir.rstrip('/') + '/'
        s3_output_path = self._s3_model_dir.rstrip('/') + '/'
        channel_name = 'metaspore'
        metaspore_entrypoint = 'bash /opt/ml/input/data/%s/custom_entrypoint.sh' % channel_name
        job_config = dict(
            TrainingJobName=self._training_job_name,
            AlgorithmSpecification={
                'TrainingImage': docker_image,
                'TrainingInputMode': 'Pipe',
            },
            RoleArn=role_arn,
            InputDataConfig=[
                {
                    'ChannelName': channel_name,
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataDistributionType': 'FullyReplicated',
                            'S3DataType': 'S3Prefix',
                            'S3Uri': s3_config_dir,
                        },
                    },
                    'InputMode': 'File',
                },
            ],
            OutputDataConfig={
                'S3OutputPath': s3_output_path,
            },
            ResourceConfig={
                # NOTE: Default to ml.m5.2xlarge with 8 vCPUs and 32 GiB Memory
                'InstanceType': 'ml.m5.2xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 20,
            },
            VpcConfig={
                'SecurityGroupIds': security_groups,
                'Subnets': subnets,
            },
            StoppingCondition={
                # NOTE: Default to 20 hours
                'MaxRuntimeInSeconds': 72000,
                'MaxWaitTimeInSeconds': 72000,
            },
            EnableNetworkIsolation=False,
            EnableInterContainerTrafficEncryption=False,
            EnableManagedSpotTraining=False,
            Environment={
                'AWS_ENDPOINT': s3_endpoint,
                'METASPORE_ENTRYPOINT': metaspore_entrypoint,
                'METASPORE_FLOW_S3_WORK_DIR': s3_work_dir,
                'METASPORE_FLOW_SCENE_NAME': self._scene_name,
                # NOTE: only one NN model is supported for the moment
                'METASPORE_FLOW_MODEL_NAME': 'widedeep',
                'METASPORE_FLOW_MODEL_VERSION': self._model_version,
                # NOTE: incremental training is not supported for the moment
                'METASPORE_FLOW_LAST_MODEL_VERSION': '',
                # TODO: cf: check this later
                'SPARK_JAVA_OPTS': '-Djava.io.tmpdir=/opt/spark/work-dir',
            },
            RetryStrategy={
                # NOTE: MaximumRetryAttempts must be positive integer >= 1.
                'MaximumRetryAttempts': 1
            }
        )
        return job_config

    def _get_boto3_client_config(self):
        from botocore.config import Config
        config = Config(connect_timeout=5, read_timeout=60, retries={'max_attempts': 20})
        return config

    def _get_training_job_status(self, job_name):
        import boto3
        import botocore
        config = self._get_boto3_client_config()
        client = boto3.client('sagemaker', self._aws_region, config=config)
        try:
            response = client.describe_training_job(TrainingJobName=job_name)
        except botocore.exceptions.ClientError as ex:
            message = "training job %r not found" % job_name
            raise RuntimeError(message) from ex
        status = response['TrainingJobStatus']
        return status

    def _wait_training_job(self, job_name):
        import time
        counter = 0
        while True:
            status = self._get_training_job_status(job_name)
            if counter > 72000:
                message = 'fail to wait training job %r' % job_name
                raise RuntimeError(message)
            if counter % 60 == 0:
                print('Wait training job %r ... [%s]' % (job_name, status))
            if status in ('Completed', 'Failed', 'Stopped'):
                return status
            time.sleep(60)
            counter += 60

    def _get_model_paths(self):
        import boto3
        from urllib.parse import urlparse
        config = self._get_boto3_client_config()
        s3 = boto3.client('s3', self._aws_region, config=config)
        results = urlparse(self._s3_model_dir, allow_fragments=False)
        bucket = results.netloc
        prefix = results.path.strip('/') + '/'
        model_paths = dict()
        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/').get('CommonPrefixes')
        if objects is not None:
            for obj in objects:
                dir_name = obj.get('Prefix')[len(prefix):].strip('/')
                dir_url = 's3://%s/%s%s/' % (bucket, prefix, dir_name)
                model_paths[dir_name] = dir_url
        return model_paths

    def _update_online_service(self):
        import pprint
        from metasporeflow.online.sagemaker_executor import SageMakerExecutor
        print('models:')
        model_paths = self._get_model_paths()
        pprint.pprint(model_paths)
        if len(model_paths) != 1:
            # NOTE: only one NN model is supported for the moment
            message = "expect one model; found %d" % (len(model_paths))
            raise RuntimeError(message)
        executor = SageMakerExecutor(self._resources)
        executor.execute_reload(models=model_paths)

    def _create_training_job(self):
        import boto3
        job_config = self._create_training_job_config()
        config = self._get_boto3_client_config()
        sagemaker_client = boto3.client('sagemaker', self._aws_region, config=config)
        response = sagemaker_client.create_training_job(**job_config)
        print('response: %s' % response)
        status = self._wait_training_job(self._training_job_name)
        print('status: %s' % status)
        if status == 'Completed':
            self._update_online_service()

    def run(self):
        self._parse_args()
        with self:
            self._create_training_job()

def main():
    runner = CrontabModelArtsRunner()
    runner.run()

if __name__ == '__main__':
    main()
