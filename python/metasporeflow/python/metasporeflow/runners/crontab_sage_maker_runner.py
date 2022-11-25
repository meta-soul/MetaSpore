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

class CrontabSageMakerRunner(object):
    def __init__(self):
        self._scene_name = None
        self._resources = None
        self._sage_maker_config = None
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
    def _s3_config_dir(self):
        import os
        s3_work_dir = self._sage_maker_config.s3WorkDir
        flow_dir = os.path.join(s3_work_dir, 'flow')
        config_dir = os.path.join(flow_dir, 'scene', self._scene_name)
        return config_dir

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
        from metasporeflow.flows.sage_maker_config import SageMakerConfig
        config_path = self._flow_config_path
        if not os.path.isfile(config_path):
            message = 'MetaSpore flow config of scene %r not found' % self._scene_name
            print(message)
            raise SystemExit(1)
        self._resources = ResourceManager.load(config_path)
        self._sage_maker_config = self._resources.find_by_type(SageMakerConfig).data

    def _set_training_job_name(self):
        import re
        import datetime
        scene_name = re.sub('[^A-Za-z0-9]', '-', self._scene_name)
        time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        training_job_name = '%s-%s-train' % (scene_name, time_tag)
        self._training_job_name = training_job_name

    def _parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(description='runner for MetaSpore Flow crontab SageMaker entry')
        parser.add_argument('-s', '--scene', type=str, required=True,
            help='scene name')
        parser.add_argument('-r', '--redirect-stdio', action='store_true',
            help='redirect stdout and stderr to logs dir')
        args = parser.parse_args()
        self._scene_name = args.scene
        self._redirect_logs = args.redirect_stdio
        self._load_flow_config()
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
        sys.stdout = io.open(self._stdout_path, 'w')
        sys.stderr = io.open(self._stderr_path, 'w')

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
        repo_url = '132825542956.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dmetasoul-repo'
        # TODO: cf: check this later
        docker_image = '%s/metaspore-spark-training-release:v1.1.1-sagemaker-entrypoint' % repo_url
        role_arn = self._sage_maker_config.roleArn
        security_groups = self._sage_maker_config.securityGroups
        subnets = self._sage_maker_config.subnets
        s3_endpoint = self._sage_maker_config.s3Endpoint
        s3_config_dir = self._s3_config_dir
        # TODO: cf: check this later
        s3_output_path = 's3://dmetasoul-test-bucket/demo/sg-demo/ecommerce/output/model/ctr/nn/widedeep/model_export/amazonfashion_widedeep/'
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
            # TODO: cf: check this later
            ResourceConfig={
                'InstanceType': 'ml.m4.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 2,
            },
            VpcConfig={
                'SecurityGroupIds': security_groups,
                'Subnets': subnets,
            },
            # TODO: cf: check this later
            StoppingCondition={
                'MaxRuntimeInSeconds': 7200,
                'MaxWaitTimeInSeconds': 7200,
            },
            EnableNetworkIsolation=False,
            EnableInterContainerTrafficEncryption=False,
            EnableManagedSpotTraining=False,
            Environment={
                'AWS_ENDPOINT': s3_endpoint,
                'METASPORE_ENTRYPOINT': metaspore_entrypoint,
                # TODO: cf: check this later
                'SPARK_JAVA_OPTS': '-Djava.io.tmpdir=/opt/spark/work-dir',
            },
            # TODO: cf: check this later
            RetryStrategy={
                'MaximumRetryAttempts': 1
            }
        )
        return job_config

    def _get_training_job_status(self, job_name):
        import boto3
        import botocore
        client = boto3.client('sagemaker')
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
            if counter > 7200:
                message = 'fail to wait training job %r' % job_name
                raise RuntimeError(message)
            if counter % 60 == 0:
                print('Wait training job %r ... [%s]' % (job_name, status))
            if status in ('Completed', 'Failed', 'Stopped'):
                return status
            time.sleep(1)
            counter += 1

    def _update_online_service(self):
        from metasporeflow.online.sagemaker_executor import SageMakerExecutor
        executor = SageMakerExecutor(self._resources)
        models = dict(
            amazonfashion_widedeep='s3://dmetasoul-test-bucket/demo/demo_metaspore_flow/ecommerce/output/model/ctr/nn/widedeep/model_export/amazonfashion_widedeep/'
        )
        executor.execute_reload(models=models)

    def _create_training_job(self):
        import boto3
        job_config = self._create_training_job_config()
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.create_training_job(**job_config)
        print('response: %s' % response)
        status = self._wait_training_job(self._training_job_name)
        print('status: %s' % status)
        if status == 'Completed':
            self._update_online_service()

    def run(self):
        # TODO: cf: check this later
        import os
        os.environ['AWS_DEFAULT_REGION'] = 'cn-northwest-1'

        self._parse_args()
        with self:
            self._create_training_job()

def main():
    runner = CrontabSageMakerRunner()
    runner.run()

if __name__ == '__main__':
    main()
