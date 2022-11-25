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

    def _parse_args(self):
        import argparse
        parser = argparse.ArgumentParser(description='runner for MetaSpore Flow crontab SageMaker entry')
        parser.add_argument('-s', '--scene', type=str, required=True, help='scene name')
        args = parser.parse_args()
        self._scene_name = args.scene
        self._load_flow_config()
        self._set_training_job_name()

    @property
    def _scene_dir(self):
        import os
        home_dir = os.path.expanduser('~')
        flow_dir = os.path.join(home_dir, '.metaspore', 'flow')
        scene_dir = os.path.join(flow_dir, 'scene', self._scene_name)
        return scene_dir

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
        scene_name = re.sub('[^A-Za-z0-9]', '-', self._scene_name)
        time_tag = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        training_job_name = 'metaspore-flow-scene-%s-run-%%s-offline' % (scene_name, time_tag)
        self._training_job_name = training_job_name

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
        import os
        os.environ['AWS_DEFAULT_REGION'] = 'cn-northwest-1'

        if 0:
            import io
            import sys
            import datetime
            tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            stdout_path = os.path.expanduser('~/stdout%s.txt' % tag)
            stderr_path = os.path.expanduser('~/stderr%s.txt' % tag)
            sys.stdout = io.open(stdout_path, 'a')
            sys.stderr = io.open(stderr_path, 'a')

        self._parse_args()
        self._create_training_job()

def main():
    runner = CrontabSageMakerRunner()
    runner.run()

if __name__ == '__main__':
    main()
