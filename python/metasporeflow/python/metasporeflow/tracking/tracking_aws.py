#  Copyright 2023 DMetaSoul
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sys

print(sys.path)
from metasporeflow.tracking.tracking import Tracking
import boto3
import time
import os


class TrackingAws(Tracking):
    def __init__(self, resources):
        super(TrackingAws, self).__init__(resources)
        self._sage_maker_config = self._get_sage_maker_config()
        self._function_name = None
        self._image_uri = None
        self._ima_role = None
        self._security_group_ids = None
        self._subnet_ids = None
        self._s3_bucket_name = None
        self._metasporeflow_tracking_db_enable = None
        self._metasporeflow_tracking_db_type = None
        self._metasporeflow_tracking_db_uri = None
        self._metasporeflow_tracking_db_database = None
        self._metasporeflow_tracking_db_table = None
        self._metasporeflow_tracking_log_buffer_timeout_ms = None
        self._metasporeflow_tracking_log_buffer_max_bytes = None
        self._metasporeflow_tracking_log_buffer_max_items = None
        self._region_name = None
        self._iam_resource = None
        self._lambda_client = None
        self._init_tracking_params()

    async def execute_up(self):
        if self._metasporeflow_tracking_enable:
            print('[enableTracking] is True, creating tracking function')
            self._create_function()
        else:
            print('[enableTracking] is False, skip create tracking function')

    def _get_sage_maker_config(self):
        from metasporeflow.flows.sage_maker_config import SageMakerConfig
        sage_maker_resource = self._resources.find_by_type(SageMakerConfig)
        sage_maker_config = sage_maker_resource.data
        return sage_maker_config

    @property
    def _aws_region(self):
        import re
        pattern = r's3\.([A-Za-z0-9\-]+?)\.amazonaws\.com(\.cn)?$'
        sage_maker_config = self._sage_maker_config
        s3_endpoint = sage_maker_config.s3Endpoint
        match = re.match(pattern, s3_endpoint)
        if match is None:
            message = 'invalid s3 endpoint %r' % s3_endpoint
            raise RuntimeError(message)
        aws_region = match.group(1)
        return aws_region

    def _get_bucket_name(self, url):
        from urllib.parse import urlparse
        results = urlparse(url, allow_fragments=False)
        bucket = results.netloc
        return bucket

    def _init_tracking_params(self):
        self._function_name = 'tracking'
        self._image_uri = '132825542956.dkr.ecr.cn-northwest-1.amazonaws.com.cn/dmetasoul-repo/tracking-log-extension-function:latest'
        self._ima_role = self._sage_maker_config.roleArn
        self._security_group_ids = self._sage_maker_config.securityGroups
        self._subnet_ids = self._sage_maker_config.subnets
        self._s3_bucket_name = self._get_bucket_name(self._sage_maker_config.s3WorkDir)
        self._metasporeflow_tracking_enable = self._sage_maker_config.enableTracking
        self._metasporeflow_tracking_db_enable = 'True'
        self._metasporeflow_tracking_db_type = 'mongodb'
        self._metasporeflow_tracking_db_uri = self._sage_maker_config.trackingDbUri
        self._metasporeflow_tracking_db_database = self._sage_maker_config.trackingDbDatabase
        self._metasporeflow_tracking_db_table = self._sage_maker_config.trackingDbTable
        self._metasporeflow_tracking_log_buffer_timeout_ms = str(self._sage_maker_config.trackingLogBufferTimeoutMs)
        self._metasporeflow_tracking_log_buffer_max_bytes = str(self._sage_maker_config.trackingLogBufferMaxBytes)
        self._metasporeflow_tracking_log_buffer_max_items = str(self._sage_maker_config.trackingLogBufferMaxItems)
        self._region_name = self._aws_region
        self._iam_resource = boto3.resource('iam', region_name=self._region_name)
        self._lambda_client = boto3.client('lambda', region_name=self._region_name)

    def _create_function(self):
        response = self._lambda_client.create_function(
            FunctionName=self._function_name,
            Role=self._ima_role,
            Code={'ImageUri': self._image_uri},
            PackageType='Image',
            Description="Tracking Log",
            VpcConfig={
                'SecurityGroupIds': self._security_group_ids,
                'SubnetIds': self._subnet_ids,
            },
            Environment={
                'Variables': {
                    'S3_BUCKET_NAME': self._s3_bucket_name,
                    'METASPOREFLOW_TRACKING_DB_ENABLE': self._metasporeflow_tracking_db_enable,
                    'METASPOREFLOW_TRACKING_DB_TYPE': self._metasporeflow_tracking_db_type,
                    'METASPOREFLOW_TRACKING_DB_URI': self._metasporeflow_tracking_db_uri,
                    'METASPOREFLOW_TRACKING_DB_DATABASE': self._metasporeflow_tracking_db_database,
                    'METASPOREFLOW_TRACKING_DB_TABLE': self._metasporeflow_tracking_db_table,
                    'METASPOREFLOW_TRACKING_LOG_BUFFER_TIMEOUT_MS': self._metasporeflow_tracking_log_buffer_timeout_ms,
                    'METASPOREFLOW_TRACKING_LOG_BUFFER_MAX_BYTES': self._metasporeflow_tracking_log_buffer_max_bytes,
                    'METASPOREFLOW_TRACKING_LOG_BUFFER_MAX_ITEMS': self._metasporeflow_tracking_log_buffer_max_items
                }
            }
        )

        if response['ResponseMetadata']['HTTPStatusCode'] in [200, 201]:
            print('OK --> Created AWS Lambda function {}'.format(self._function_name))
            retries = 45  # VPC lambdas take longer to deploy
            while retries > 0:
                response = self._lambda_client.get_function(
                    FunctionName=self._function_name
                )
                state = response['Configuration']['State']
                if state == 'Pending':
                    time.sleep(5)
                    print(
                        'Function is being deployed... (status: {})'.format(response['Configuration']['State']))
                    retries -= 1
                    if retries == 0:
                        raise Exception('Function not deployed: {}'.format(response))
                elif state == 'Active':
                    break

            print('Ok --> Function active')
        else:
            msg = 'An error occurred creating/updating function {}: {}'.format(self._function_name, response)
            raise Exception(msg)

    def _update_function_code(self):
        response = self._lambda_client.update_function_code(
            FunctionName=self._function_name,
            ImageUri=self._image_uri,
            Publish=True
        )
        if response['ResponseMetadata']['HTTPStatusCode'] in [200, 201]:
            print('OK --> Updated AWS Lambda function {}'.format(self._function_name))
            retries = 45
            while retries > 0:
                response = self._lambda_client.get_function(
                    FunctionName=self._function_name
                )
                state = response['Configuration']['State']
                if state == 'Pending':
                    time.sleep(5)
                    print(
                        'Function is being deployed... (status: {})'.format(response['Configuration']['State']))
                    retries -= 1
                    if retries == 0:
                        raise Exception('Function not deployed: {}'.format(response))
                elif state == 'Active':
                    break


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        raise RuntimeError('Invalid number of arguments')
    operation = args[0]
    print('Config file: {}'.format(operation))

    print('Starting Tracking AWS')
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.flows.sage_maker_config import SageMakerConfig

    flow_loader = FlowLoader()
    flow_loader._file_name = 'metasporeflow/tracking/test/metaspore-flow.yml'
    resources = flow_loader.load()
    sagemaker_config = resources.find_by_type(SageMakerConfig)
    print(type(sagemaker_config))
    print(sagemaker_config)
    tracking = TrackingAws(resources)

    if operation == 'create':
        tracking._create_function()
    elif operation == 'update':
        tracking._update_function_code()
