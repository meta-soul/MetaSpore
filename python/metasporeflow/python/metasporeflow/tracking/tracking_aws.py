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
        self._create_function()

    def _get_sage_maker_config(self):
        from metasporeflow.flows.sage_maker_config import SageMakerConfig
        sage_maker_resource = self._resources.find_by_type(SageMakerConfig)
        sage_maker_config = sage_maker_resource.data
        return sage_maker_config

    def _init_tracking_params(self):
        self._function_name = self._sage_maker_config.tracking_function_name
        self._image_uri = self._sage_maker_config.tracking_image_uri
        self._ima_role = self._sage_maker_config.tracking_ima_role
        self._security_group_ids = self._sage_maker_config.securityGroups
        self._subnet_ids = self._sage_maker_config.subnets
        self._s3_bucket_name = self._sage_maker_config.s3WorkDir + '/tracking_log'
        self._metasporeflow_tracking_db_enable = self._sage_maker_config.metasporeflow_tracking_db_enable
        self._metasporeflow_tracking_db_type = self._sage_maker_config.metasporeflow_tracking_db_type
        self._metasporeflow_tracking_db_uri = self._sage_maker_config.metasporeflow_tracking_db_uri
        self._metasporeflow_tracking_db_database = self._sage_maker_config.metasporeflow_tracking_db_database
        self._metasporeflow_tracking_db_table = self._sage_maker_config.metasporeflow_tracking_db_table
        self._metasporeflow_tracking_log_buffer_timeout_ms = self._sage_maker_config.metasporeflow_tracking_log_buffer_timeout_ms
        self._metasporeflow_tracking_log_buffer_max_bytes = self._sage_maker_config.metasporeflow_tracking_log_buffer_max_bytes
        self._metasporeflow_tracking_log_buffer_max_items = self._sage_maker_config.metasporeflow_tracking_log_buffer_max_items
        self._region_name = os.environ['AWS_DEFAULT_REGION']
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
