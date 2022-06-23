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

from .node import PipelineNode
import metaspore as ms
import subprocess

import sys
sys.path.append('../../') 
from utils import start_logging

class InitSparkNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        logger = start_logging(payload['logging'])

        spark_confs = payload['spark']
        if not spark_confs:
            logger.info("Spark configuration is none")
            return payload
        
        session_confs = spark_confs['session_confs']
        if not session_confs:
            logger.info('Spark session configuration is none')
            return payload

        extended_confs = spark_confs['extended_confs'] or {}
        if spark_confs['pyzip']:
            cwd_path = spark_confs['pyzip']['cwd_path']
            zip_file_path = spark_confs['pyzip']['zip_file_path']
            subprocess.run(['zip', '-r', zip_file_path, 'python'], cwd=cwd_path)
            extended_confs['spark.submit.pyFiles'] = zip_file_path
        
        spark = ms.spark.get_session(local=session_confs['local'],
                                     app_name=session_confs['app_name'],
                                     batch_size=session_confs['batch_size'],
                                     worker_count=session_confs['worker_count'],
                                     server_count=session_confs['server_count'],
                                     worker_memory=session_confs['worker_memory'],
                                     server_memory=session_confs['server_memory'],
                                     coordinator_memory=session_confs['coordinator_memory'],
                                     spark_confs=extended_confs)
        sc = spark.sparkContext
        logger.info('Spark init, version: {}, applicationId: {}, uiWebUrl:{}'\
                    .format( sc.version, sc.applicationId, sc.uiWebUrl))
        payload['spark'] = spark
        return payload