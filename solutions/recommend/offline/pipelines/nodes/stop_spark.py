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

from solutions.recommend.offline.utils.logger import start_logging
from .node import PipelineNode

import sys
sys.path.append('../../') 
from utils import start_logging

class StopSparkNode(PipelineNode):
    def __call__(self, **payload) -> dict:
        logger = start_logging(payload['logging'])
        spark = payload['spark']
        if not spark:
            logger.info('Spark session is none')
            return payload
        spark.sparkContext.stop()
        logger.info('Spark session stop')
        return payload