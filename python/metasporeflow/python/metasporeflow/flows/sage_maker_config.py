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

from attrs import frozen
from typing import Tuple, Optional


@frozen
class SageMakerConfig:
    roleArn: str
    securityGroups: Tuple[str, ...]
    subnets: Tuple[str, ...]
    s3Endpoint: str
    s3WorkDir: str
    enableTracking: Optional[bool] = False
    trackingDbUri: Optional[str] = None
    trackingDbDatabase: Optional[str] = 'tracking'
    trackingDbTable: Optional[str] = 'tracking'
    trackingLogBufferTimeoutMs: Optional[int] = 1000
    trackingLogBufferMaxBytes: Optional[int] = 262144
    trackingLogBufferMaxItems: Optional[int] = 10000
