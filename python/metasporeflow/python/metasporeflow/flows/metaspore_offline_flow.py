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
from typing import Tuple, Dict, Optional

@frozen
class OfflineLocalFlow:
    offlineLocalImage: str
    offlineLocalContainerName: str

@frozen
class OfflineScheduler:
    cronExpr: str
    dag: Dict[str, Tuple[str,...]]

@frozen
class OfflineCrontabScheduler(OfflineScheduler):
    pass

@frozen
class SharedConfigVolume:
    name: str
    configmap: str
    mountPath: str

@frozen
class OfflineK8sCronjobScheduler(OfflineScheduler):
    namespace: str
    serviceAccountName: str
    containerImage: str
    sharedConfigVolume: Optional[SharedConfigVolume] = None

@frozen
class OfflineSageMakerScheduler(OfflineScheduler):
    configDir: Optional[str] = None

@frozen
class OfflineModelArtsScheduler(OfflineScheduler):
    configDir: Optional[str] = None

@frozen
class OfflineTask:
    scriptPath: str
    configPath: str

@frozen
class OfflinePythonTask(OfflineTask):
    pass
