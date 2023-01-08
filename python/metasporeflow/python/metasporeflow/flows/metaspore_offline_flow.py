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

from ..config import config
from ..config import Optional
from ..config import List
from ..config import Dict

@config
class OfflineLocalFlow:
    offlineLocalImage: str
    offlineLocalContainerName: str

@config
class OfflineScheduler:
    cronExpr: str
    dag: Dict[str, List[str]]

@config
class OfflineCrontabScheduler(OfflineScheduler):
    pass

@config
class SharedConfigVolume:
    name: str
    configmap: str
    mountPath: str

@config
class OfflineK8sCronjobScheduler(OfflineScheduler):
    namespace: str
    serviceAccountName: str
    containerImage: str
    sharedConfigVolume: Optional[SharedConfigVolume] = None

@config
class OfflineSageMakerScheduler(OfflineScheduler):
    configDir: Optional[str] = None

@config
class OfflineTask:
    scriptPath: str
    configPath: str

@config
class OfflinePythonTask(OfflineTask):
    pass
