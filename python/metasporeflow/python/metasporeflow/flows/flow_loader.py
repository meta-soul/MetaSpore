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

class FlowLoader(object):
    _NAMESPACE = 'metaspore'
    _FILE_NAME = 'metaspore-flow.yml'

    def __init__(self):
        self._namespace = self._NAMESPACE
        self._file_name = self._FILE_NAME
        self._resource_types = self._get_resource_types()
        self._context = None

    @classmethod
    def _get_resource_types(cls):
        from metasporeflow.online.online_flow import OnlineFlow
        from .metaspore_flow import MetaSporeFlow
        from .sage_maker_config import SageMakerConfig
        from .model_arts_config import ModelArtsConfig
        from .aws_tracking_config import AwsTrackingConfig
        from metasporeflow.flows.metaspore_offline_flow import \
            OfflineScheduler, \
            OfflineCrontabScheduler, \
            OfflineK8sCronjobScheduler, \
            OfflineSageMakerScheduler, \
            OfflineModelArtsScheduler, \
            OfflineTask, \
            OfflinePythonTask, \
            OfflineLocalFlow
        resource_types = (
            MetaSporeFlow,
            SageMakerConfig,
            ModelArtsConfig,
            AwsTrackingConfig,
            OnlineFlow,
            OfflineScheduler,
            OfflineCrontabScheduler,
            OfflineK8sCronjobScheduler,
            OfflineSageMakerScheduler,
            OfflineModelArtsScheduler,
            OfflineTask,
            OfflinePythonTask,
            OfflineLocalFlow,
        )
        return resource_types

    def _create_resource_loader(self):
        from ..resources.resource_loader import ResourceLoader
        resource_loader = ResourceLoader(self._namespace, self._resource_types, self._context)
        return resource_loader

    def load(self):
        resource_loader = self._create_resource_loader()
        resources = resource_loader.load(self._file_name)
        return resources

    def load_resource(self, text):
        resource_loader = self._create_resource_loader()
        return resource_loader.load_text(text)
