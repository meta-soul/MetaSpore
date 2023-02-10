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

class FlowExecutorFactory(object):
    def __init__(self, resources):
        self._resources = resources

    def _get_flow_deploy_mode(self):
        from ..flows.metaspore_flow import MetaSporeFlow
        flow_resource = self._resources.find_by_type(MetaSporeFlow)
        deploy_mode = flow_resource.data.deployMode
        if deploy_mode is None:
            deploy_mode = 'Local'
        else:
            if deploy_mode not in ('Local', 'K8sCluster', 'SageMaker', 'ModelArts'):
                message = "deployMode must be one of: Local, K8sCluster, SageMaker, ModelArts; "
                message += f"{deploy_mode:r} is invalid"
                raise ValueError(message)
        return deploy_mode

    def create_flow_executor(self):
        deploy_mode = self._get_flow_deploy_mode()
        if deploy_mode == 'Local':
            from .local_flow_executor import LocalFlowExecutor
            flow_executor = LocalFlowExecutor(self._resources)
        elif deploy_mode == 'K8sCluster':
            from .k8s_cluster_flow_executor import K8sClusterFlowExecutor
            flow_executor = K8sClusterFlowExecutor(self._resources)
        elif deploy_mode == 'SageMaker':
            from .sage_maker_flow_executor import SageMakerFlowExecutor
            flow_executor = SageMakerFlowExecutor(self._resources)
        elif deploy_mode == 'ModelArts':
            from .model_arts_flow_executor import ModelArtsFlowExecutor
            flow_executor = ModelArtsFlowExecutor(self._resources)
        else:
            assert False
        return flow_executor
