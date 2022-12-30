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
            if deploy_mode not in ('Local', 'K8sCluster', 'SageMaker'):
                message = "deployMode must be one of: Local, K8sCluster, SageMaker; "
                message += f"{deploy_mode:r} is invalid"
                raise ValueError(message)
        return deploy_mode
    
    @staticmethod
    def init_flow(scene_name: str, scheduler_mode: str):
        from ..config_yamls.generate_flow import generate_flow
        generate_flow(scene_name, scheduler_mode)

    def create_flow_executor(self):
        from .local_flow_executor import LocalFlowExecutor
        from .k8s_cluster_flow_executor import K8sClusterFlowExecutor
        from .sage_maker_flow_executor import SageMakerFlowExecutor
        deploy_mode = self._get_flow_deploy_mode()
        if deploy_mode == 'Local':
            flow_executor = LocalFlowExecutor(self._resources)
        elif deploy_mode == 'K8sCluster':
            flow_executor = K8sClusterFlowExecutor(self._resources)
        elif deploy_mode == 'SageMaker':
            flow_executor = SageMakerFlowExecutor(self._resources)
        else:
            assert False
        return flow_executor
