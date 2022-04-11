class Experiment(object):
    def __init__(self,
                dic,
                scene=None,
                model_name=None,
                model_out_path=None,
                model_export_path=None,
                model_version=None,
                consul_host=None,
                consul_port=None,
                worker_count=None,
                server_count=None
                ):
        self.scene = dic["scene"]
        self.experiment_name = dic["experiment_name"]
        self.model_name = dic["model_name"]
        self.model_out_path = dic["model_out_path"]
        self.model_export_path = dic["model_export_path"]
        self.model_version = dic["model_version"]
        self.consul_host = dic["consul_host"]
        self.consul_port = dic["consul_port"]
        self.worker_count = dic["worker_count"]
        self.server_count = dic["server_count"]

    def fill_parameter(self, estimator):
        estimator.model_out_path=self.model_out_path
        estimator.model_export_path=self.model_export_path
        estimator.consul_endpoint_prefix= self.scene
        estimator.consul_host =self.consul_host
        estimator.consul_port = self.consul_port
        estimator.model_version = self.model_version
        estimator.experiment_name= self.experiment_name
        estimator.model_name = self.model_name
        estimator.worker_count = self.worker_count
        estimator.server_count = self.server_count