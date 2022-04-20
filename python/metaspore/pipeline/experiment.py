class Experiment(object):
    def __init__(self,
                dic:dict,
                consul_endpoint_prefix=None,
                scene=None,
                model_name=None,
                model_out_path=None,
                model_export_path=None,
                consul_host=None,
                consul_port=None,
                scheduled_time=None
                ):
        self.consul_endpoint_prefix = dic.get("consul_endpoint_prefix") # use .get method to avoid None issue
        self.scene = dic.get("scene")   
        self.experiment_name = dic.get("experiment_name")
        self.model_name = dic.get("model_name")
        self.model_out_path = dic.get("model_out_path")
        self.model_export_path = dic.get("model_export_path")

        time_int = dic.get("ScheduledTime") # use Kubeflow's [[ScheduledTime]], the type is int
        self.scheduled_time = str(time_int) # trans to str

        import os
        self.consul_host = os.environ.get('CONSUL_HOST') # use local env
        port_str = os.environ.get('CONSUL_PORT')
        self.consul_port = int(port_str)        # port(eg: 8500) need tobe int type

    def fill_parameter(self, estimator):
        estimator.consul_endpoint_prefix= self.consul_endpoint_prefix
        estimator.scene = self.scene
        estimator.experiment_name= self.experiment_name
        estimator.model_name = self.model_name
        estimator.consul_host =self.consul_host
        estimator.consul_port = self.consul_port

        estimator.model_version = self.scheduled_time
        estimator.model_out_path = self.get_model_out_path_by_version()   # /out_path/veriosn/
        estimator.model_export_path = self.get_model_export_path_by_version()
    
    def get_consul_endpoint_prefix(self):
        return self.consul_endpoint_prefix
    
    def get_scene(self):
        return self.scene

    def get_experiment_name(self):
        return self.scene

    def get_model_name(self):
        return self.model_name

    def get_scene(self):
        return self.scene

    def get_scheduledTime(self):
        return self.scheduled_time

    def get_model_out_path(self):
        return self.model_out_path
    
    def get_model_export_path(self):
        return self.model_export_path

    def get_consul_host(self):
        return self.consul_host

    def get_consul_port(self):
        return self.consul_port

    def get_model_out_path_by_version(self):
        return self.model_out_path + '%s/' % self.scheduled_time
    
    def get_model_export_path_by_version(self):
        return self.model_export_path + '%s/' % self.scheduled_time