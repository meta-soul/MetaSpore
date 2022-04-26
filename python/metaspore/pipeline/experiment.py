class Experiment(object):
    def __init__(self,
                dic:dict,
                consul_endpoint_prefix=None,
                experiment_name = None,
                scene=None,
                model_name=None,
                scheduled_time=None,
                train_path=None,
                train_name=None,
                model_in_path=None,
                model_out_path=None,
                model_export_path=None,
                ):
        import os
        self.consul_endpoint_prefix = dic['consul_endpoint_prefix']
        self.experiment_name = dic['experiment_name']
        self.scene = dic['scene']
        self.model_name = dic['model_name']
        self.train_path = dic['train_path']
        self.test_path = dic['test_path']                   # orgin input, not add version yet                  
        self.model_in_path = dic['model_in_path']           # orgin input, not add version yet
        self.model_out_path = dic['model_out_path']        # orgin input, not add version yet
        self.model_export_path = dic['model_export_path']
        self.consul_host = os.environ.get('CONSUL_HOST')        # use local env
        self.consul_port = int(os.environ.get('CONSUL_PORT'))   # str -> int
        self.scheduled_time = self.time_unifier(dic['ScheduledTime'])
        
    def time_unifier(self, input_time):
        if input_time is not None:
            # recurring run -> accurate to minutes, remove the second info
            # notice that Kubeflow's [[ScheduledTime]] type is int
            input_time = str(input_time)[:-2]
        # is none return none
        return input_time

    def fill_parameter(self, estimator):
        estimator.consul_endpoint_prefix= self.consul_endpoint_prefix
        estimator.experiment_name= self.experiment_name
        estimator.scene = self.scene
        estimator.model_name = self.model_name
        estimator.model_version = self.scheduled_time
        estimator.consul_host =self.consul_host
        estimator.consul_port = self.consul_port

        if self.scheduled_time is not None:
            estimator.model_out_path = self.get_model_out_path_by_version()         # add version
            estimator.model_export_path = self.get_model_export_path_by_version()   # add version
        else:
            estimator.model_out_path =self.get_model_out_path()
            estimator.model_export_path = self.get_model_export_path()

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

    def get_model_in_path(self):
        return self.model_in_path

    def get_model_in_path_by_version(self):
        return self.model_in_path + '%s/' % self.scheduled_time

    def get_model_out_path(self):
        return self.model_out_path

    def get_model_out_path_by_version(self):
        return self.model_out_path + '%s/' % self.scheduled_time
    
    def get_model_export_path(self):
        return self.model_export_path

    def get_model_export_path_by_version(self):
        return self.model_export_path + '%s/' % self.scheduled_time

    def get_train_path(self):
        return self.train_path
        
    def get_train_path_by_version(self):
        return self.train_path + '%s/' % self.scheduled_time
    
    def get_test_path(self):
        return self.test_path

    def get_test_path_by_version(self):
        return self.test_path + '%s/' % self.scheduled_time

    def get_consul_host(self):
        return self.consul_host

    def get_consul_port(self):
        return self.consul_port
