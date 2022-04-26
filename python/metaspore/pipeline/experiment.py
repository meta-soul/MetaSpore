class Experiment(object):
    def __init__(self, **kwargs):
        import os
        self.consul_endpoint_prefix = kwargs['consul_endpoint_prefix']
        self.experiment_name = kwargs['experiment_name']
        self.scene = kwargs['scene']
        self.model_name = kwargs['model_name']
        self.train_path = kwargs['train_path']
        self.test_path = kwargs['test_path']            
        self.model_in_path = kwargs['model_in_path']          
        self.model_out_path = kwargs['model_out_path']
        self.model_export_path = kwargs['model_export_path']
        self.consul_host = os.environ.get('CONSUL_HOST')        # local env
        self.consul_port = int(os.environ.get('CONSUL_PORT'))   # local env, str -> int
        self.scheduled_time = self.time_unifier(kwargs['ScheduledTime'],kwargs.get('time_format','%Y%m%d'))
        
    def time_unifier(self, input_time, unified_format):
        if input_time is not None:
            from datetime import datetime
            # Notice that Kubeflow's [[ScheduledTime]] type is int,
            # and its default format is '%Y%m%d%M%S'
            default_format = '%Y%m%d%M%S' 
            datetime_obj = datetime.strptime(str(input_time),default_format)
            # transform to customized format
            unified_time = datetime.strftime(datetime_obj, unified_format)
        else:
            unified_time = None
        return unified_time

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
