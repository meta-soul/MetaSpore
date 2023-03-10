from metasporeflow.tracking.tracking import Tracking
import subprocess
from string import Template
from metasporeflow.tracking.k8s.common import dictToObj
from metasporeflow.flows.k8s_tracking_config import K8sTrackingConfig
import os


class TrackingK8sExecutor(Tracking):
    def __init__(self, resources):
        super(TrackingK8sExecutor, self).__init__(resources)
        self.tracking_resouce = self._get_k8s_tracking_resource(resources)
        self._tracking_conf = self._get_tracking_config()
        self._service_name = "tracking-service"
        self._service_k8s_filename_template = "%s/k8s-%%s.yaml" % os.getcwd()
        self._k8s_content = self._get_k8s_template()
        self._service_k8s_filename = self.generate_k8s_file(self._k8s_content)

    def run(self):
        self.create_tracking_service()
        # self.delete_k8s_service()

    def _get_k8s_tracking_resource(self, resources):
        from metasporeflow.flows.k8s_tracking_config import K8sTrackingConfig
        k8s_tracking_resource = resources.find_by_type(K8sTrackingConfig)
        k8s_tracking_config = k8s_tracking_resource.data
        return k8s_tracking_config

    # def is_k8s_active(self, namespace="saas-demo"):
    #     cmd = "echo $( kubectl describe -n {} service {} )".format(namespace, self._service_name)
    #     res = subprocess.run(cmd, shell=True, check=True,
    #                          capture_output=True, text=True)
    #     return res.stderr.strip() == ""

    def _get_tracking_config(self):
        from metasporeflow.tracking.k8s.template.tracking_template import default
        tracking_conf = {}
        for key, value in default.items():
            if key not in tracking_conf or tracking_conf.get(key) is None:
                tracking_conf[key] = value
        tracking_conf_attr = [key for key in dir(self.tracking_resouce) if not key.startswith('__')]
        for key in tracking_conf_attr:
            value = getattr(self.tracking_resouce, key)
            if value is not None:
                tracking_conf[key] = value
        return tracking_conf

    def _get_k8s_template(self):
        from metasporeflow.tracking.k8s.template.tracking_template import template
        tempTemplate = Template(template)
        return tempTemplate.safe_substitute(self._tracking_conf)

    def generate_k8s_file(self, k8s_content):
        service_k8s_filename = self._service_k8s_filename_template % (self._service_name)
        service_k8s_file = open(service_k8s_filename, "w")
        service_k8s_file.write(k8s_content)
        service_k8s_file.close()
        return service_k8s_filename

    def create_tracking_service(self):

        # clear_ret = subprocess.run("kubectl delete -f %s" % self._service_k8s_filename, shell=True,
        #                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")

        ret = subprocess.run("kubectl create -f %s" % self._service_k8s_filename, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        print(ret)
        if ret.returncode != 0:
            print("service: %s k8s create fail!" % (self._service_name), ret)
            return False
        return True

    def delete_k8s_service(self):
        ret = subprocess.run("kubectl delete -f %s" % self._service_k8s_filename, shell=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        if ret.returncode != 0:
            print("service: %s k8s delete fail!" % (self._service_name), ret)
            return False

    def gen_tracking_config(self):
        tracking_data = {}
        if not self.configure:
            return tracking_data
        print(self.configure.accessKeyId)


if __name__ == '__main__':
    from metasporeflow.flows.flow_loader import FlowLoader

    flow_loader = FlowLoader()
    flow_loader._file_name = 'metasporeflow/tracking/k8s/test/metaspore-flow.yml'
    resources = flow_loader.load()
    resource = resources.find_by_type(K8sTrackingConfig)
    k8s_tracking_config = resource.data
    flow_executor = TrackingK8sExecutor(resources)
    flow_executor.run()
