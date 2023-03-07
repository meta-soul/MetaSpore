from metasporeflow.tracking.tracking import Tracking
import subprocess
from string import Template


class TrackingK8s(Tracking):
    def __init__(self, resources):
        super(TrackingK8s, self).__init__(resources)
        self._k8s_tracking_config = self._get_k8s_tracking_config()
        self.upload_type = self._k8s_tracking_config.uploadType
        self.upload_path = self._k8s_tracking_config.uploadPath
        self.access_key_id = self._k8s_tracking_config.accessKeyId
        self.secret_access_key = self._k8s_tracking_config.secretAccessKey
        self.endpoint = self._k8s_tracking_config.endpoint
        self.upload_when = self._k8s_tracking_config.uploadWhen
        self.upload_interval = self._k8s_tracking_config.uploadInterval
        self.upload_backup_count = self._k8s_tracking_config.uploadBackupCount

    # def start_docker(self):
    #     from metasporeflow.tracking.k8s.docker import Docker
    #     docker = Docker(self._resources)
    #     docker.start()

    def _get_k8s_tracking_config(self):
        from metasporeflow.flows.k8s_tracking_config import K8sTrackingConfig
        k8s_tracking_resource = self._resources.find_by_type(K8sTrackingConfig)
        k8s_tracking_config = k8s_tracking_resource.data
        return k8s_tracking_config

    def is_k8s_active(self, service_name, namespace="saas-demo"):
        cmd = "echo $( kubectl describe -n {} service {} )".format(namespace, service_name)
        res = subprocess.run(cmd, shell=True, check=True,
                             capture_output=True, text=True)
        return res.stderr.strip() == ""

    def k8s_template(self, template_content, data):
        tempTemplate = Template(template_content)
        return tempTemplate.safe_substitute(data)

    def generate_k8s_file(self, service_name, k8s_content):
        service_k8s_filename = self._service_k8s_filename_template % (service_name)
        service_k8s_file = open(service_k8s_filename, "w")
        service_k8s_file.write(k8s_content)
        service_k8s_file.close()
        return service_k8s_filename

    def create_tracking_service(self, service_name, template_content, data):
        k8s_content = self.k8s_template(template_content, data)
        if not k8s_content:
            print("service: %s k8s config is empty!" % service_name)
            return False
        service_k8s_filename = self.generate_k8s_file(service_name, k8s_content)
        clear_ret = subprocess.run("kubectl delete -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        ret = subprocess.run("kubectl create -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        if ret.returncode != 0:
            print("service: %s k8s create fail!" % (service_name), ret)
            return False
        return True