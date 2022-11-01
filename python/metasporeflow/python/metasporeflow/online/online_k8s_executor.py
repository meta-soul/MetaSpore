import os
import subprocess
import time
from string import Template

from metasporeflow.online.check_service import notifyRecommendService, healthRecommendService
from metasporeflow.online.cloud_consul import putServiceConfig, Consul, putConfigByKey
from metasporeflow.online.online_generator import OnlineGenerator


def is_k8s_active(service_name, namespace="saas-demo"):
    cmd = "echo $( kubectl describe -n {} service {} )".format(namespace, service_name)
    res = subprocess.run(cmd, shell=True, check=True,
                         capture_output=True, text=True)
    return res.stderr.strip() == ""

def k8s_template_by_file(filename, data):
    with open(filename, 'r') as template_file:
        template_content = template_file.read()
        tempTemplate = Template(template_content)
        print(tempTemplate.safe_substitute(data))


def k8s_template(template_content, data):
    tempTemplate = Template(template_content)
    return tempTemplate.safe_substitute(data)


class OnlineK8sExecutor(object):
    def __init__(self, resources):
        self._online_resource = resources.find_by_name("online_local_flow")
        self._generator = OnlineGenerator(resource=self._online_resource)
        self._service_k8s_filename_template = "%s/k8s-%%s.yaml" % os.getcwd()

    def execute_up(self, **kwargs):
        consul_data, recommend_data, model_data = self._generator.gen_k8s_config()
        if consul_data is None or recommend_data is None or model_data is None:
            print("k8s online service config is empty")
            return
        print("*" * 80)
        print(consul_data)
        print(recommend_data)
        print(model_data)
        print("*" * 80)
        self.k8s_consul(consul_data, "up")
        time.sleep(3)
        self.k8s_model(model_data, "up")
        time.sleep(3)
        self.k8s_recommend(recommend_data, "up")
        time.sleep(10)
        online_recommend_config = self._generator.gen_server_config()
        consul_client = Consul("%s.%s" % (consul_data.setdefault("name", "consul-k8s-service"),
                                          consul_data.setdefault("domain", "huawei.dmetasoul.com")), 80)
        putServiceConfig(consul_client, online_recommend_config)

    def execute_down(self, **kwargs):
        consul_data, recommend_data, model_data = self._generator.gen_k8s_config()
        if consul_data is None or recommend_data is None or model_data is None:
            print("k8s online service config is empty")
            return
        self.k8s_recommend(recommend_data, "down")
        self.k8s_model(model_data, "down")
        self.k8s_consul(consul_data, "down")

    def execute_status(self, **kwargs):
        consul_data, recommend_data, model_data = self._generator.gen_k8s_config()
        info = {"status": "UP"}
        if not is_k8s_active(consul_data["name"], consul_data.setdefault("namespace", "saas-demo")):
            info["status"] = "DOWN"
            info["consul"] = "consul k8s service is not up!"
        else:
            info["consul"] = "consul k8s service:{} is up!".format(consul_data["name"])
            info["consul_image"] = consul_data["image"]
            info["consul_port"] = consul_data["port"]
        if not is_k8s_active(recommend_data["name"], recommend_data.setdefault("namespace", "saas-demo")):
            info["status"] = "DOWN"
            info["recommend"] = "recommend k8s service is not up!"
        else:
            info["recommend"] = "recommend k8s service:{} is up!".format(recommend_data["name"])
            info["recommend_image"] = recommend_data["image"]
            info["recommend_port"] = recommend_data["port"]
        if not is_k8s_active(model_data["name"], model_data.setdefault("namespace", "saas-demo")):
            info["status"] = "DOWN"
            info["model"] = "model k8s service is not up!"
        else:
            info["model"] = "model k8s service:{} is up!".format(model_data["name"])
            info["model_image"] = model_data["image"]
            info["model_port"] = model_data["port"]
        if info["status"] == 'UP':
            info["service_status"] = healthRecommendService(
                "%s.%s" % (recommend_data.setdefault("name", "recommend-k8s-service"),
                           recommend_data.setdefault("domain", "huawei.dmetasoul.com")), 80)
            info["status"] = info["service_status"].setdefault("status", "DOWN")
        return info

    @staticmethod
    def execute_update(resource):
        generator = OnlineGenerator(resource=resource)
        consul_data, _, _ = generator.gen_k8s_config()
        if not is_k8s_active(consul_data["name"], consul_data.setdefault("namespace", "saas-demo")):
            return False, "consul k8s service is not up!"
        try:
            online_recommend_config = generator.gen_server_config()
        except Exception as ex:
            return False, "recommend service config generate fail ex:{}!".format(ex.args)
        consul_client = Consul("%s.%s" % (consul_data.setdefault("name", "consul-k8s-service"),
                                          consul_data.setdefault("domain", "huawei.dmetasoul.com")), 80)
        try:
            putServiceConfig(consul_client, online_recommend_config)
        except Exception as ex:
            return False, "put service config to consul fail ex:{}!".format(ex.args)
        return True, "update config successfully!"

    def execute_reload(self, **kwargs):
        new_flow = kwargs.setdefault("resource", None)
        if not new_flow:
            print("config update to None")
            self.execute_down(**kwargs)
        else:
            self._resource = new_flow
            self._generator = OnlineGenerator(resource=self._resource)
            self.execute_up(**kwargs)
        print("online flow reload success!")

    def generate_k8s_file(self, service_name, k8s_content):
        service_k8s_filename = self._service_k8s_filename_template % (service_name)
        service_k8s_file = open(service_k8s_filename, "w")
        service_k8s_file.write(k8s_content)
        service_k8s_file.close()
        return service_k8s_filename

    def create_k8s_service(self, service_name, template_content, data):
        k8s_content = k8s_template(template_content, data)
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

    def delete_k8s_service(self, service_name, template_content, data):
        k8s_content = k8s_template(template_content, data)
        if not k8s_content:
            print("service: %s k8s config is empty!" % service_name)
            return False
        service_k8s_filename = self.generate_k8s_file(service_name, k8s_content)
        ret = subprocess.run("kubectl delete -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        if ret.returncode != 0:
            print("service: %s k8s delete fail!" % (service_name), ret)
            return False
        return True

    def k8s_service(self, service_name, command, template, data, default):
        if not data:
            data = {}
        if not data and not default:
            print("service:%s config data is empty! %s fail" % (service_name, command))
            return
        for key, value in default.items():
            if key not in data or data.get(key) is None:
                data[key] = value
        if command == "up":
            if self.create_k8s_service(service_name, template, data):
                print("%s k8s service create successfully!"%service_name)
            else:
                print("%s k8s service create fail!" % service_name)
        elif command == "down":
            if self.delete_k8s_service(service_name, template, data):
                print("%s k8s service delete successfully!" % service_name)
            else:
                print("%s k8s service delete fail!" % service_name)

    def k8s_consul(self, data, command):
        from metasporeflow.online.k8s_template.consul_template import template, default
        self.k8s_service("consul-server", command, template, data, default)

    def k8s_recommend(self, data, command):
        from metasporeflow.online.k8s_template.recommend_template import template, default
        self.k8s_service("recommend-service", command, template, data, default)

    def k8s_model(self, data, command):
        from metasporeflow.online.k8s_template.model_template import template, default
        self.k8s_service("model-serving", command, template, data, default)


if __name__ == '__main__':
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.online.online_flow import OnlineFlow
    import asyncio

    flow_loader = FlowLoader()
    with open('test/online_local_flow.yml') as input:
        text = input.read()
        online_resource = flow_loader.load_resource(text)
        print(online_resource)
        print(OnlineK8sExecutor.execute_update(online_resource))
    flow_loader._file_name = 'test/metaspore-flow.yml'
    resources = flow_loader.load()

    online_flow = resources.find_by_type(OnlineFlow)
    print(type(online_flow))
    print(online_flow)

    flow_executor = OnlineK8sExecutor(resources)
    print(flow_executor.execute_status())
