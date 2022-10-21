import subprocess
from string import Template

def k8s_template_by_file(filename, data):
    with open(filename, 'r') as template_file:
        template_content = template_file.read()
        tempTemplate = Template(template_content)
        print(tempTemplate.safe_substitute(data))

def k8s_template(template_content, data):
    tempTemplate = Template(template_content)
    return tempTemplate.safe_substitute(data)

def create_k8s_service(service_name, template_content, data):
    service_k8s_filename = "k8s-%s.yaml" %(service_name)
    k8s_content = k8s_template(template_content, data)
    if not k8s_content:
        print("service: %s k8s config is empty!" % service_name)
        return False
    service_k8s_file = open(service_k8s_filename, "w")
    service_k8s_file.write(k8s_content)
    service_k8s_file.close()
    clear_ret = subprocess.run("kubectl delete -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    ret = subprocess.run("kubectl create -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if ret.returncode != 0:
        print("service: %s k8s create fail!" % (service_name), ret)
        return False
    return True

def delete_k8s_service(service_name, template_content, data):
    service_k8s_filename = "k8s-%s.yaml" %(service_name)
    k8s_content = k8s_template(template_content, data)
    if not k8s_content:
        print("service: %s k8s config is empty!" % service_name)
        return False
    service_k8s_file = open(service_k8s_filename, "w")
    service_k8s_file.write(k8s_content)
    service_k8s_file.close()
    ret = subprocess.run("kubectl delete -f %s" % service_k8s_filename, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if ret.returncode != 0:
        print("service: %s k8s delete fail!" % (service_name), ret)
        return False
    return True

def k8s_service(service_name, command, template, keys, data, default):
    if not data:
        data = default
    if not data and not default:
        print("service:%s config data is empty! %s fail" % (service_name, command))
        return
    for key in keys:
        if data.get(key) is None:
            data[key] = default.get(key)
    if command == "up":
        if create_k8s_service(service_name, template, data):
            print("%s k8s service create successfully!"%service_name)
        else:
            print("%s k8s service create fail!" % service_name)
    elif command == "down":
        if create_k8s_service(service_name, template, data):
            print("%s k8s service delete successfully!"%service_name)
        else:
            print("%s k8s service delete fail!" % service_name)

def k8s_consul(data, command):
    default = {'image': 'consul:1.13.1',
         'port': 8500,
         'node_port': 30500,
         'name': 'consul'}
    from metasporeflow.online.k8s_template.consul_template import template, keys
    k8s_service("consul-server", command, template, keys, data, default)

def k8s_recommend(data, command):
    default = {'image': 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service:1.0.0',
         'port': 13013,
         'node_port': 30313,
         'consul_port': 8500,
         'name': 'recommend'}
    from metasporeflow.online.k8s_template.recommend_template import template, keys
    k8s_service("recommend-service", command, template, keys, data, default)

def k8s_model(data, command):
    default = {'image': 'swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1',
         'port': 50000,
         'node_port': 30188,
         'name': 'model-serving'}
    from metasporeflow.online.k8s_template.model_template import template, keys
    k8s_service("model-serving", command, template, keys, data, default)


if __name__=='__main__':
    k8s_consul({}, "down")
    k8s_consul({}, "up")
    k8s_recommend({}, "down")
    k8s_recommend({}, "up")
