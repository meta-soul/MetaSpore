import json
import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError
import asyncio

prefix = "/opt/ml/"
config_name = "recommend-config.yaml"
model_path = os.path.join(prefix, "model")
model_info_file = os.path.join(prefix, "model-infos.json")


def process_model_data():
    print("/opt/ml/:", os.path.exists(prefix))
    if os.path.exists(prefix):
        print("list:", os.listdir(prefix))
    else:
        print("list opt:", os.listdir("/opt"))
        print("list root:", os.listdir("/root"))
    config_path = os.path.join(prefix, config_name)
    if not os.path.exists(config_path) and not os.path.isfile(config_path):
        print("no model config file in data!", config_path)
        return "", ""
    if not os.path.isdir(model_path):
        print("no model found!")
    elif not os.path.isfile(model_info_file):
        model_infos = list()
        for model_name in os.listdir(model_path):
            model_info = dict()
            model_info["modelName"] = model_name
            model_info["version"] = "1"
            model_info["dirPath"] = os.path.join(model_path, model_name)
            model_info["host"] = "127.0.0.1"
            model_info["port"] = 50000
            model_infos.append(model_info)
        with open(model_info_file, "w") as model_file:
            model_file.write(json.dumps(model_infos))
            model_file.flush()
    return config_name, model_info_file


def serve():
    config_name, model_info_file = process_model_data()
    print("model_serving start!")
    asyncio.run(_start_model_serving(50000, "/data/models"))
    print("recommend service start!")
    asyncio.run(_start_recommend_service(8080, "false", model_info_file, config_name))
    print("model handle start!")


async def _start_recommend_service(service_port, consul_enable, init_model_info="", init_config=""):
    recommend_base_cmd = "java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar " \
                         "/opt/recommend-service.jar  --init_config={} --init_config_format=yaml " \
                         "--init_model_info={} --SERVICE_PORT={} --CONSUL_ENABLE={}".format(init_config,
                                                                                            init_model_info,
                                                                                            service_port,
                                                                                            consul_enable)
    p = subprocess.Popen(recommend_base_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('recommend log: {}'.format(line))


async def _start_model_serving(grpc_listen_port, init_load_path):
    if os.path.isfile(init_load_path):
        os.remove(init_load_path)
    if not os.path.exists(init_load_path):
        os.makedirs(init_load_path)
    serving_cmd = "/opt/metaspore-serving/bin/metaspore-serving-bin -grpc_listen_port {} -init_load_path {}".format(
        grpc_listen_port, init_load_path)
    p = subprocess.Popen(serving_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('model_serving log: {}'.format(line))


def train():
    pass


def main():
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "serve":
        serve()
    else:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

    # prevent docker exit
    subprocess.call(["tail", "-f", "/dev/null"])


main()
