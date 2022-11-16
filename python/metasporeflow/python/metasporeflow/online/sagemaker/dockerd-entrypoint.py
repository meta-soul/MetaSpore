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
config_path = os.path.join(prefix, config_name)

def process_model_data():
    print("/opt/ml/:", os.path.exists(prefix))
    if os.path.exists(prefix):
        print("list:", os.listdir(prefix))
    else:
        print("list opt:", os.listdir("/opt"))
        print("list root:", os.listdir("/root"))
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
    return config_path, model_info_file


def serve():
    config_name, model_info_file = process_model_data()
    print("model_serving start!")
    asyncio.run(_start_model_serving(50000, "/data/models"))
    print("recommend service start!")
    asyncio.run(_start_recommend_service(8080, "false", model_info_file, config_name))
    print("model handle start!")


async def _start_recommend_service(service_port, consul_enable, init_model_info=model_info_file,
                                   init_config=config_path):
    recommend_base_cmd = "java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar " \
                         "/opt/recommend-service.jar  --init_config={} --init_config_format=yaml " \
                         "--init_model_info={}".format(init_config, init_model_info)
    print("recommend_base_cmd:", recommend_base_cmd)
    subprocess.Popen(recommend_base_cmd, shell=True, env={"SERVICE_PORT": service_port, "CONSUL_ENABLE": consul_enable})


async def _start_model_serving(grpc_listen_port, init_load_path):
    if os.path.isfile(init_load_path):
        os.remove(init_load_path)
    if not os.path.exists(init_load_path):
        os.makedirs(init_load_path)
    serving_cmd = "/opt/metaspore-serving/bin/metaspore-serving-bin -grpc_listen_port {} -init_load_path {}".format(
        grpc_listen_port, init_load_path)
    subprocess.Popen(serving_cmd, shell=True)


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
