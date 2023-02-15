import json
import os
import shlex
import subprocess
import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

prefix = "/opt/ml/model/"
config_name = "recommend-config.yaml"
model_path = os.path.join(prefix, "model")
model_info_file = os.path.join(prefix, "model-infos.json")
config_path = os.path.join(prefix, config_name)


def process_model_data():
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


def download_file():
    from obs import ObsClient
    from urllib.parse import urlparse
    access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    obs_endpoint = os.environ['OBS_ENDPOINT']
    if obs_endpoint.startswith('http://') or obs_endpoint.startswith('https://'):
        s3_endpoint = obs_endpoint
    else:
        s3_endpoint = 'https://' + obs_endpoint
    obs_model_url = os.environ['OBS_MODEL_URL']
    s = urlparse(obs_model_url, allow_fragments=False)
    bucket_name = s.netloc
    url_path = s.path
    obs_model_path = url_path.lstrip("/")
    local_model_path = prefix + url_path.split("/")[-1]
    obs_client = ObsClient(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        server=s3_endpoint
    )
    try:
        resp = obs_client.getObject(bucket_name, obs_model_path, downloadPath=local_model_path)
        if resp.status < 300:
            logging.info("requestId: %s. url: %s", resp.requestId, resp.body.url)
        else:
            logging.info("errorCode: %s. errorMessage: %s", resp.errorCode, resp.errorMessage)
    except:
        import traceback
        logging.info(traceback.format_exc())
    return local_model_path


def unzip_model_file(local_model_file):
    import tarfile
    import shutil
    # config_file = prefix + "recommend-config.yaml"
    model_info_file = prefix + "model-infos.json"
    with tarfile.open(local_model_file, "r:gz") as tar:
        if os.path.exists(model_info_file):
            os.remove(model_info_file)
        if os.path.exists("model"):
            shutil.rmtree("model")
        tar.extractall(prefix)


def print_dir():
    prefix2 = "/opt/ml/model"
    subprocess.Popen("tree {}".format(prefix2), shell=True)


def prepare_model_data():
    local_model_path = download_file()
    unzip_model_file(local_model_path)

def serve():
    prepare_model_data()
    config_name, model_info_file = process_model_data()
    print("starting model serving")
    asyncio.run(_start_model_serving(50000, prefix + "model"))
    print("starting recommend service")
    logging.info("model_info_file2: %s. config_name: %s", model_info_file, config_name)
    asyncio.run(_start_recommend_service(8080, "false", model_info_file, config_name))


async def _start_recommend_service(service_port, consul_enable, init_model_info=model_info_file,
                                   init_config=config_path):
    recommend_base_cmd = "java -XX:MaxRAMPercentage=50 -Dlogging.level.com.dmetasoul.metaspore=INFO -jar " \
                         "/opt/recommend-service.jar  --init_config={} --init_config_format=yaml " \
                         "--init_model_info={}".format(init_config, init_model_info)
    logging.info("recommend_base_cmd: %s", recommend_base_cmd)

    subprocess.Popen(recommend_base_cmd, shell=True, env={
        "SERVICE_PORT": str(service_port),
        "CONSUL_ENABLE": str(consul_enable),
    })


async def _start_model_serving(grpc_listen_port, init_load_path):
    if os.path.isfile(init_load_path):
        os.remove(init_load_path)
    if not os.path.exists(init_load_path):
        os.makedirs(init_load_path)
    serving_cmd = "/opt/metaspore-serving/bin/metaspore-serving-bin -compute_thread_num 10 -ort_intraop_thread_num 1 -ort_interop_thread_num 1 -grpc_listen_port {} -init_load_path {}".format(
        grpc_listen_port, init_load_path)
    print("serving_cmd:", serving_cmd)
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
