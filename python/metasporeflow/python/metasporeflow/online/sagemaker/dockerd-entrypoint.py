import os
import shlex
import subprocess
import sys
from subprocess import CalledProcessError
import asyncio
from retrying import retry
from sagemaker_inference import model_server

serving_bin="/opt/metaspore-serving/bin/metaspore-serving-bin"
recommend_base_cmd="java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar /opt/recommend-service.jar"
serving_grpc_port_name="-grpc_listen_port"
serving_grpc_port=50000
init_load_path_name="-init_load_path"
init_load_path="/data/models"
service_port=8080
consul_enable="false"

def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50, retry_on_exception=_retry_if_error)
def _start_mms():
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    # os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = '2'
    model_server.start_model_server(handler_service="/home/model-server/model_handler.py:handle")

async def _start_recommend_service(service_port, consul_enable):
    recommend_base_cmd="java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar /opt/recommend-service.jar  --SERVICE_PORT={} --CONSUL_ENABLE={}".format(service_port, consul_enable)
    subprocess.Popen(recommend_base_cmd, shell=True, stdout=subprocess.PIPE)

async def _start_model_serving(grpc_listen_port, init_load_path):
    if os.path.isfile(init_load_path):
        os.remove(init_load_path)
    if not os.path.exists(init_load_path):
        os.mkdir()
    serving_cmd="/opt/metaspore-serving/bin/metaspore-serving-bin -grpc_listen_port {} -init_load_path {}".format(grpc_listen_port, init_load_path)
    subprocess.Popen(serving_cmd, shell=True, stdout=subprocess.PIPE)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='recommend arguments')
    parser.add_argument('--grpc_listen_port', dest='grpc_listen_port', type=int, default=50000)
    parser.add_argument('--init_load_path', dest='init_load_path', type=str, default='/data/models')
    parser.add_argument('--service_port', dest='service_port', type=int, default=8080)
    parser.add_argument('--consul_enable', dest='consul_enable', type=str, default="false")
    args = parser.parse_args()
    print("model_serving start!")
    asyncio.run(_start_model_serving(args.grpc_listen_port, args.init_load_path))
    print("recommend service start!")
    asyncio.run(_start_recommend_service(args.service_port, args.consul_enable))
    print("model handle start!")
    _start_mms()

    # prevent docker exit
    subprocess.call(["tail", "-f", "/dev/null"])


main()
