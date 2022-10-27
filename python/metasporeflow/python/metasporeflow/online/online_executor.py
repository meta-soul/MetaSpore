#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import random
import subprocess
import time
import os

from metasporeflow.online.check_service import notifyRecommendService, healthRecommendService
from metasporeflow.online.cloud_consul import putServiceConfig, Consul
from metasporeflow.online.online_flow import OnlineFlow
from metasporeflow.online.online_generator import OnlineGenerator
from metasporeflow.online.common import DumpToYaml


def run_cmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(ret)
    return ret.returncode


def is_container_active(container_name):
    cmd = "echo $( docker container inspect -f '{{.State.Running}}' %s )" % container_name
    res = subprocess.run(cmd, shell=True, check=True,
                         capture_output=True, text=True)
    return res.stdout.strip() == "true"


def stop_local_container(container_name):
    cmd = "docker stop %s" % container_name
    subprocess.run(cmd, shell=True)


def remove_local_container(container_name):
    if is_container_active(container_name):
        stop_local_container(container_name)
    cmd = "docker rm %s" % container_name
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class OnlineLocalExecutor(object):
    def __init__(self, resources):
        self._online_resource = resources.find_by_name("online_local_flow")
        self._generator = OnlineGenerator(resource=self._online_resource)
        self._docker_compose_file = "%s/docker-compose.yml" % os.getcwd()

    def execute_up(self, **kwargs):
        compose_info = self._generator.gen_docker_compose()
        docker_compose = open(self._docker_compose_file, "w")
        docker_compose.write(DumpToYaml(compose_info))
        docker_compose.close()
        consul_container_name = compose_info.services["consul"].container_name
        consul_port = compose_info.services["consul"].ports[0]
        recommend_container_name = compose_info.services["recommend"].container_name
        recommend_port = compose_info.services["recommend"].ports[0]
        if run_cmd(["docker compose -f %s up -d" % self._docker_compose_file]) == 0:
            while not is_container_active(consul_container_name):
                print("wait consul start...")
                time.sleep(1)
            online_recommend_config = self._generator.gen_server_config()
            consul_client = Consul("localhost", consul_port)
            putServiceConfig(consul_client, online_recommend_config)
            time.sleep(3)
            while not is_container_active(recommend_container_name):
                print("wait recommend start...")
                time.sleep(1)
            notifyRecommendService("localhost", recommend_port)
        else:
            print("online flow up fail!")

    def execute_down(self, **kwargs):
        if run_cmd(["docker compose -f %s down" % self._docker_compose_file]) == 0:
            print("online flow down success!")
        else:
            print("online flow down fail!")

    def execute_status(self, **kwargs):
        compose_info = self._generator.gen_docker_compose()
        consul_container_name = compose_info.services["consul"].container_name
        recommend_container_name = compose_info.services["recommend"].container_name
        model_container_name = compose_info.services["model"].container_name
        recommend_port = compose_info.services["recommend"].ports[0]
        info = {"status": "UP"}
        if not is_container_active(consul_container_name):
            info["status"] = "DOWN"
            info["consul"] = "consul docker container is not up!"
        else:
            info["consul"] = "consul docker container:{} is up!".format(consul_container_name)
            info["consul_image"] = compose_info.services["consul"].image
            info["consul_port"] = compose_info.services["consul"].ports[0]
        if not is_container_active(recommend_container_name):
            info["status"] = "DOWN"
            info["recommend"] = "recommend docker container is not up!"
        else:
            info["recommend"] = "recommend docker container:{} is up!".format(recommend_container_name)
            info["recommend_image"] = compose_info.services["recommend"].image
            info["recommend_port"] = compose_info.services["recommend"].ports[0]
        if not is_container_active(model_container_name):
            info["status"] = "DOWN"
            info["model"] = "model docker container is not up!"
        else:
            info["model"] = "model docker container:{} is up!".format(model_container_name)
            info["model_image"] = compose_info.services["model"].image
            info["model_port"] = compose_info.services["model"].ports[0]
        if info["status"] != 'UP':
            return info
        info["service_status"] = healthRecommendService("localhost", recommend_port)
        info["status"] = info["service_status"].setdefault("status", "DOWN")
        return info

    @staticmethod
    def execute_update(resource):
        generator = OnlineGenerator(resource=resource)
        compose_info = generator.gen_docker_compose()
        consul_container_name = compose_info.services["consul"].container_name
        if not is_container_active(consul_container_name):
            return False, "consul docker is not up!"
        try:
            online_recommend_config = generator.gen_server_config()
        except Exception as ex:
            return False, "recommend service config generate fail ex:{}!".format(ex.args)
        consul_port = compose_info.services["consul"].ports[0]
        consul_client = Consul("localhost", consul_port)
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
            self.execute_down(**kwargs)
            self.execute_up(**kwargs)
        print("online flow reload success!")


if __name__ == "__main__":
    from metasporeflow.flows.flow_loader import FlowLoader
    from metasporeflow.online.online_flow import OnlineFlow

    flow_loader = FlowLoader()
    flow_loader._file_name = 'test/metaspore-flow.yml'
    resources = flow_loader.load()

    online_flow = resources.find_by_type(OnlineFlow)
    print(type(online_flow))
    print(online_flow)

    flow_executor = OnlineLocalExecutor(resources)
    flow_executor.execute_up()
