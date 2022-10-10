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

from metasporeflow.online.check_service import notifyRecommendService
from metasporeflow.online.cloud_consul import putServiceConfig
from metasporeflow.online.online_flow import OnlineFlow
from metasporeflow.online.online_generator import OnlineGenerator, get_demo_jpa_flow
from metasporeflow.online.common import DumpToYaml


def run_cmd(command):
    ret = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
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
        self._local_resource = resources.find_by_name("demo_metaspore_flow")
        self._online_resource = resources.find_by_name("online_local_flow")
        self._generator = OnlineGenerator(resource=self._online_resource, local_resource=self._local_resource)
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
            putServiceConfig(online_recommend_config, "localhost", consul_port)
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
        pass

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
    online = get_demo_jpa_flow()
    executor = OnlineLocalExecutor(online)
    executor.execute_up()
