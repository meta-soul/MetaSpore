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
from attrs import define
from attrs import field
from typing import Literal

from .common import DumpToYaml, S, BaseDefaultConfig


@define
class DockerBuildInfo(BaseDefaultConfig):
    context: str = field(init=False, default=".")
    dockerfile: str = field(init=False, default="Dockerfile")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@define
class OnlineService(BaseDefaultConfig):
    container_name: str
    image: str
    command: list = field()
    environment: dict = field(init=False, default={})
    ports: list = field(init=False, default=[])
    depends_on: list = field(init=False, default=[])
    volumes: list = field(init=False, default=[])
    healthcheck: dict = field(init=False, default={})
    restart: Literal['on-failure', 'always'] = field(init=False, default="on-failure")
    build: DockerBuildInfo = field(init=False, default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "container_name" not in kwargs:
            raise ValueError("no container_name configure")
        if "restart" not in kwargs:
            self.dict_data["restart"] = self.restart
        if "image" not in kwargs:
            self.build = DockerBuildInfo()
            self.dict_data["build"] = self.build.to_dict()
        if "ports" in kwargs:
            self.dict_data["ports"] = [S("%d:%d" % (port, port)) for port in self.ports]

    def add_env(self, key, value):
        self.environment[key] = value
        if "environment" not in self.dict_data:
            self.dict_data["environment"] = self.environment


@define
class OnlineDockerCompose(BaseDefaultConfig):
    version: Literal['3.5'] = field(init=False, default='3.5')
    services: dict = field(init=False, default={})
    networks: dict = field(init=False, default={"default": {"name": "recommend"}})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dict_data["version"] = self.version
        #self.dict_data["networks"] = self.networks

    def to_dict(self):
        if self.services:
            self.dict_data["services"] = {key: value.to_dict() for key, value in self.services.items()}
        return self.dict_data

    def add_service(self, name, container_name, **kwargs):
        if not name:
            return
        service_kwargs = dict()
        service_kwargs.update(kwargs)
        if name == "recommend":
            service_kwargs["ports"] = kwargs.setdefault("ports", [13013])
            service_kwargs["image"] = kwargs.setdefault("image", "swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/recommend-service-11:1.0.14")
            service_kwargs["command"] = kwargs.setdefault("command", "java -Xmx2048M -Xms2048M -Xmn768M -XX:MaxMetaspaceSize=256M -XX:MetaspaceSize=256M -jar recommend-service-1.0-SNAPSHOT.jar")
            service_kwargs["depends_on"] = kwargs.setdefault("depends_on", ["consul"])
        if name == "consul":
            service_kwargs["ports"] = kwargs.setdefault("ports", [8500, 8600, 8300])
            service_kwargs["environment"] = {'CONSUL_LOCAL_CONFIG': r"{\"skip_leave_on_interrupt\": true}"}
            service_kwargs["environment"].update(kwargs.setdefault("environment", {}))
            service_kwargs["image"] = kwargs.setdefault("image", "consul:1.13.1@sha256:4f54d5ddb23771cf79d9ad543d1e258b7da802198bc5dbc3ff85992cc091a50e")
            service_kwargs["command"] = kwargs.setdefault("command",
                                                          "consul agent -server -bootstrap-expect 1 -data-dir=/consul/data -bind=127.0.0.1 -client=0.0.0.0 -ui")
        if str(name).startswith("model"):
            service_kwargs["ports"] = kwargs.setdefault("ports", [50000])
            service_kwargs["image"] = kwargs.setdefault("image", "swr.cn-southwest-2.myhuaweicloud.com/dmetasoul-repo/metaspore-serving-release:cpu-v1.0.1@sha256:99b62896bf2904b1e2814eb247e1e644f83b9c90128454d96261088bb24ec80a")
            service_kwargs["command"] = kwargs.setdefault("command", "/opt/metaspore-serving/bin/metaspore-serving-bin -grpc_listen_port 50000 -init_load_path /data/models")
            service_kwargs["volumes"] = kwargs.setdefault("volumes", ["${DOCKER_VOLUME_DIRECTORY:-.}/volumes/serving_models:/data/models"])
        if str(name).startswith("mongo"):
            service_kwargs["ports"] = kwargs.setdefault("ports", [27017])
            service_kwargs["image"] = kwargs.setdefault("image", "mongo:6.0.1")
            service_kwargs["restart"] = kwargs.setdefault("restart", "always")
        if str(name).startswith("redis"):
            service_kwargs["ports"] = kwargs.setdefault("ports", [6379])
            service_kwargs["image"] = kwargs.setdefault("image", "redis:7.0.4")
            service_kwargs["restart"] = kwargs.setdefault("restart", "always")
        if str(name).startswith("mysql"):
            service_kwargs["ports"] = kwargs.setdefault("ports", [3306])
            service_kwargs["image"] = kwargs.setdefault("image", "mysql:8.0.30")
            service_kwargs["restart"] = kwargs.setdefault("restart", "always")
        if name == "etcd":
            service_kwargs["volumes"] = kwargs.setdefault("volumes", ["${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd"])
            service_kwargs["environment"] = {'ETCD_AUTO_COMPACTION_MODE': "revision", "ETCD_AUTO_COMPACTION_RETENTION": 1000,
                               "ETCD_QUOTA_BACKEND_BYTES": 4294967296}
            service_kwargs["environment"].update(kwargs.setdefault("environment", {}))
            service_kwargs["image"] = kwargs.setdefault("image", "quay.io/coreos/etcd:v3.5.0")
            service_kwargs["command"] = kwargs.setdefault("command",
                                                          "etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd")
        if name == "minio":
            service_kwargs["volumes"] = kwargs.setdefault("volumes", ["${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data"])
            service_kwargs["environment"] = {'MINIO_ACCESS_KEY': "minioadmin", "MINIO_SECRET_KEY": "minioadmin"}
            service_kwargs["environment"].update(kwargs.setdefault("environment", {}))
            service_kwargs["image"] = kwargs.setdefault("image", "minio/minio:RELEASE.2020-12-03T00-03-10Z")
            service_kwargs["command"] = kwargs.setdefault("command",
                                                          "minio server /minio_data")
            service_kwargs["healthcheck"] = kwargs.setdefault("healthcheck",
                               {"test": [S("CMD"), S("curl"), S("-f"), S("http://localhost:9000/minio/health/live")],
                               "interval": "30s", "timeout": "20s", "retries": 3})
        if str(name).startswith("milvus"):
            service_kwargs["ports"] = kwargs.setdefault("ports", [19530])
            service_kwargs["environment"] = {'ETCD_ENDPOINTS': "etcd:2379", "MINIO_ADDRESS": "minio:9000"}
            service_kwargs["environment"].update(kwargs.setdefault("environment", {}))
            service_kwargs["image"] = kwargs.setdefault("image", "milvusdb/milvus:v2.0.1")
            service_kwargs["command"] = kwargs.setdefault("command",
                                                          [S("milvus"), S("run"), S("standalone")])
            service_kwargs["volumes"] = kwargs.setdefault("volumes",
                                                          ["${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus"])
            service_kwargs["depends_on"] = kwargs.setdefault("depends_on", ["etcd", "minio"])
        if "depends_on" in service_kwargs:
            for depend in service_kwargs["depends_on"]:
                if depend not in self.services:
                    self.add_service(depend, "contain_%s_service" % depend)
            service_kwargs["depends_on"] = [S(x) for x in service_kwargs["depends_on"]]
        service_kwargs["container_name"] = container_name
        self.services[name] = OnlineService(**service_kwargs)


if __name__ == '__main__':
    online = OnlineDockerCompose()
    online.add_service("recommend", "recommend-service")
    online.add_service("model", "model-serving")
    online.add_service("mongo", "mongodb-service", environment={
        "MONGO_INITDB_ROOT_USERNAME": "root",
        "MONGO_INITDB_ROOT_PASSWORD": "example"
    })
    online.add_service("milvus", "milvus-service")
    print(DumpToYaml(online))
