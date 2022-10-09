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
from typing import Optional

from attrs import frozen


@frozen
class DockerInfo(object):
    image: Optional[str] = None
    environment: Optional[dict] = dict()


@frozen
class ServiceInfo(object):
    host: Optional[str] = "172.17.0.1"
    port: Optional[int] = 27017
    kind: Optional[str] = "mongodb"
    collection: Optional[list] = list()
    options: Optional[dict] = dict()


@frozen
class DataSource(object):
    table: str
    serviceName: str
    collection: str
    columns: Optional[list] = None


@frozen
class MilvusInfo(object):
    collection: str
    fields: list
    serviceName: str


@frozen
class RandomModelInfo(object):
    name: str
    bound: int
    source: DataSource


@frozen
class CFModelInfo(object):
    name: str
    source: DataSource


@frozen
class TwoTowerModelInfo(object):
    name: str
    model: str
    milvus: MilvusInfo


@frozen
class CrossFeature(object):
    name: str
    join: str
    fields: list


@frozen
class RankModelInfo(object):
    name: str
    model: str
    column_info: dict
    cross_features: list


@frozen
class FeatureInfo(object):
    user: DataSource
    item: DataSource
    summary: DataSource
    request: list
    user_key_name: str
    item_key_name: str
    user_item_ids_name: str
    user_item_ids_split: str


@frozen
class OnlineFlow(object):
    source: Optional[FeatureInfo] = None
    random_models: Optional[list] = None
    cf_models: Optional[list] = None
    twotower_models: Optional[list] = None
    rank_models: Optional[list] = None
    services: Optional[dict] = None
    dockers: Optional[dict] = None




