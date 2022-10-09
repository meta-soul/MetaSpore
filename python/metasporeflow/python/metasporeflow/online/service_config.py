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

from .common import BaseDefaultConfig, dictToObj
from urllib.parse import quote_plus


def get_source_option(online_config, name, collection):
    options = {}
    if not name or not online_config or name not in online_config.services:
        return options
    service = online_config.services.get(name)
    service = dictToObj(service)
    if service.options is None:
        service.options = {}
    if service.kind.lower() == "mongodb":
        options["uri"] = service.options.setdefault("uri",
                                                    "mongodb://root:example@${MONGO_HOST:172.17.0.1}:${MONGO_PORT:27017}/jpa?authSource=admin")
    return options


@define
class Source(BaseDefaultConfig):
    name: str
    kind: Literal['MongoDB', 'JDBC', 'Redis', 'Request'] = field(init=False, default="Request")
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "name" not in kwargs:
            raise ValueError("source config name must not be empty!")
        self.dict_data['kind'] = 'Request'
        if self.kind.lower() == "mongodb":
            self.dict_data['kind'] = 'MongoDB'
            if not self.options.get("uri") or not str(self.options.get("uri")).startswith("mongodb://"):
                raise ValueError("source mongodb config uri error!")
        if self.kind.lower() == "jdbc":
            self.dict_data['kind'] = 'JDBC'
            if not self.options.get("uri") or not str(self.options.get("uri")).startswith("jdbc:"):
                raise ValueError("source jdbc config uri error!")
            if not self.options.get("user"):
                self.options["user"] = "root"
            if not self.options.get("password"):
                self.options["password"] = "example"
            if str(self.options.get("uri")).startswith("jdbc:mysql"):
                if not self.options.get("driver"):
                    self.options["driver"] = "com.mysql.cj.jdbc.Driver"
                if str(self.options["driver"]) != "com.mysql.cj.jdbc.Driver":
                    raise ValueError("source jdbc mysql config driver must be com.mysql.cj.jdbc.Driver!")
        if self.kind.lower() == "redis":
            self.dict_data['kind'] = 'Redis'
            if not self.options.get("standalone") and not self.options.get("sentinel") and not self.options.get(
                    "cluster"):
                self.options["standalone"] = {"host": "localhost", "port": 6379}
        if self.options:
            self.dict_data["options"] = self.options


@define
class SourceTable(BaseDefaultConfig):
    name: str
    source: str
    columns: list
    table: str = field(init=False)
    prefix: str = field(init=False, default="")
    sqlFilters: list = field(init=False, default=[])
    filters: list = field(init=False, default=[])
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "name" not in kwargs:
            raise ValueError("SourceTable config name must not be empty!")
        if "source" not in kwargs:
            raise ValueError("SourceTable config source must not be empty!")


@define
class Condition(BaseDefaultConfig):
    left: str
    right: str
    type: Literal['left', 'inner', 'right', "full"] = field(init=False, default="inner")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for item in self.__attrs_attrs__:
            if item.name not in kwargs:
                setattr(self, item.name, item.default)

    def to_dict(self):
        data = {self.left: self.right}
        if self.type != 'inner':
            data["type"] = self.type
        return data


@define
class Feature(BaseDefaultConfig):
    name: str
    depend: list
    select: list
    condition: list = field(init=False, default=[])
    immediateFrom: list = field(init=False, default=[])
    filters: list = field(init=False, default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        from_tables = self.dict_data.pop("depend")
        self.dict_data["from"] = from_tables
        if self.condition:
            self.dict_data["condition"] = [x.to_dict() for x in self.condition]
        if self.immediateFrom:
            self.dict_data["immediateFrom"] = self.immediateFrom
        if self.filters:
            self.dict_data["filters"] = self.filters
        return self.dict_data


@define
class FieldAction(BaseDefaultConfig):
    names: list
    types: list
    fields: list
    input: list
    func: str
    options: dict = field(init=False, default={})
    algoColumns: list = field(init=False, default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        data = dict()
        if len(self.names) == 1:
            data["name"] = self.names[0]
        else:
            data["names"] = self.names
        if len(self.types) == 1:
            data["type"] = self.types[0]
        else:
            data["types"] = self.types
        if self.fields:
            data["fields"] = self.fields[0] if len(self.fields) == 1 else self.fields
        if self.input:
            data["input"] = self.input[0] if len(self.input) == 1 else self.input
        if self.func:
            data["func"] = self.func
        if self.algoColumns:
            data["algoColumns"] = self.algoColumns
        if self.options:
            data["options"] = self.options
        return data


@define
class AlgoTransform(BaseDefaultConfig):
    name: str
    fieldActions: list
    output: list
    taskName: str = field(init=False, default=None)
    feature: list = field(init=False, default=[])
    algoTransform: list = field(init=False, default=[])
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.fieldActions:
            self.dict_data["fieldActions"] = [x.to_dict() for x in self.fieldActions]
        if self.taskName:
            self.dict_data["taskName"] = self.taskName
        if self.feature:
            self.dict_data["feature"] = self.feature[0] if len(self.feature) == 1 else self.feature
        if self.output:
            self.dict_data["output"] = self.output
        if self.algoTransform:
            self.dict_data["algoTransform"] = self.algoTransform[0] if len(
                self.algoTransform) == 1 else self.algoTransform
        if self.options:
            self.dict_data["options"] = self.options
        return self.dict_data


@define
class TransformConfig(BaseDefaultConfig):
    name: str
    option: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.option:
            self.dict_data["option"] = self.option
        return self.dict_data


@define
class Chain(BaseDefaultConfig):
    then: list = field(init=False, default=[])
    when: list = field(init=False, default=[])
    options: dict = field(init=False, default={})
    transforms: list = field(init=False, default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.then:
            self.dict_data["then"] = self.then[0] if len(self.then) == 1 else self.then
        if self.when:
            self.dict_data["when"] = self.when[0] if len(self.when) == 1 else self.when
        if self.options:
            self.dict_data["options"] = self.options
        if self.transforms:
            self.dict_data["transforms"] = [x.to_dict() for x in self.transforms]
        return self.dict_data


@define
class ExperimentItem(BaseDefaultConfig):
    name: str
    ratio: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@define
class Layer(BaseDefaultConfig):
    name: str
    bucketizer: str
    taskName: str = field(init=False, default=None)
    experiments: list = field(init=False, default=[])
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.taskName:
            self.dict_data["taskName"] = self.taskName
        if self.options:
            self.dict_data["options"] = self.options
        if self.experiments:
            self.dict_data["experiments"] = [x.to_dict() for x in self.experiments]
        return self.dict_data


@define
class Experiment(BaseDefaultConfig):
    name: str
    taskName: str = field(init=False, default=None)
    chains: list = field(init=False, default=[])
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.taskName:
            self.dict_data["taskName"] = self.taskName
        if self.options:
            self.dict_data["options"] = self.options
        if self.chains:
            self.dict_data["chains"] = [x.to_dict() for x in self.chains]
        return self.dict_data


@define
class Scene(BaseDefaultConfig):
    name: str
    taskName: str = field(init=False, default=None)
    chains: list = field(init=False, default=[])
    columns: list = field(init=False, default=[])
    options: dict = field(init=False, default={})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.taskName:
            self.dict_data["taskName"] = self.taskName
        if self.columns:
            self.dict_data["columns"] = self.columns
        if self.options:
            self.dict_data["options"] = self.options
        if self.chains:
            self.dict_data["chains"] = [x.to_dict() for x in self.chains]
        return self.dict_data


@define
class Service(BaseDefaultConfig):
    name: str
    taskName: str = field(init=False, default=None)
    tasks: list = field(init=False, default=[])
    options: dict = field(init=False, default={})
    preTransforms: list = field(init=False, default=[])
    transforms: list = field(init=False, default=[])
    columns: list = field(init=False, default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        if self.taskName:
            self.dict_data["taskName"] = self.taskName
        if self.tasks:
            self.dict_data["tasks"] = self.tasks
        if self.columns:
            self.dict_data["columns"] = self.columns
        if self.options:
            self.dict_data["options"] = self.options
        if self.preTransforms:
            self.dict_data["preTransforms"] = [x.to_dict() for x in self.preTransforms]
        if self.transforms:
            self.dict_data["transforms"] = [x.to_dict() for x in self.transforms]
        return self.dict_data


@define
class RecommendConfig(BaseDefaultConfig):
    layers: list = field(default=[])
    experiments: list = field(default=[])
    scenes: list = field(default=[])
    services: list = field(default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_service(self, **kwargs):
        self.services.append(Service(**kwargs))

    def add_experiment(self, **kwargs):
        self.experiments.append(Experiment(**kwargs))

    def add_layer(self, **kwargs):
        self.layers.append(Layer(**kwargs))

    def add_scene(self, **kwargs):
        self.scenes.append(Scene(**kwargs))

    def to_dict(self):
        if self.layers:
            self.dict_data["layers"] = [x.to_dict() for x in self.layers]
        if self.experiments:
            self.dict_data["experiments"] = [x.to_dict() for x in self.experiments]
        if self.scenes:
            self.dict_data["scenes"] = [x.to_dict() for x in self.scenes]
        if self.services:
            self.dict_data["services"] = [x.to_dict() for x in self.services]
        return self.dict_data


@define
class FeatureConfig(BaseDefaultConfig):
    source: list = field(default=[])
    sourceTable: list = field(default=[])
    feature: list = field(default=[])
    algoTransform: list = field(default=[])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_source(self, **kwargs):
        self.source.append(Source(**kwargs))

    def find_source(self, name):
        if self.source:
            for item in self.source:
                if item.name == name:
                    return True
        return False

    def add_sourceTable(self, **kwargs):
        self.sourceTable.append(SourceTable(**kwargs))

    def add_feature(self, **kwargs):
        self.feature.append(Feature(**kwargs))

    def add_algoTransform(self, **kwargs):
        self.algoTransform.append(AlgoTransform(**kwargs))

    def to_dict(self):
        if self.source:
            self.dict_data["source"] = [x.to_dict() for x in self.source]
        if self.sourceTable:
            self.dict_data["sourceTable"] = [x.to_dict() for x in self.sourceTable]
        if self.feature:
            self.dict_data["feature"] = [x.to_dict() for x in self.feature]
        if self.algoTransform:
            self.dict_data["algoTransform"] = [x.to_dict() for x in self.algoTransform]
        return self.dict_data


@define
class OnlineServiceConfig(BaseDefaultConfig):
    feature_service: FeatureConfig
    recommend_service: RecommendConfig

    def to_dict(self):
        return {"feature-service": self.feature_service.to_dict(),
                "recommend-service": self.recommend_service.to_dict()}

