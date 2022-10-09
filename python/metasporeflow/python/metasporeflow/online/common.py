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
import ruamel.yaml
from attr import define

S = ruamel.yaml.scalarstring.DoubleQuotedScalarString


def Object2Dict(obj):
    return {name: getattr(obj, name) for name in dir(obj)
            if not name.startswith("__") and getattr(obj, name) is not None and not callable(getattr(obj, name))}


def GetObjFields(obj):
    return [name for name in dir(obj)
            if not name.startswith("__") and getattr(obj, name) is not None and not callable(getattr(obj, name))]


class BaseConfig(object):
    def __init__(self, **kwargs):
        self.dict_data = {}
        for key, value in kwargs.items():
            self.__setattr__(key, value)
            if value is not None:
                self.dict_data[key] = value

    def to_dict(self):
        return self.dict_data if self.dict_data else Object2Dict(self)


@define
class BaseDefaultConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for item in self.__attrs_attrs__:
            if item.name not in kwargs:
                self.__setattr__(item.name, item.default)


def DumpToYaml(obj):
    if isinstance(obj, BaseConfig):
        return ruamel.yaml.round_trip_dump(obj.to_dict(), width=160)
    return ruamel.yaml.round_trip_dump(Object2Dict(obj), width=160)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def setDefault(data, key, value):
    if not isinstance(data, Dict):
        if not hasattr(data, key) or not getattr(data, key):
            setattr(data, key, value)
        return getattr(data, key)
    if not data:
        data = Dict()
    if key not in data or not data[key]:
        data[key] = value
    return data[key]


def dictToObj(obj):
    if not isinstance(obj, dict):
        return obj
    data = Dict()
    for k, v in obj.items():
        data[k] = dictToObj(v)
    return data
