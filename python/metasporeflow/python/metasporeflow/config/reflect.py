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

import attrs
import json
import types

from typing import Optional
from typing import List

from .decorator import config
from .structure import unstructure
from .types import is_config_type
from .types import type_name

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, types.MappingProxyType):
            return dict(obj)
        return super().default(self, obj)

def dump_field_default(v):
    value = unstructure(v)
    data = json.dumps(value, separators=(',', ':'), cls=Encoder)
    return data

@config
class ConfigField:
    name: str
    type: str
    default: Optional[str] = None

@config
class ConfigClass:
    name: str
    fields: List[ConfigField]

def reflect(cls):
    if not is_config_type(cls):
        message = "invalid config class: %r" % (cls,)
        raise TypeError(message)
    name = cls.__name__
    fields = []
    for attr in cls.__attrs_attrs__:
        field_name = attr.name
        field_type = type_name(attr.type)
        field_default = None
        if attr.default is not attrs.NOTHING:
            field_default = dump_field_default(attr.default)
        field = ConfigField(name=field_name, type=field_type, default=field_default)
        fields.append(field)
    cfg_cls = ConfigClass(name=name, fields=fields)
    return cfg_cls
