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

import types

from .types import is_primitive_type
from .types import is_config_type
from .types import is_enum_type
from .types import is_optional_type
from .types import is_union_type
from .types import is_list_type
from .types import is_set_type
from .types import is_dict_type

def convert_value(v, t):
    if is_primitive_type(t) or is_config_type(t):
        return v
    if is_enum_type(t):
        if isinstance(v, str) and v in t.__members__:
            return t.__members__[v]
        return v
    if is_optional_type(t):
        if v is None:
            return None
        return convert_value(v, t.__args__[0])
    if is_union_type(t):
        return v
    if is_list_type(t):
        return tuple(convert_value(item, t.__args__[0]) for item in v)
    if is_set_type(t):
        return frozenset(convert_value(item, t.__args__[0]) for item in v)
    if is_dict_type(t):
        return types.MappingProxyType({
            convert_value(key, t.__args__[0]) :
            convert_value(value, t.__args__[1])
            for key, value in v.items()
        })
    message = "unexpected type: %r" % (t,)
    raise RuntimeError(message)
