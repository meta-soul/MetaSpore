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

import collections.abc

from .types import uint
from .types import pint
from .types import ufloat
from .types import pfloat

from .types import is_config_type
from .types import is_enum_type
from .types import is_optional_type
from .types import is_union_type
from .types import is_list_type
from .types import is_set_type
from .types import is_dict_type

from .types import format_type

def validate_value(v, t):
    if t in (str, bool, int, float):
        if isinstance(v, t):
            return
    elif t is uint:
        if isinstance(v, int) and v >= 0:
            return
    elif t is pint:
        if isinstance(v, int) and v > 0:
            return
    elif t is ufloat:
        if isinstance(v, (int, float)) and v >= 0:
            return
    elif t is pfloat:
        if isinstance(v, (int, float)) and v > 0:
            return
    elif is_config_type(t) or is_enum_type(t):
        if isinstance(v, t):
            return
    elif is_optional_type(t):
        if v is None:
            return
        validate_value(v, t.__args__[0])
        return
    elif is_union_type(t):
        for a in t.__args__:
            if isinstance(v, a):
                return
    elif is_list_type(t):
        if isinstance(v, collections.abc.Sequence):
            for item in v:
                validate_value(item, t.__args__[0])
            return
    elif is_set_type(t):
        if isinstance(v, collections.abc.Set):
            for item in v:
                validate_value(item, t.__args__[0])
            return
    elif is_dict_type(t):
        if isinstance(v, collections.abc.Mapping):
            for key, value in v.items():
                validate_value(key, t.__args__[0])
                validate_value(value, t.__args__[1])
            return
    else:
        assert False
    message = "value must be %s; " % format_type(t)
    message += "%r is invalid" % (v,)
    raise TypeError(message)
