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

import enum
import typing
import attrs

uint = typing.NewType('uint', int)
pint = typing.NewType('pint', int)
ufloat = typing.NewType('ufloat', float)
pfloat = typing.NewType('pfloat', float)

def is_primitive_type(t):
    return t in (str, bool, int, uint, pint, float, ufloat, pfloat)

def is_config_type(t):
    return attrs.has(t)

def is_enum_type(t):
    return isinstance(t, type) and issubclass(t, enum.Enum)

def is_optional_type(t):
    return (isinstance(t, typing._GenericAlias) and
            t.__origin__ is typing.Union and
            len(t.__args__) == 2 and
            is_type(t.__args__[0]) and
            t.__args__[1] is type(None))

def is_list_type(t):
    return (isinstance(t, typing._GenericAlias) and
            isinstance(t.__origin__, type) and
            issubclass(t.__origin__, list) and
            len(t.__args__) == 1 and
            is_type(t.__args__[0]))

def is_set_type(t):
    return (isinstance(t, typing._GenericAlias) and
            isinstance(t.__origin__, type) and
            issubclass(t.__origin__, set) and
            len(t.__args__) == 1 and
            is_type(t.__args__[0]))

def is_dict_type(t):
    return (isinstance(t, typing._GenericAlias) and
            isinstance(t.__origin__, type) and
            issubclass(t.__origin__, dict) and
            len(t.__args__) == 2 and
            is_type(t.__args__[0]) and
            is_type(t.__args__[1]))

def is_type(t):
    return (is_primitive_type(t) or
            is_config_type(t) or
            is_enum_type(t) or
            is_optional_type(t) or
            is_list_type(t) or
            is_set_type(t) or
            is_dict_type(t))

def format_type(t):
    if not is_type(t):
        message = "unsupported type: %r" % (t,)
        raise TypeError(message)
    if t in (str, bool, int, float):
        return t.__name__
    if t is uint:
        return 'non-negative int'
    if t is pint:
        return 'positive int'
    if t is ufloat:
        return 'non-negative float'
    if t is pfloat:
        return 'positive float'
    if is_config_type(t):
        return t.__name__
    if is_enum_type(t):
        string = t.__name__
        if all(isinstance(m.value, str) for m in t.__members__.values()):
            choices = ', '.join(m.value for m in t.__members__.values())
        else:
            choices = ', '.join(m.name for m in t.__members__.values())
        string += ' (one of: %s)' % (choices,)
        return string
    if is_optional_type(t):
        string = format_type(t.__args__[0])
        string += ' or None'
        return string
    if is_list_type(t):
        string = format_type(t.__args__[0])
        string = 'list of ' + string
        return string
    if is_set_type(t):
        string = format_type(t.__args__[0])
        string = 'set of ' + string
        return string
    if is_dict_type(t):
        string = format_type(t.__args__[0])
        string += ' to ' + format_type(t.__args__[1])
        string = 'dict of ' + string
        return string
    assert False

def type_name(t):
    if not is_type(t):
        message = "unsupported type: %r" % (t,)
        raise TypeError(message)
    if is_primitive_type(t) or is_config_type(t):
        return t.__name__
    if is_enum_type(t):
        string = 'enum[' + t.__name__
        if all(isinstance(m.value, str) for m in t.__members__.values()):
            choices = ', '.join(m.value for m in t.__members__.values())
        else:
            choices = ', '.join(m.name for m in t.__members__.values())
        string += ', [%s]]' % (choices,)
        return string
    if is_optional_type(t):
        return 'Optional[' + type_name(t.__args__[0]) + ']'
    if is_list_type(t):
        return 'List[' + type_name(t.__args__[0]) + ']'
    if is_set_type(t):
        return 'Set[' + type_name(t.__args__[0]) + ']'
    if is_dict_type(t):
        return 'Dict[' + type_name(t.__args__[0]) + ', ' + type_name(t.__args__[1]) + ']'
    assert False
