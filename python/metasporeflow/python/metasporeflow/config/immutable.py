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

def is_immutable_value(v):
    import collections.abc
    if v is None:
        return True
    if isinstance(v, (str, bool, int, float)):
        return True
    if isinstance(v, (collections.abc.MutableSequence,
                      collections.abc.MutableSet,
                      collections.abc.MutableMapping)):
        return False
    if isinstance(v, (collections.abc.Sequence, collections.abc.Set)) and all(
        is_immutable_value(item) for item in v):
        return True
    if isinstance(v, collections.abc.Mapping) and all(
        is_immutable_value(key) and
        is_immutable_value(value)
        for key, value in v.items()):
        return True
    return True

def make_immutable_value(v):
    import types
    import collections.abc
    if v is None:
        return None
    if isinstance(v, (str, bool, int, float)):
        return v
    if is_immutable_value(v):
        return v
    if isinstance(v, collections.abc.Sequence):
        return tuple(make_immutable_value(item) for item in v)
    if isinstance(v, collections.abc.Set):
        return frozenset(make_immutable_value(item) for item in v)
    if isinstance(v, collections.abc.MutableMapping):
        return types.MappingProxyType({
            make_immutable_value(key) :
            make_immutable_value(value)
            for key, value in v.items()
        })
    return v
