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
import cattrs

from .types import uint
from .types import pint
from .types import ufloat
from .types import pfloat
from .immutable import make_immutable_value

class Converter(cattrs.GenConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_structure_hook(uint, lambda x, _: int(x))
        self.register_structure_hook(pint, lambda x, _: int(x))
        self.register_structure_hook(ufloat, lambda x, _: float(x))
        self.register_structure_hook(pfloat, lambda x, _: float(x))
        self.register_structure_hook(enum.Enum, self._structure_enum)

    @staticmethod
    def _structure_enum(val, type):
        if isinstance(val, str) and val in type.__members__:
            return type.__members__[val]
        return super()._structure_enum_literal(val, type)

    def _unstructure_enum(self, obj):
        if isinstance(obj.value, str):
            return obj.value
        return obj.name

    def structure(self, obj, cl):
        data = super().structure(obj, cl)
        value = make_immutable_value(data)
        return value

    def _structure_list(self, obj, cl):
        data = super()._structure_list(obj, cl)
        value = make_immutable_value(data)
        return value

    def _structure_set(self, obj, cl, structure_to=frozenset):
        data = super()._structure_set(obj, cl, structure_to=structure_to)
        value = make_immutable_value(data)
        return value

    def _structure_dict(self, obj, cl):
        data = super()._structure_dict(obj, cl)
        value = make_immutable_value(data)
        return value
