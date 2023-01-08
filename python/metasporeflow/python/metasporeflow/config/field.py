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

from .types import is_type
from .types import format_type
from .convert import convert_value
from .validate import validate_value

METASPOREFLOW_CONFIG_METADATA = '__METASPOREFLOW_CONFIG_METADATA'

def transform_field(cls, field):
    if field.metadata.get(METASPOREFLOW_CONFIG_METADATA):
        return field
    field_type = field.type
    if not is_type(field_type):
        message = "field %r of %r " % (field.name, cls.__qualname__)
        message += "has unsupported type %r" % (field_type,)
        raise TypeError(message)
    assert field.converter is None
    assert field.validator is None
    assert not field.metadata
    def validate_field(inst, attr, value):
        try:
            validate_value(value, attr.type)
        except TypeError:
            message = "field %r of %r " % (attr.name, inst.__class__.__qualname__)
            message += "must be %s; " % format_type(attr.type)
            message += "%r is invalid" % (value,)
            raise TypeError(message)
    def convert_field(value):
        return convert_value(value, field_type)
    field_default = field.default
    if field_default is not attrs.NOTHING:
        field_default = convert_field(field_default)
        try:
            validate_value(field_default, field_type)
        except TypeError:
            message = "field %r of %r " % (field.name, cls.__qualname__)
            message += "must be %s; " % format_type(field.type)
            message += "%r is invalid" % (field_default,)
            raise TypeError(message)
    field_metadata = {METASPOREFLOW_CONFIG_METADATA: True}
    return field.evolve(type=field_type,
                        default=field_default,
                        converter=convert_field,
                        validator=validate_field,
                        metadata=field_metadata)

def transform_fields(cls, fields):
    fields = tuple(transform_field(cls, field) for field in fields)
    return fields
