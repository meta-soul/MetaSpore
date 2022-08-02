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

def dim_validator(instance, attribute, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError("'{}' must be a positive integer!".format(attribute.name))

def hidden_units_validator(instance, attribute, value):
    if not isinstance(value, list):
        raise ValueError("'{}' must be a list!".format(attribute.name))
    for e in value:
        if not isinstance(e, int) or e <= 0:
            raise ValueError("'{}' must be a list of positive integers!".format(attribute.name))

def prob_like_validator(instance, attribute, value):
    if value < 0 or value > 1:
        raise ValueError("'{}' must be a number which is in interval [0, 1] !".format(attribute.name))


@attrs.frozen(kw_only=True)
class DeepFMConfig:
    use_wide = attrs.field(validator=attrs.validators.instance_of(bool))
    use_dnn = attrs.field(validator=attrs.validators.instance_of(bool))
    use_fm = attrs.field(validator=attrs.validators.instance_of(bool))
    wide_embedding_dim = attrs.field(validator=dim_validator)
    deep_embedding_dim = attrs.field(validator=dim_validator)
    wide_column_name_path = attrs.field(validator=attrs.validators.instance_of(str))
    wide_combine_schema_path = attrs.field(validator=attrs.validators.instance_of(str))
    deep_column_name_path = attrs.field(validator=attrs.validators.instance_of(str))
    deep_combine_schema_path = attrs.field(validator=attrs.validators.instance_of(str))
    sparse_init_var = attrs.field(validator=attrs.validators.instance_of(float))
    dnn_hidden_units = attrs.field(validator=hidden_units_validator)
    dnn_hidden_activations = attrs.field(validator=attrs.validators.instance_of(str))
    use_bias = attrs.field(validator=attrs.validators.instance_of(bool))
    net_dropout = attrs.field(validator=prob_like_validator)
    batch_norm = attrs.field(validator=attrs.validators.instance_of(bool))
    embedding_regularizer = attrs.field(default=None, validator=attrs.validators.instance_of((type(None), str)))
    net_regularizer = attrs.field(default=None, validator=attrs.validators.instance_of((type(None), str)))
    ftrl_l1 = attrs.field(validator=attrs.validators.instance_of(float))
    ftrl_l2 = attrs.field(validator=attrs.validators.instance_of(float))
    ftrl_alpha = attrs.field(validator=attrs.validators.instance_of(float))
    ftrl_beta = attrs.field(validator=attrs.validators.instance_of(float))
    