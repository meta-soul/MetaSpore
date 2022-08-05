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

from attrs.validators import optional, instance_of, gt
from .common_validators import array_index_validator, learning_rate_validator, dim_validator, hidden_units_validator, prob_like_validator

@attrs.frozen(kw_only=True)
class DeepCTREstimatorConfig:
    model_in_path = attrs.field(default=None, validator=optional(instance_of(str)))
    model_out_path = attrs.field(validator=instance_of(str))
    model_export_path = attrs.field(default=None, validator=optional(instance_of(str)))
    model_version = attrs.field(validator=instance_of(str))
    experiment_name = attrs.field(validator=instance_of(str))
    input_label_column_index = attrs.field(validator=array_index_validator)
    metric_update_interval = attrs.field(validator=[instance_of(int), gt(0)])
    adam_learning_rate = attrs.field(validator=learning_rate_validator)
    training_epoches = attrs.field(validator=[instance_of(int), gt(0)])
    shuffle_training_dataset = attrs.field(validator=instance_of(bool))
    
@attrs.frozen(kw_only=True)
class WideDeepConfig:
    use_wide = attrs.field(validator=instance_of(bool))
    wide_embedding_dim = attrs.field(validator=dim_validator)
    deep_embedding_dim = attrs.field(validator=dim_validator)
    wide_column_name_path = attrs.field(validator=instance_of(str))
    wide_combine_schema_path = attrs.field(validator=instance_of(str))
    deep_column_name_path = attrs.field(validator=instance_of(str))
    deep_combine_schema_path = attrs.field(validator=instance_of(str))
    dnn_hidden_units = attrs.field(validator=hidden_units_validator)
    dnn_hidden_activations = attrs.field(validator=instance_of(str))
    use_bias = attrs.field(validator=instance_of(bool))
    net_dropout = attrs.field(validator=prob_like_validator)
    batch_norm = attrs.field(validator=instance_of(bool))
    embedding_regularizer = attrs.field(default=None, validator=optional(instance_of(str)))
    net_regularizer = attrs.field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = attrs.field(validator=instance_of(float))
    ftrl_l2 = attrs.field(validator=instance_of(float))
    ftrl_alpha = attrs.field(validator=instance_of(float))
    ftrl_beta = attrs.field(validator=instance_of(float))
    
@attrs.frozen(kw_only=True)
class DeepFMConfig:
    use_wide = attrs.field(validator=instance_of(bool))
    use_dnn = attrs.field(validator=instance_of(bool))
    use_fm = attrs.field(validator=instance_of(bool))
    wide_embedding_dim = attrs.field(validator=dim_validator)
    deep_embedding_dim = attrs.field(validator=dim_validator)
    wide_column_name_path = attrs.field(validator=instance_of(str))
    wide_combine_schema_path = attrs.field(validator=instance_of(str))
    deep_column_name_path = attrs.field(validator=instance_of(str))
    deep_combine_schema_path = attrs.field(validator=instance_of(str))
    sparse_init_var = attrs.field(validator=instance_of(float))
    dnn_hidden_units = attrs.field(validator=hidden_units_validator)
    dnn_hidden_activations = attrs.field(validator=instance_of(str))
    use_bias = attrs.field(validator=instance_of(bool))
    net_dropout = attrs.field(validator=prob_like_validator)
    batch_norm = attrs.field(validator=instance_of(bool))
    embedding_regularizer = attrs.field(default=None, validator=optional(instance_of(str)))
    net_regularizer = attrs.field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = attrs.field(validator=instance_of(float))
    ftrl_l2 = attrs.field(validator=instance_of(float))
    ftrl_alpha = attrs.field(validator=instance_of(float))
    ftrl_beta = attrs.field(validator=instance_of(float))
    