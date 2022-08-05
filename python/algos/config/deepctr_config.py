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

from attrs import frozen, field
from attrs.validators import optional, instance_of, gt
from .common_validators import array_index_validator, learning_rate_validator, dim_validator, hidden_units_validator, prob_like_validator

@frozen(kw_only=True)
class DeepCTREstimatorConfig:
    model_in_path = field(default=None, validator=optional(instance_of(str)))
    model_out_path = field(validator=instance_of(str))
    model_export_path = field(default=None, validator=optional(instance_of(str)))
    model_version = field(validator=instance_of(str))
    experiment_name = field(validator=instance_of(str))
    input_label_column_index = field(validator=array_index_validator)
    metric_update_interval = field(validator=[instance_of(int), gt(0)])
    adam_learning_rate = field(validator=learning_rate_validator)
    training_epoches = field(validator=[instance_of(int), gt(0)])
    shuffle_training_dataset = field(validator=instance_of(bool))
    
@frozen(kw_only=True)
class WideDeepConfig:
    use_wide = field(validator=instance_of(bool))
    wide_embedding_dim = field(validator=dim_validator)
    deep_embedding_dim = field(validator=dim_validator)
    wide_column_name_path = field(validator=instance_of(str))
    wide_combine_schema_path = field(validator=instance_of(str))
    deep_column_name_path = field(validator=instance_of(str))
    deep_combine_schema_path = field(validator=instance_of(str))
    dnn_hidden_units = field(validator=hidden_units_validator)
    dnn_hidden_activations = field(validator=instance_of(str))
    use_bias = field(validator=instance_of(bool))
    net_dropout = field(validator=prob_like_validator)
    batch_norm = field(validator=instance_of(bool))
    embedding_regularizer = field(default=None, validator=optional(instance_of(str)))
    net_regularizer = field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = field(validator=instance_of(float))
    ftrl_l2 = field(validator=instance_of(float))
    ftrl_alpha = field(validator=instance_of(float))
    ftrl_beta = field(validator=instance_of(float))
    
@frozen(kw_only=True)
class DeepFMConfig:
    use_wide = field(validator=instance_of(bool))
    use_dnn = field(validator=instance_of(bool))
    use_fm = field(validator=instance_of(bool))
    wide_embedding_dim = field(validator=dim_validator)
    deep_embedding_dim = field(validator=dim_validator)
    wide_column_name_path = field(validator=instance_of(str))
    wide_combine_schema_path = field(validator=instance_of(str))
    deep_column_name_path = field(validator=instance_of(str))
    deep_combine_schema_path = field(validator=instance_of(str))
    sparse_init_var = field(validator=instance_of(float))
    dnn_hidden_units = field(validator=hidden_units_validator)
    dnn_hidden_activations = field(validator=instance_of(str))
    use_bias = field(validator=instance_of(bool))
    net_dropout = field(validator=prob_like_validator)
    batch_norm = field(validator=instance_of(bool))
    embedding_regularizer = field(default=None, validator=optional(instance_of(str)))
    net_regularizer = field(default=None, validator=optional(instance_of(str)))
    ftrl_l1 = field(validator=instance_of(float))
    ftrl_l2 = field(validator=instance_of(float))
    ftrl_alpha = field(validator=instance_of(float))
    ftrl_beta = field(validator=instance_of(float))
    