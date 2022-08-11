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
class TwoTowerEstimatorConfig:
    model_in_path = field(default=None, validator=optional(instance_of(str)))
    model_out_path = field(validator=instance_of(str))
    model_export_path = field(default=None, validator=optional(instance_of(str)))
    model_version = field(validator=instance_of(str))
    experiment_name = field(validator=instance_of(str))
    retrieval_item_count = field(validator=instance_of(int))
    metric_update_interval = field(default=None, validator=optional(instance_of(int)))
    training_epoches = field(validator=[instance_of(int), gt(0)])
    shuffle_training_dataset = field(default=None, validator=optional(instance_of(bool)))

    item_ids_column_indices = field(validator=instance_of((list, tuple)))
    input_label_column_index = field(validator=array_index_validator)
    input_label_column_index = field(default=None, validator=optional(instance_of(int)))
    input_feature_column_num = field(default=None, validator=optional(instance_of(int)))
    input_item_id_column_index = field(default=None, validator=optional(instance_of(int)))
    input_item_probability_column_index = field(default=None, validator=optional(instance_of(int)))
    input_sample_weight_column_index = field(default=None, validator=optional(instance_of(int)))
    
    use_remove_accidental_hits = field(default=None, validator=optional(instance_of(bool)))
    use_sampling_probability_correction = field(default=None, validator=optional(instance_of(bool)))
    use_sample_weight = field(default=None, validator=optional(instance_of(bool)))
    
    milvus_description = field(validator=instance_of(str))
    milvus_host = field(validator=instance_of(str))
    milvus_port = field(validator=instance_of(str))
    milvus_embedding_field = field(validator=instance_of(str))
    milvus_index_type = field(validator=instance_of(str))
    milvus_metric_type = field(validator=instance_of(str))
    milvus_nlist = field(validator=instance_of(int))
    milvus_nprobe = field(validator=instance_of(int))

@frozen(kw_only=True)
class DSSMModelConfig:
    user_column_name = field(validator=instance_of(str))
    user_combine_schema = field(validator=instance_of(str))
    item_column_name = field(validator=instance_of(str))
    item_combine_schema = field(validator=instance_of(str))

    tau = field(validator=instance_of(float))
    sparse_init_var = field(validator=instance_of(float))
    net_dropout = field(validator=instance_of(float))
    batch_size = field(validator=[instance_of(int), gt(0)])
    vector_embedding_size = field(validator=[instance_of(int), gt(0)])
    item_embedding_size = field(validator=[instance_of(int), gt(0)])
    dnn_hidden_units = field(validator=hidden_units_validator)
    dnn_hidden_activations = field(default=None, validator=optional(instance_of(str)))
    adam_learning_rate = field(default=None, validator=learning_rate_validator)
    ftrl_l1 = field(default=None, validator=optional(learning_rate_validator))
    ftrl_l2 = field(default=None, validator=optional(learning_rate_validator))
    ftrl_alpha = field(default=None, validator=optional(learning_rate_validator))
    ftrl_beta = field(default=None, validator=optional(learning_rate_validator))

    use_bias = field(default=None, validator=optional(instance_of(bool)))
    net_dropout = field(default=None, validator=optional((prob_like_validator)))
    batch_norm = field(default=None, validator=optional(instance_of(bool)))
    embedding_regularizer = field(default=None, validator=optional(instance_of(str)))
    net_regularizer = field(default=None, validator=optional(instance_of(str)))
    

    
