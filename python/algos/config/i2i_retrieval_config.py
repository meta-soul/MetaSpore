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
from attrs.validators import optional, instance_of
from .common_validators import recommendation_count_validator

@frozen(kw_only=True)
class SwingEstimatorConfig:
    user_id_column_name = field(default=None, validator=optional(instance_of(str)))
    item_id_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_filter_value = field(default=None, validator=optional(instance_of(str)))
    use_plain_weight = field(default=None, validator=optional(instance_of(bool)))
    smoothing_coefficient = field(default=None, validator=optional(instance_of(float)))
    max_recommendation_count = field(default=None, validator=optional(recommendation_count_validator))
    key_column_name = field(default=None, validator=optional(instance_of(str)))
    value_column_name = field(default=None, validator=optional(instance_of(str)))
    item_score_delimiter = field(default=None, validator=optional(instance_of(str)))
    item_score_pair_delimiter = field(default=None, validator=optional(instance_of(str)))
    cassandra_catalog = field(default=None, validator=optional(instance_of(str)))
    cassandra_host_ip = field(default=None, validator=optional(instance_of(str)))
    cassandra_port = field(default=None, validator=optional(instance_of(int)))
    cassandra_user_name = field(default=None, validator=optional(instance_of(str)))
    cassandra_password = field(default=None, validator=optional(instance_of(str)))
    cassandra_db_name = field(default=None, validator=optional(instance_of(str)))
    cassandra_db_properties = field(default=None, validator=optional(instance_of(str)))
    cassandra_table_name = field(default=None, validator=optional(instance_of(str)))
    
    
@frozen(kw_only=True)
class ItemCFEstimatorConfig:
    user_id_column_name = field(default=None, validator=optional(instance_of(str)))
    item_id_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_column_name = field(default=None, validator=optional(instance_of(str)))
    behavior_filter_value = field(default=None, validator=optional(instance_of(str)))
    max_recommendation_count = field(default=None, validator=optional(recommendation_count_validator))
    key_column_name = field(default=None, validator=optional(instance_of(str)))
    value_column_name = field(default=None, validator=optional(instance_of(str)))
    item_score_delimiter = field(default=None, validator=optional(instance_of(str)))
    item_score_pair_delimiter = field(default=None, validator=optional(instance_of(str)))
    debug = field(default=None, validator=optional(instance_of(bool)))
