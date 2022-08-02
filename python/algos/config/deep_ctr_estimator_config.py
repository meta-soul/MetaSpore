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

def array_index_validator(instance, attribute, value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("'{}' must be a non-nagative integer!".format(attribute.name))

def learning_rate_validator(instance, attribute, value):
    if not isinstance(value, float) or value <= 0:
        raise ValueError("'{}' must be a positive float!".format(attribute.name))

@attrs.frozen(kw_only=True)
class DeepCTREstimatorConfig:
    model_in_path = attrs.field(validator=attrs.validators.instance_of((type(None),str)))
    model_out_path = attrs.field(validator=attrs.validators.instance_of(str))
    model_export_path = attrs.field(default=None, validator=attrs.validators.instance_of((type(None), str)))
    model_version = attrs.field(validator=attrs.validators.instance_of(str))
    experiment_name = attrs.field(validator=attrs.validators.instance_of(str))
    input_label_column_index = attrs.field(validator=array_index_validator)
    metric_update_interval = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)])
    adam_learning_rate = attrs.field(validator=learning_rate_validator)
    training_epoches = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)])
    shuffle_training_dataset = attrs.field(validator=attrs.validators.instance_of(bool))
    
