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

def array_index_validator(instance, attribute, value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("'{}' must be a non-nagative integer!".format(attribute.name))

def learning_rate_validator(instance, attribute, value):
    if not isinstance(value, float) or value <= 0:
        raise ValueError("'{}' must be a positive float!".format(attribute.name))


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


def recommendation_count_validator(instance, attribute, value):
    if not isinstance(value, int) or value <= 0:
        raise ValueError("'{}' must be a positive integer!".format(attribute.name))
