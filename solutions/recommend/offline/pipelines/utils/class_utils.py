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

import importlib

def get_class(class_dict):
    clazz = getattr(importlib.import_module(class_dict['module_name']), class_dict['class_name'])
    print('Debug - clazz: ', clazz)
    return clazz

def get_class(module_name, class_name):
    clazz = getattr(importlib.import_module(module_name), class_name)
    print('Debug - clazz: ', clazz)
    return clazz