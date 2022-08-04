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

def get_class(full_class_name):
    def parse_class_desc(class_desc, path_sep='.'):
        if not class_desc or len(class_desc)<=0:
            print('Error -- empty class desc:', class_desc)
            return None
        class_path_list = class_desc.split(path_sep)
        if len(class_path_list) <= 1:
            print('Error -- empty module name:', class_desc)
        return path_sep.join(class_path_list[:-1]), class_path_list[-1]

    if not full_class_name or len(full_class_name) == 0:
        raise TypeError("Class name is None, current class is {}".format(type(full_class_name)))
    module_name, class_name = parse_class_desc(full_class_name)
    clazz = getattr(importlib.import_module(module_name), class_name)
    print('Debug -- get class:', clazz)
    return clazz
