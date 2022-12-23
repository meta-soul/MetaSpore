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

import os
import yaml


class FileUtil:

    @staticmethod
    def write_dict_to_yaml_file(file_path, content):
        FileUtil.check_and_overwrite_file(file_path)
        with open(file_path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)

    @staticmethod
    def write_file(file_path, content):
        FileUtil.check_and_overwrite_file(file_path)
        with open(file_path, "w") as f:
            f.write(content)

    @staticmethod
    def check_and_overwrite_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)
        file_dir = os.path.split(file_path)[0]
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
