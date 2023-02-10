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

from attrs import frozen
from typing import Any
from typing import Optional
from typing import List
from typing import Dict

class ModelArtsEntrypointGenerator(object):
    def __init__(self, dag_tasks):
        self._dag_tasks = dag_tasks

    def generate_entrypoint(self):
        string = '#!/bin/bash'
        string += '\n\nset -ex'
        string += '\n\ncd $(dirname ${BASH_SOURCE[0]})'
        string += '\necho "PWD: ${PWD}"'
        string += '\n\necho "MetaSpore Offline Flow begin ..."'
        string += '\n'
        for task in self._dag_tasks:
            string += '\n%s' % task.execute
        string += '\n\necho "MetaSpore Offline Flow done"'
        string += '\n'
        return string
