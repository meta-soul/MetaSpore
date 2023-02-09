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

from metasporeflow.offline.task.task import Task


class OfflinePythonTask(Task):

    def __init__(self,
                 name,
                 type,
                 data
                 ):
        super().__init__(name,
                         type,
                         data
                         )
        self._script_path = data.scriptPath
        self._config_path = data.configPath

    def _execute(self):
        return "python -u %s --conf %s" % (self._script_path, self._config_path)
