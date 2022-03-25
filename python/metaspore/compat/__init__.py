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

from . import ps

import sys
sys.modules['ps.initializer'] = ps.initializer
sys.modules['ps.updater'] = ps.updater

def fixup_attributes(obj):
    names = dir(obj)
    for name in names:
        if not name.startswith('_'):
            continue
        if name.endswith('__'):
            continue
        i = name.find('__')
        if i == -1:
            continue
        new_name = name[i + 1:]
        value = getattr(obj, name)
        setattr(obj, new_name, value)
        delattr(obj, name)

ps.Agent._criterion = ps.Agent._metric
ps.Agent.update_criterion = ps.Agent.update_metric
ps.Agent.push_criterion = ps.Agent.push_metric
ps.Agent.clear_criterion = ps.Agent.clear_metric
