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

def normalize_storage_size(size):
    import re
    if not isinstance(size, str) or not re.match(r'\d+[MG]', size):
        message = "'size' must be a string like 4096M or 4G; "
        message += "%r is invalid" % size
        raise ValueError(message)
    value = int(size[:-1])
    unit = size[-1]
    if unit == 'G':
        value *= 1024
    return value

def merge_storage_size(worker_memory, server_memory):
    mem1 = normalize_storage_size(worker_memory)
    mem2 = normalize_storage_size(server_memory)
    mem = max(mem1, mem2)
    if mem % 1024 == 0:
        mem = '%dG' % (mem // 1024)
    else:
        mem = '%dM' % mem
    return mem
