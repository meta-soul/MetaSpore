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

import sys

res = {}
for line in sys.stdin:
    line = line.strip('\r\n')
    if not line:
        continue
    model, exp, score = line.split('\t')
    if model not in res:
        res[model] = []
    res[model].append((exp, score))

cols = [e for e,s in list(res.items())[0][1]]
print('', *cols, sep='\t')
for model, exp_list in res.items():
    exp_list = sorted(exp_list, key=lambda x:cols.index(x[0]))
    print(model, *[x[1] for x in exp_list], sep='\t')
