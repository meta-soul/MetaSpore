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
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

def load_emb(emb_file):
    embs = []
    with open(emb_file, 'r') as fin:
        for line in fin:
            line = line.strip('\r\n').strip()
            if not line:
                continue
            values = [float(v) for v in line.split(' ')]
            embs.append(values)
    return embs

emb1_file, emb2_file = sys.argv[1], sys.argv[2]
embs1 = load_emb(emb1_file)
embs2 = load_emb(emb2_file)
scores = cosine_similarity(embs1, embs2)
for i in range(len(embs1)):
    print(scores[i][i])
