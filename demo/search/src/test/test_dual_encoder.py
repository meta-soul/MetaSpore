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
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from modeling import TransformerDualEncoder

if __name__ == '__main__':
    text_pair_file = sys.argv[1]
    device = 'cuda:0'

    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)
    print('norm', model.normalize_output)
    
    texts_a, texts_b = [], []
    with open(text_pair_file, 'r', encoding='utf8') as fin:
        for line in fin:
            text_a, text_b = line.strip().split('\t')[:2]
            texts_a.append(text_a)
            texts_b.append(text_b)

    with torch.no_grad():
        embs_a = model.encode(texts_a, device=device)['sentence_embedding']
        embs_b = model.encode(texts_b, device=device)['sentence_embedding']
        print(embs_a.cpu().numpy())
        print(embs_b.cpu().numpy())
        simi = torch.mm(embs_a, embs_b.transpose(0, 1))
        print(simi)
