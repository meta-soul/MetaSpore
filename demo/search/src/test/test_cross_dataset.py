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
import numpy as np

from data import create_cross_encoder_dataloader
from modeling import TransformerDualEncoder

if __name__ == '__main__':
    data_file = sys.argv[1]

    device = 'cuda:0'
    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)

    dataloader = create_cross_encoder_dataloader(data_file, 
        model.tokenize, text_indices=[0], device=device, batch_size=16, shuffle=False)

    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(**features)
            embs = outputs['sentence_embedding']
            embs = embs.cpu().numpy() if embs.is_cuda else embs.numpy()
            print(embs)
            break

