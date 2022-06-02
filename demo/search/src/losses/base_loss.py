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

import torch
from torch import nn, Tensor
from typing import Iterable, Dict, List, Tuple

class BaseLoss(nn.Module):

    def __init__(self, model: nn.Module, dual_model: nn.Module=None):
        super(BaseLoss, self).__init__()
        self.model = model
        self.dual_model = dual_model

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        if self.dual_model is None:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(os.path.join(save_path, 'model1'))
            self.dual_model.save_pretrained(os.path.join(save_path, 'model2'))

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor=None):
        raise NotImplementedError
