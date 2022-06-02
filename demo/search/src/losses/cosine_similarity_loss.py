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

# adapted from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py
import os

import torch
from torch import nn, Tensor
from typing import Iterable, Dict

from .base_loss import BaseLoss

class CosineSimilarityLoss(BaseLoss):

    def __init__(self, model: nn.Module, dual_model: nn.Module=None, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__(model, dual_model)
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        num_sentences = len(sentence_features)  # should be 2
        assert num_sentences == 2
        embeddings = []
        # encode query
        embeddings.append(self.model(**sentence_features.pop(0))['sentence_embedding'])
        # encode passages
        if self.dual_model is None:
            embeddings.append(self.model(**sentence_features.pop(0))['sentence_embedding'])
        else:
            embeddings.append(self.dual_model(**sentence_features.pop(0))['sentence_embedding'])

        score = torch.cosine_similarity(embeddings[0], embeddings[1])  # between -1~1
        output = self.cos_score_transformation(score)
        return self.loss_fct(output, labels.view(-1).float())
