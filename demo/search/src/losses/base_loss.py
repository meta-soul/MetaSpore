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
