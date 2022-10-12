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

import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIENAgent(ms.PyTorchAgent):
    def __init__(self,
                 dien_target_loss_weight=1.0,
                 dien_auxilary_loss_weight=1.0,
                 **kwargs):
        super().__init__()
        self.target_loss = nn.BCELoss()
        self.target_loss_weight = dien_target_loss_weight
        self.auxilary_loss_weight = dien_auxilary_loss_weight

    def train_minibatch(self, minibatch):
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions, auxilary_loss = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        target_loss = self.target_loss(predictions, labels)
        loss = auxilary_loss * self.auxilary_loss_weight \
               + target_loss * self.target_loss_weight
        self.trainer.train(loss)

        self.update_progress(predictions, labels)

    def validate_minibatch(self, minibatch):
        self.model.eval()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions, _ = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        self.update_progress(predictions, labels)
        return predictions.detach().reshape(-1)

