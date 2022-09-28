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
import torch.nn.functional as F
import numpy as np

class GRU4RecBatchNegativeSamplingModule(ms.TwoTowerRetrievalModule):
    def __init__(self, user_module, item_module, similarity_module):
        super().__init__(user_module, user_module, similarity_module)
        if not isinstance(user_module, torch.nn.Module):
            raise TypeError(f"user_module must be torch.nn.Module; {user_module!r} is invalid")
        if not isinstance(item_module, torch.nn.Module):
            raise TypeError(f"item_module must be torch.nn.Module; {item_module!r} is invalid")
        if not isinstance(similarity_module, torch.nn.Module):
            raise TypeError(f"similarity_module must be torch.nn.Module; {similarity_module!r} is invalid")
        self._user_module = user_module
        self._item_module = item_module
        self._similarity_module = similarity_module

    @property
    def user_module(self):
        return self._user_module

    @property
    def item_module(self):
        return self._item_module

    @property
    def similarity_module(self):
        return self._similarity_module

    def forward(self, x):
        user_emb = self._user_module(x)
        item_emb = self._item_module(x)
        scores = torch.matmul(user_emb, item_emb.T)
        targets = torch.tensor(range(len(scores)), dtype=torch.long)
        predictions = F.softmax(scores, dim=1).diag()
        return predictions, scores, targets

class GRU4RecBatchNegativeSamplingAgent(ms.PyTorchAgent):
    def __init__(self,
                 tau = 1.0,
                 loss_type='bpr', # loss_type in ['bpr', 'top1']
                ):
        super().__init__()
        self.tau = tau
        if loss_type == 'top1':
            self.loss_fct = self.top1_loss
        else:
            self.loss_fct = self.bpr_loss

    def bpr_loss(self, scores):
        pairwise_scores = torch.unsqueeze(scores.diag(), 1) - scores
        log_prob = F.logsigmoid(pairwise_scores).mean()
        return -log_prob

    def top1_loss(self, scores):
        pairwise_scores = scores - torch.unsqueeze(scores.diag(), 1)
        prob = torch.sigmoid(pairwise_scores).mean()
        reg = torch.sigmoid(scores**2-scores.diag()**2).mean()
        return prob + reg

    def preprocess_minibatch(self, minibatch):
        ndarrays = [col.values for col in minibatch]
        # exclude sampling probability and sample weight
        if self.input_feature_column_num is not None:
            ndarrays = ndarrays[:self.input_feature_column_num]
        labels = minibatch[self.input_label_column_index].values.astype(np.float32)
        return ndarrays, labels

    def train_minibatch(self, minibatch):
        # prepare the training process
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions, scores, targets = self.model(ndarrays)
        # temperature control
        if self.tau is not None and self.tau > 1e-6:
            scores = scores/self.tau
        loss = self.loss_fct(scores)
        # backward the loss
        self.trainer.train(loss)
        # print(loss)
        # update trainning progress
        labels = torch.from_numpy(labels).reshape(-1, 1)
        self.update_progress(predictions, labels)
