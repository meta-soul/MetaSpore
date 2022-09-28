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

import torch
import metaspore as ms
import torch.nn.functional as F
import json
from datetime import datetime

from metaspore._metaspore import Message
from ..retrieval_metric import RetrievalModelMetric

class SimpleXAgent(ms.PyTorchAgent):
    def _create_metric(self):
        metric = RetrievalModelMetric(use_auc=False)
        return metric

    def update_metric(self, predictions, labels, loss):
        self._metric.accumulate(predictions.data.numpy(), labels.data.numpy(), loss.data.numpy())

    def handle_request(self, req):
        body = json.loads(req.body)
        command = body.get('command')
        if command == 'PushMetric':
            states = ()
            for i in range(req.slice_count):
                states += req.get_slice(i),
            accum = self._metric
            delta = RetrievalModelMetric.from_states(states)
            accum.merge(delta)
            string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            string += f' -- auc: {accum.compute_auc()}'
            string += f', \u0394auc: {delta.compute_auc()}'
            string += f', loss: {accum.compute_loss()}'
            string += f', \u0394loss: {delta.compute_loss()}'
            string += f', pcoc: {accum.compute_pcoc()}'
            string += f', \u0394pcoc: {delta.compute_pcoc()}'
            string += f', #instance: {accum.instance_count}'
            if accum.threshold > 0.0:
                string += f', accuracy: {accum.compute_accuracy()}'
                string += f', precision: {accum.compute_precision()}'
                string += f', recall: {accum.compute_recall()}'
                string += f', F{accum.beta:g}_score: {accum.compute_f_score()}'
            print(string)
            res = Message()
            self.send_response(req, res)
            return
        super().handle_request(req)

    @staticmethod
    def nansum(x):
        return torch.where(torch.isnan(x), torch.zeros_like(x), x).sum()

    @staticmethod
    def cosine_contrastive_loss(yhat, y, nsc, w, m):
        z = yhat-m
        z[z<0] = 0
        loss_vector = y*(1-yhat) + (1-y)*(w/nsc*z)

        return torch.sum(loss_vector)

    def compute_loss(self, predictions, labels):
        loss = self.cosine_contrastive_loss(predictions, labels, self._negative_sample_count, self._w, self._m) / labels.shape[0] * (1+self._negative_sample_count)
        return loss

    def update_metric(self, predictions, labels, loss):
        self._metric.accumulate(predictions.data.numpy(), labels.data.numpy(), loss.data.numpy())

    def update_progress(self, predictions, labels, loss):
        self.minibatch_id += 1
        self.update_metric(predictions, labels, loss)
        if self.minibatch_id % self.metric_update_interval == 0:
            self.push_metric()

    def train_minibatch(self, minibatch):
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions = self.model(ndarrays)
        labels = torch.from_numpy(labels).reshape(-1, 1)
        loss = self.compute_loss(predictions, labels)
        self.trainer.train(loss)
        self.update_progress(predictions, labels, loss)
