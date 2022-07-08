import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..retrieval_metric import RetrievalModelMetric

MIN_FLOAT = np.finfo(np.float32).min / 100.0
MAX_FLOAT = np.finfo(np.float32).max / 100.0

class TwoTowerBatchNegativeSamplingModule(ms.TwoTowerRetrievalModule):
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
        predictions = F.softmax(scores, dim=1).clone().detach().diag()
        return predictions, scores, targets

class TwoTowerBatchNegativeSamplingAgent(ms.PyTorchAgent):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_loss_no_reduction = nn.CrossEntropyLoss(reduction='none')
        
    def train_minibatch(self, minibatch):
        # prepare the training process
        self.model.train()
        ndarrays, labels = self.preprocess_minibatch(minibatch)
        predictions, scores, targets = self.model(ndarrays)
        # temperature control
        if self.tau is not None and self.tau > 1e-6:
            scores = scores/self.tau
        # remove accidental hits
        if self.use_remove_accidental_hits:
            candidate_ids = minibatch[self.input_item_id_column_index].values.astype(np.long) 
            scores = self.remove_accidental_hits(targets, candidate_ids, scores)
        # sampling probability correction
        if self.use_sampling_probability_correction:
            sampling_probability = minibatch[self.input_item_probability_column_index]
            scores = self.sampling_probability_correction(sampling_probability, scores)
        # sample weight
        if self.use_sample_weight:
            sample_weight = minibatch[self.input_sample_weight_column_index]
            loss = self.cross_entropy_loss_with_sample_weight(scores, targets, sample_weight)
        else:    
            loss = self.cross_entropy_loss(scores, targets)
        # backward the loss
        self.trainer.train(loss)
        # update trainning progress
        labels = torch.from_numpy(labels).reshape(-1, 1)
        self.update_progress(predictions, labels, loss)
    
    def preprocess_minibatch(self, minibatch):
        ndarrays = [col.values for col in minibatch]
        # exclude sampling probability and sample weight
        if self.input_feature_column_num is not None:
            ndarrays = ndarrays[:self.input_feature_column_num]
        labels = minibatch[self.input_label_column_index].values.astype(np.float32)
        return ndarrays, labels
    
    def remove_accidental_hits(self, targets, candidate_ids, logits):
        labels = torch.zeros(logits.shape)
        row_indices = torch.tensor(range(len(logits)), dtype=torch.long)
        labels[row_indices, targets] = 1
        
        candidate_ids = torch.unsqueeze(torch.tensor(candidate_ids), 1)      
        positive_indices = torch.tensor(targets)
        positive_candidate_ids = candidate_ids[positive_indices]
        
        duplicate = torch.eq(positive_candidate_ids, candidate_ids.T).type(labels.type())
        duplicate = duplicate - labels
        
        return logits +  duplicate * MIN_FLOAT
    
    def sampling_probability_correction(self, sampling_probability, scores):
        return scores - torch.log(torch.clamp(torch.tensor(sampling_probability), 1e-6, 1.))
    
    def cross_entropy_loss_with_sample_weight(self, scores, targets, sample_weight):
        loss = self.cross_entropy_loss_no_reduction(scores, targets)
        loss = loss * torch.tensor(sample_weight)
        return loss.mean()
    
    def update_progress(self, predictions, labels, loss):
        self.minibatch_id += 1
        self.update_metric(predictions, labels, loss)
        if self.minibatch_id % self.metric_update_interval == 0:
            self.push_metric()
            
    def _create_metric(self):
        metric = RetrievalModelMetric(use_auc=False)
        return metric
    
    def update_metric(self, predictions, labels, loss):
        self._metric.accumulate(predictions.data.numpy(), labels.data.numpy(), loss.data.numpy())
    
    def handle_request(self, req):
        import json
        body = json.loads(req.body)
        command = body.get('command')
        if command == 'PushMetric':
            states = ()
            for i in range(req.slice_count):
                states += req.get_slice(i),
            accum = self._metric
            delta = RetrievalModelMetric.from_states(states)
            accum.merge(delta)
            from datetime import datetime
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
            from metaspore._metaspore import Message
            res = Message()
            self.send_response(req, res)
            return
        super().handle_request(req)