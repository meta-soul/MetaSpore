import metaspore as ms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def _default_train_minibatch(self, minibatch):
        # prepare the training process
        self.model.train()
        minibatch, labels = self.preprocess_minibatch(minibatch)
        predictions, scores, targets = self.model(minibatch)
        # temperature control
        if self.tau is not None and self.tau > 1e-6:
            scores = scores/self.tau
        # remove accidental hits
        if self.use_remove_accidental_hits:
            candidate_ids = minibatch[self.input_item_id_column_name].values.astype(np.long)
            scores = self.remove_accidental_hits(targets, candidate_ids, scores)
        # sampling probability correction
        if self.use_sampling_probability_correction:
            sampling_probability = minibatch[self.input_item_probability_column_name]
            scores = self.sampling_probability_correction(sampling_probability, scores)
        # sample weight
        if self.use_sample_weight:
            sample_weight = minibatch[self.input_sample_weight_column_name]
            loss = self.cross_entropy_loss_with_sample_weight(scores, targets, sample_weight)
        else:
            loss = self.cross_entropy_loss(scores, targets)
        # backward the loss
        self.trainer.train(loss)
        # update trainning progress
        labels = torch.from_numpy(labels).reshape(-1, 1)
        self.update_progress(batch_size=len(labels), batch_loss=loss)

    def remove_accidental_hits(self, targets, candidate_ids, logits):
        labels = torch.zeros(logits.shape)
        row_indices = torch.tensor(range(len(logits)), dtype=torch.long)
        labels[row_indices, targets] = 1

        candidate_ids = torch.unsqueeze(torch.tensor(candidate_ids), 1)
        positive_indices = targets.clone().detach()
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
