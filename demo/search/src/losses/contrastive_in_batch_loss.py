# adapted from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py
import torch
from torch import nn, Tensor
from typing import Iterable, Dict

from .base_loss import BaseLoss


class ContrastiveInBatchLoss(BaseLoss):

    def __init__(self, model: nn.Module, dual_model: nn.Module=None, log_scale: float = 2.6593):
        super(ContrastiveInBatchLoss, self).__init__(model, dual_model)
        self.scale = nn.Parameter(torch.tensor(log_scale))  # will be exp
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def get_config_dict(self):
        return {'scale': self.scale}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor=None):
        num_sentences = len(sentence_features)

        # encode query
        embeddings_a = self.model(**sentence_features.pop(0))['sentence_embedding']

        # encode passage, first is positive and others are negatives
        if self.dual_model is None:
            reps = [self.model(**sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        else:
            reps = [self.dual_model(**sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_b = torch.cat(reps)

        # normalize
        norm_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
        scores = torch.mm(norm_a, norm_b.transpose(0, 1)) * self.scale.exp()
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
        return self.cross_entropy_loss(scores, labels)
