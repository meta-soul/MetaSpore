# adapted from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/TripletLoss.py

import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum

from .base_loss import BaseLoss

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(BaseLoss):
    
    def __init__(self, model: nn.Module, dual_model: nn.Module=None, distance_metric="EUCLIDEAN", triplet_margin: float = 5):
        super(TripletLoss, self).__init__(model, dual_model)
        distance_metric = vars(TripletDistanceMetric)[distance_metric]
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(TripletDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'triplet_margin': self.triplet_margin}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        num_sentences = len(sentence_features)  # must be 3
        assert num_sentences == 3
        reps = []
        # encode query
        reps.append(self.model(**sentence_features.pop(0))['sentence_embedding'])
        # encode passages
        if self.dual_model is None:
            reps.extend([self.model(**sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features])
        else:
            reps.extend([self.dual_model(**sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features])

        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()
