# adapted from: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
from enum import Enum
from typing import Iterable, Dict
import torch.nn.functional as F
from torch import nn, Tensor

from .base_loss import BaseLoss

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)


class ContrastiveLoss(BaseLoss):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, model: nn.Module, dual_model: nn.Module=None, distance_metric="COSINE_DISTANCE", margin: float = 0.5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__(model, dual_model)
        self.distance_metric = vars(SiameseDistanceMetric)[distance_metric] 
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'margin': self.margin, 'size_average': self.size_average}

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        num_sentences = len(sentence_features)
        assert num_sentences == 2

        reps = []
        reps.append(self.model(**sentence_features.pop(0))['sentence_embedding'])
        if self.dual_model is None:
            reps.append(self.model(**sentence_features.pop(0))['sentence_embedding'])
        else:
            reps.append(self.dual_model(**sentence_features.pop(0))['sentence_embedding'])

        rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()
