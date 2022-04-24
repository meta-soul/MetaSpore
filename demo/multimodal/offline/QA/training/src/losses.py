import os
import copy
import json
import logging
from typing import Union, Tuple, List, Iterable, Dict, Callable

import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, models, losses, util

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator, LabelAccuracyEvaluator

logger = logging.getLogger(__name__)


class SoftmaxLoss(losses.SoftmaxLoss):

    def __init__(self, model, num_labels, concatenation_sent_rep=True, concatenation_sent_difference=True, concatenation_sent_multiplication=False, **kwargs):
        sent_emb_dim = model.get_sentence_embedding_dimension()
        super(SoftmaxLoss, self).__init__(model, sentence_embedding_dimension=sent_emb_dim, concatenation_sent_rep=concatenation_sent_rep, concatenation_sent_difference=concatenation_sent_difference, concatenation_sent_multiplication=concatenation_sent_multiplication, num_labels=num_labels, loss_fct=nn.CrossEntropyLoss())


class LogisticLoss(nn.Module):
    """
    :param model: SentenceTransformer model
    :param num_labels: Number of different labels
    :param concatenation_sent_rep: Concatenate vectors u,v for the softmax classifier?
    :param concatenation_sent_difference: Add abs(u-v) for the softmax classifier?
    :param concatenation_sent_multiplication: Add u*v for the softmax classifier?
    :param loss_fct: Optional: Custom pytorch loss function. If not set, uses nn.BCEWithLogitsLoss()
    """
    def __init__(self,
                 model: SentenceTransformer,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 num_labels: int = 1,
                 loss_fct: Callable = nn.BCEWithLogitsLoss()):
        super(LogisticLoss, self).__init__()
        self.model = model
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logger.info("Logistic loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        sentence_embedding_dimension = model.get_sentence_embedding_dimension()
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.loss_fct = loss_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)

        if labels is not None:
            #loss = self.loss_fct(output, labels.view(-1))
            #print(output.shape, labels.shape)
            loss = self.loss_fct(output.view(-1), labels)
            return loss
        else:
            return reps, output


class MoCoMultipleNegativesRankingLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, gamma=0.99, q_size=128):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        :param gamma: the momentum encoder coefficient
        :param q_size: the size of momentum-updated queue
        """
        super(MoCoMultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        # for momentum-updated
        self.gamma = gamma
        self.q = []
        self.q_size = q_size
        #self.momentum_model = copy.deepcopy(model)
        self.momentum_model = SentenceTransformer(model[0].auto_model.config._name_or_path, device=model._target_device)
        for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def enqueue(self, features):
        if isinstance(features, dict):
            embs = self.momentum_model(features)['sentence_embedding']
        else:
            embs = features
        batch_size = embs.size(0)
        if len(self.q)+batch_size >= self.q_size:
            del self.q[:batch_size]
        self.q.extend(embs)

    def dequeue(self, batch_size=-1):
        embeddings = None
        if len(self.q) > 0:
            embeddings = torch.vstack(self.q[:batch_size])
        return embeddings

    def get_queue(self):
        embeddings = None
        if len(self.q) > 0:
            embeddings = torch.vstack([t.detach() for t in self.q])
        return embeddings

    #def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
    #    reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
    #    embeddings_a = reps[0]
    #    embeddings_b = torch.cat(reps[1:])

    #    # for momentum-updated
    #    embeddings_neg = None
    #    features_a = sentence_features[0]
    #    batch_size = embeddings_a.size(0)
    #    with torch.no_grad():
    #        # dequeue
    #        embeddings_neg = self.dequeue(batch_size)
    #        # enqueue
    #        self.enqueue(features_a)
    #        # update momentum encoder
    #        for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
    #            param_k.data = self.gamma * param_k.data + param_q.data * (1. - self.gamma)

    #    if embeddings_neg is not None:
    #        embeddings_b = torch.cat([embeddings_b, embeddings_neg], dim=0)

    #    scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
    #    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
    #    return self.cross_entropy_loss(scores, labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        features_q, features_k = sentence_features
        embeddings_q = self.model(features_q)['sentence_embedding']

        # for momentum-updated
        embeddings_neg = None
        with torch.no_grad():
            # update momentum encoder
            for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
                param_k.data = self.gamma * param_k.data + param_q.data * (1. - self.gamma)
            # encoder k
            embeddings_k = self.momentum_model(features_k)['sentence_embedding']
            # get queue k
            embeddings_neg = self.get_queue()
            # enqueue
            self.enqueue(embeddings_k)

        if embeddings_neg is not None:
            embeddings_k = torch.cat([embeddings_k, embeddings_neg], dim=0)

        scores = self.similarity_fct(embeddings_q, embeddings_k) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}

class CosineCircleLoss(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0):
        super(CosineCircleLoss, self).__init__()
        self.model = model
        self.scale = scale

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # input sentence must be a pair
        embeddings_a, embeddings_b = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # consine similarity of each pair i
        embeddings_a = F.normalize(embeddings_a, p=2, dim=-1, eps=1e-8)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=-1, eps=1e-8)
        cosine_sim = torch.sum(embeddings_a * embeddings_b, dim=1) * self.scale
        # collect all pairs (i, j) if true_i>true_j
        cosine_sim = cosine_sim[:, None] - cosine_sim[None, :] # diff of pair (i, j)
        labels = (labels[:, None] < labels[None, :]).long()  # true similar relation between i and j
        cosine_sim = cosine_sim - (1 - labels) * 1e12  # for all pair i<j, plus a large int to disapper in the next exp op
        cosine_sim = torch.cat((torch.zeros(1).to(cosine_sim.device), cosine_sim.view(-1)), dim=0) # add exp^0
        return torch.logsumexp(cosine_sim.view(-1), dim=0)

    def get_config_dict(self):
        return {'scale': self.scale}
