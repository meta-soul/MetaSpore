import os
import sys
import json

import torch
from torch import nn
from sentence_transformers import SentenceTransformer, models, losses, util
#from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from losses import LogisticLoss, SoftmaxLoss, MoCoMultipleNegativesRankingLoss, CosineCircleLoss
from dataset import STSDataset, NLIDataset, QMCDataset, QueryPositivePairDataset, QueryPositiveNegativeTripletDataset, QuerySingleDataset


TASK_LISTS = {
    'sts': {
        "class": STSDataset, 
        "losses": [{
            "name": "cosine", 
            "class": losses.CosineSimilarityLoss, 
            "kwargs": {}
        }, {
            "name": "circle",
            "class": CosineCircleLoss,
            "kwargs": {"scale": 20.0}
        }]
    },
    'nli': {
        "class": NLIDataset, 
        "losses": [{
            "name": "softmax3", 
            "class": SoftmaxLoss, 
            "kwargs": {"num_labels": 3}
        }, {
            "name": "ranking", 
            "class": losses.MultipleNegativesRankingLoss, 
            "kwargs": {"scale": 20.0}
        }, {
            "name": "circle",
            "class": CosineCircleLoss,
            "kwargs": {"scale": 20.0}
        }]
    },
    'qmc': {
        "class": QMCDataset, 
        "losses": [{
            "name": "contrastive", 
            "class": losses.OnlineContrastiveLoss, 
            "kwargs": {"margin": 0.5}
        }, {
            "name": "ranking", 
            "class": losses.MultipleNegativesRankingLoss, 
            "kwargs": {"scale": 20.0}
        }, {
            "name": "cosine", 
            "class": losses.CosineSimilarityLoss, 
            "kwargs": {}
        }, {
            "name": "logistic", 
            "class": LogisticLoss, 
            "kwargs": {}
        }, {
            "name": "circle",
            "class": CosineCircleLoss,
            "kwargs": {"scale": 20.0}
        }]
    },
    "single": {
        "class": QuerySingleDataset,
        "losses": [{
            "name": "simcse",
            "class": losses.MultipleNegativesRankingLoss,
            "kwargs": {"scale": 20.0}
        }, {
            "name": "esimcse",
            "class": MoCoMultipleNegativesRankingLoss,
            "kwargs": {"scale": 20.0, "gamma": 0.99, "q_size": 128}
        }, {
            "name": "tsdae",
            "class": losses.DenoisingAutoEncoderLoss,
            "kwargs": {"tie_encoder_decoder": True, "decoder_name_or_path": None}
        }, {
            "name": "ct",
            "class": losses.ContrastiveTensionLoss,
            "kwargs": {}
        }, {
            "name": "ct2",
            "class": losses.ContrastiveTensionLossInBatchNegatives,
            "kwargs": {"scale": 20.0}
        }]
    },
    'pair': {
        "class": QueryPositivePairDataset, 
        "losses": [{
            "name": "ranking", 
            "class": losses.MultipleNegativesRankingLoss,
            "kwargs": {"scale": 20.0}
        }]
    },
    'triplet': {
        "class": QueryPositiveNegativeTripletDataset, 
        "losses": [{
            "name": "ranking", 
            "class": losses.MultipleNegativesRankingLoss,
            "kwargs": {"scale": 20.0}
        }, {
            "name": "triplet", 
            "class": losses.TripletLoss,
            "kwargs": {"triplet_margin": 5.0}
        }]
    },
}

class TaskConfig(object):

    @classmethod
    def task_list(cls):
        return list(TASK_LISTS.keys())

    @classmethod
    def dump_config(cls, file):
        cfg = {"losses": {}}
        for task, item in TASK_LISTS.items():
            for x in item['losses']:
                if x['name'] not in cfg['losses']:
                    cfg['losses'][x['name']] = {}
                cfg['losses'][x['name']].update(x.get('kwargs', {}))
        with open(file, 'w', encoding='utf8') as fout:
            json.dump(cfg, fout, indent=4)

    def __init__(self, name, loss_params_file=''):
        assert name in TASK_LISTS, "Invalid task: {}!".format(name)
        self._cfg = TASK_LISTS[name]
        self.name = name
        self.data_class = self._cfg['class']
        self.loss_classes = {}
        self.loss_kwargs = {}
        self.loss_params = {}
        if loss_params_file and os.path.exists(loss_params_file):
            with open(loss_params_file, 'r') as fin:
                params_data = json.load(fin)
                if isinstance(params_data, dict):
                    self.loss_params = params_data.get('losses', {})
        # default loss of task
        self.add_loss_class(self._cfg['losses'][0]['name'])

    #def get_task_losses(self):
    #    if self.name not in TASK_LISTS:
    #        return []
    #    return [x['name'] for x in TASK_LISTS[self.name]['losses']]

    def load_dataset(self, data_file, *args, **kwargs):
        return self.data_class(data_file).load(*args, **kwargs)

    def get_losses(self):
        losses = {}
        for name in self.loss_classes:
            losses[name] = {'class': self.loss_classes[name], 'kwargs': self.loss_kwargs[name]}
        return losses

    def clear_losses(self):
        self.loss_classes = {}
        self.loss_kwargs = {}

    def add_loss_class(self, name):
        if name in self.loss_classes:
            return 0
        for x in self._cfg['losses']:
            if x['name'] == name:
                self.loss_classes[name] = x['class']
                self.loss_kwargs[name] = x.get('kwargs', {}).copy()
                self.loss_kwargs[name].update(self.loss_params.get(name, {}))
                return 1
        return -1

    def add_losses_from_str(self, names, delim=','):
        n = 0
        for name in names.split(delim):
            ret = self.add_loss_class(name)
            assert ret >= 0, f"Invalid loss: {name}!"
            n += ret
        return n


if __name__ == '__main__':
    if sys.argv[1] ==  'dump':
        TaskConfig.dump_config('./params.json')
