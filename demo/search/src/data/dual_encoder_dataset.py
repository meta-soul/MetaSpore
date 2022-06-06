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

from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader


class DualEncoderPairDataset(Dataset):
    """Pair dataset with or without label: (query, passage) or (query, passage, label)"""

    def __init__(self, data_file, with_label=False, text_indices=[0, 1], label_index=-1, label_converter=int):
        assert len(text_indices) == 2
        self.examples = []
        Example = namedtuple('Example', ['texts', 'label'])
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                row = line.rstrip('\n').split('\t')
                texts = []
                for i in text_indices:
                    row[i] = row[i].replace(' ', '')  # remove space in the text
                    texts.append(row[i])
                label = None if not with_label else label_converter(row[label_index])
                self.examples.append(Example(texts, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DualEncoderTripletDataset(Dataset):
    """Triplet dataset: (query, pos_passage, neg_passage)"""
    
    def __init__(self, data_file, text_indices=[0, 1, 2]):
        assert len(text_indices) == 3
        self.examples = []
        Example = namedtuple('Example', ['texts', 'label'])
        with open(data_file, 'r', encoding='utf8') as fin:
            for line in fin:
                row = line.rstrip('\n').split('\t')
                texts = []
                for i in text_indices:
                    row[i] = row[i].replace(' ', '')  # remove space in the text
                    texts.append(row[i])
                self.examples.append(Example(texts, None))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DualEncoderCollater(object):

    def __init__(self, tokenizers, device=None):
        self.tokenizers = tokenizers
        self.device = device

    def __call__(self, batch):
        features_list = []
        num_texts = len(batch[0].texts)
        for i in range(num_texts):
            texts = [e.texts[i] for e in batch]
            features = self.tokenizers[i](texts)
            features_list.append(features)

        if batch[0].label is None:
            labels = None
        elif isinstance(batch[0].label, int):
            labels = torch.tensor([e.label for e in batch]).long()
        else:
            labels = torch.tensor([e.label for e in batch]).float()

        if self.device is not None:
            features_list = [{k:v.to(self.device) for k,v in features.items()} for features in features_list]
            labels = labels.to(self.device) if labels is not None else None

        return features_list, labels


def create_dual_encoder_dataset(data_file, data_kind='pair', text_indices=[0, 2], label_index=-1, label_converter=int):
    if data_kind == 'pair':
        dataset = DualEncoderPairDataset(data_file, 
            with_label=False, text_indices=text_indices, label_index=-1, label_converter=None)
    elif data_kind == 'pair_with_label':
        dataset = DualEncoderPairDataset(data_file, 
            with_label=True, text_indices=text_indices, label_index=label_index, label_converter=label_converter)
    elif data_kind == 'triplet':
        dataset = DualEncoderTripletDataset(data_file, text_indices=text_indices)
    else:
        dataset = None
    return dataset


def create_dual_encoder_dataloader(data_file, data_kind, tokenizers, text_indices=[0, 1], label_index=-1, label_converter=int,
        batch_size=64, shuffle=True, device=None, num_workers=0, pin_memory=False, drop_last=False):
    assert len(tokenizers) == len(text_indices), "Each text column must match a tokenizer"
    assert data_kind in ['pair', 'pair_with_label', 'triplet'], "The data kind is invalid!"
    collater = DualEncoderCollater(tokenizers, device)
    dataset = create_dual_encoder_dataset(data_file, data_kind, text_indices, label_index, label_converter)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=collater, pin_memory=pin_memory, 
        drop_last=drop_last)

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    def tokenize(text_a, text_b=None):
        return tokenizer(text_a, text_b, return_tensors='pt', padding=True, truncation=True)

    data_file = '../../data/train/train.pos.tsv'
    data_kind = 'pair'
    dataloader = create_dual_encoder_dataloader(data_file, data_kind, [tokenize, tokenize], text_indices=[0, 1], device='cuda:0')
    for features, labels in dataloader:
        a_features, b_features = features
        print('features', a_features, b_features)
        print('labels', labels)
        break

    data_file = '../../data/train/train.rand.neg.pair.tsv'
    data_kind = 'pair_with_label'
    dataloader = create_dual_encoder_dataloader(data_file, data_kind, [tokenize, tokenize], text_indices=[0, 1], 
        label_index=2, label_converter=int, device='cuda:0')
    for features, labels in dataloader:
        a_features, b_features = features
        print('features', a_features, b_features)
        print('labels', labels)
        break
