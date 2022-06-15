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

import os
import sys

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


class TransformerCrossEncoder(nn.Module):

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, 
            device=None, max_seq_len=None, do_lower_case=False, num_labels=1, task_type='multilabel'):
        assert task_type in ['multiclass', 'multilabel', 'regression']
        super(TransformerCrossEncoder, self).__init__()
        self.model = model
        self._tokenizer = tokenizer
        self._max_len = max_seq_len
        self._do_lower_case = do_lower_case
        self._input_names = tokenizer.model_input_names
        self._num_labels = num_labels
        self._task_type = task_type
        if task_type == "regression":
            self.loss_fct = nn.MSELoss()
            self.act_fct = nn.Identity()
        elif task_type == "multilabel":
            self.loss_fct = nn.BCEWithLogitsLoss()
            self.act_fct = nn.Sigmoid()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
            self.act_fct = nn.Softmax(dim=1)
        if device is not None:
            self.to(device)

    @property
    def max_seq_len(self):
        return self._max_len

    @property
    def do_lower_case(self):
        return self._do_lower_case

    @property
    def input_names(self):
        return self._input_names

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def output_names(self):
        return ['logits']  # during train output (logits, loss), during eval output logits

    @classmethod
    def load_pretrained(cls, model_name_or_path, num_labels=1, task_type='multiclass', tokenizer_args={}, **kwargs):
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_args)
        return cls(model, tokenizer, num_labels=num_labels, task_type=task_type, **kwargs)

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def save(self, save_path):
        self.save_pretrained(save_path)

    def tokenize(self, text, text_pair=None, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt",
            device=None):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        if self._do_lower_case:
            text = [s.lower() for s in text]
        features = self._tokenizer(text, text_pair=text_pair, padding=padding, truncation=truncation, 
            add_special_tokens=add_special_tokens, return_tensors=return_tensors, 
            max_length=self._max_len)
        if device is not None:
            features = {k:v.to(device) for k, v in features.items()}
        return features

    def forward(self, input_ids: Tensor=None, token_type_ids: Tensor=None, attention_mask: Tensor=None, positions_ids: Tensor=None, labels: Tensor=None, *args, **kwargs):
        inputs = {}
        if 'input_ids' in self.input_names:
            inputs['input_ids'] = input_ids
        if 'attention_mask' in self.input_names:
            inputs['attention_mask'] = attention_mask
        if 'token_type_ids' in self.input_names:
            inputs['token_type_ids'] = token_type_ids
        if 'positions_ids' in self.input_names:
            inputs['positions_ids'] = positions_ids
        #outputs = self.model(**inputs, labels=labels)
        outputs = self.model(**inputs)
        logits = outputs.logits
        if labels is None:
            return logits
        #print(logits.shape, labels.shape)
        loss = self.loss_fct(logits, labels)
        return logits, loss

    def predict(self, sentences, batch_size=32, convert_to_numpy=True, convert_to_tensor=False, device=None, **kwargs):
        if device is None:
            device = next(self.parameters()).device

        def collate_fn(batch):
            if isinstance(batch[0], list):
                texts_a = [x[0] for x in batch]
                texts_b = [x[1] for x in batch]
            else:
                texts_a = batch 
                texts_b = None
            features = self.tokenize(texts_a, texts_b)
            features = {k:v.to(device) for k, v in features.items()}
            return features

        data_loader = DataLoader(sentences, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=0, shuffle=False)

        pred_scores = []
        with torch.no_grad():
            for features in data_loader:
                logits = self.forward(**features)
                scores = self.act_fct(logits)
                pred_scores.extend(scores)
        
        if self._num_labels == 1:
            pred_scores = [scores[0] for scores in pred_scores]  # only has one output
        elif self._num_labels == 2 and self._task_type == "multiclass":
            pred_scores = [scores[1] for scores in pred_scores]  # only return positive score

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        return pred_scores
        

def test_cross_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerCrossEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2', 
        num_labels=2, task_type='multiclass')
    model.to(device)

    texts = [['hello', 'world!'], ['abc', '123']]

    # for multiclass task with num_labels=2
    labels = torch.empty(len(texts), dtype=torch.long).random_(2) # with shape (batch,)
    print(model.predict(texts, device=device), labels)

    # for multilabel task with num_labels=5
    #labels = torch.empty(len(texts), 5, dtype=torch.long).random_(2).float()  # with shape (batch, num_labels)
    #print(model.predict(texts, device=device))

    # for multilabel task with num_labels=1
    #labels = torch.randn(len(texts), 1)
    #print(model.predict(texts, device=device))

    # for regression task with num_labels=1
    #labels = torch.randn(2, 1) # with shape (batch, num_labels)
    #print(model.predict(texts, device=device))

    labels = labels.to(device)
    #print(labels)
    texts_a, texts_b = zip(*texts)
    features = model.tokenize(texts_a, texts_b, device=device)
    #print(features)
    outputs = model(**features, labels=labels)
    print(outputs)
    

if __name__ == '__main__':
    test_cross_encoder()
