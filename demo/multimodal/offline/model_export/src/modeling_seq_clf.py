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

from collections import OrderedDict
from typing import Dict, Union, List, Tuple

import torch
from torch import nn, Tensor

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


class SeqTransformerClassifier(nn.Module):

    def __init__(self, model_name_or_path, device=None, max_seq_len=512, do_lower_case=True):
        super(SeqTransformerClassifier, self).__init__()
        device = 'cpu' if device is None else device
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.to(device)
        self._input_names = self.tokenizer.model_input_names
        self.do_lower_case = do_lower_case
        self.max_seq_len = max_seq_len

    @property
    def preprocessor_kwargs(self):
        return {'max_seq_len': self.max_seq_len, 'do_lower_case': self.do_lower_case}

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return ['logits']

    @property
    def input_axes(self):
        dynamic_axes = {}
        for name in self.input_names:
            dynamic_axes[name] = {0: 'batch_size', 1: 'max_seq_len'}
        return dynamic_axes

    @property
    def output_axes(self):
        dynamic_axes = {
            'logits': {0: 'batch_size'}
        }
        return dynamic_axes

    def tokenize(self, text, text_pair=None, padding=True, truncation=True, add_special_tokens=True, return_tensors="pt"):
        if isinstance(text, str):
            text = [text]
        if isinstance(text_pair, str):
            text_pair = [text_pair]
        if self.do_lower_case:
            text = [s.lower() for s in text]
            text_pair = [s.lower() for s in text_pair] if text_pair is not None else None
        features = self.tokenizer(text, text_pair=text_pair, padding=True, truncation=True, 
            add_special_tokens=True, return_tensors="pt", max_length=self.max_seq_len)
        return features

    def get_dummy_inputs(self, dummy=None, batch_size=1, device='cpu', return_tensors="pt"):
        text = dummy if dummy is not None else (" ".join([self.tokenizer.unk_token]) * 128)
        dummy_input = [text] * batch_size
        features = self.tokenize(dummy_input)
        inputs = {}
        for name in self.input_names:
            if return_tensors == "pt":
                inputs[name] = features[name].to(device)
            else:
                inputs[name] = features[name].cpu().numpy()
        return inputs

    def save(self, save_path):
        self.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

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
        outputs = self.model(**inputs)
        logits = outputs.logits
        ret = OrderedDict()
        ret["logits"] = logits
        return ret


if __name__ == '__main__':
    model_name_or_path = 'bert-base-chinese'
    clf = SeqTransformerClassifier('bert-base-chinese')
    print(clf.input_names)
    inputs = clf.get_dummy_inputs()
    print(inputs)
    outputs = clf(**inputs)
    print(outputs)
