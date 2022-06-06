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

from typing import List
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer


class TransformerDualEncoder(nn.Module):

    def __init__(self, model: SentenceTransformer, device=None, max_seq_len=None, normalize_output=True):
        super(TransformerDualEncoder, self).__init__()
        if max_seq_len is not None:
            model._first_module().max_seq_length = max_seq_len
        self._device = device
        self._max_len = model._first_module().max_seq_length
        self._do_lower_case = model._first_module().do_lower_case
        self._tokenizer = model.tokenizer
        self._input_names = self._tokenizer.model_input_names
        self.normalize_output = normalize_output
        self.model = model
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
    def output_names(self):
        return ['sentence_embedding', 'token_embeddings']

    def save_pretrained(self, save_path):
        self.model.save(save_path)

    @classmethod
    def load_pretrained(cls, model_name_or_path, **kwargs):
        return cls(SentenceTransformer(model_name_or_path), **kwargs)

    @classmethod
    def create_model(cls, model_name_or_path, max_seq_len=256, pooling="mean", 
            dense_features=-1, dense_act=nn.Tanh(), **kwargs):
        modules = []
        transformer_model = models.Transformer(model_name_or_path, max_seq_length=max_seq_len)
        modules.append(transformer_model)
        pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling)
        modules.append(pooling_model)
        if dense_features > 0:
            dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), 
                out_features=dense_features, activation_function=dense_act)
            modules.append(dense_model)
        return cls(SentenceTransformer(modules=modules), **kwargs)

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

    def forward(self, input_ids: Tensor=None, token_type_ids: Tensor=None, attention_mask: Tensor=None, positions_ids: Tensor=None, *args, **kwargs):
        inputs = {}
        if 'input_ids' in self.input_names:
            inputs['input_ids'] = input_ids
        if 'attention_mask' in self.input_names:
            inputs['attention_mask'] = attention_mask
        if 'token_type_ids' in self.input_names:
            inputs['token_type_ids'] = token_type_ids
        if 'positions_ids' in self.input_names:
            inputs['positions_ids'] = positions_ids
        outputs = self.model(inputs)
        ret = OrderedDict()
        for name in self.output_names:
            ret[name] = outputs[name]
        # normalize the sentence embedding
        if self.normalize_output:
            ret['sentence_embedding'] = F.normalize(ret['sentence_embedding'], p=2, dim=1)
        return ret

    def encode(self, texts: List[str], device=None, *args, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if device is None:
            device = next(self.parameters()).device
        features = self.tokenize(texts)
        features = {k:v.to(device) for k,v in features.items()}
        return self.forward(**features)['sentence_embedding']

def test_dual_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)

    texts = ['hello world!', 'abc']
    inputs = model.tokenize(texts)
    print(inputs)
    inputs = {k:v.to(device) for k,v in inputs.items()}

    print(model.tokenize('hello world', 'abc'))

    outputs = model(**inputs)
    print(outputs['sentence_embedding'].size())

    outputs = model.encode(texts, device)
    print(outputs.size())

if __name__ == '__main__':
    test_dual_encoder()
    exit()

    import sys
    import torch
    import numpy as np

    #embs_file = sys.argv[1]
    
    texts = []
    for line in sys.stdin:
        texts.append(line.strip())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerDualEncoder.load_pretrained('DMetaSoul/sbert-chinese-general-v2')
    model.to(device)
    
    with torch.no_grad():
        outputs = model.encode(texts, device)
        embs = outputs['sentence_embedding']

    embs = embs.cpu().numpy() if embs.is_cuda else embs.numpy()
    print(embs)
    #np.save(embs_file, embs)
