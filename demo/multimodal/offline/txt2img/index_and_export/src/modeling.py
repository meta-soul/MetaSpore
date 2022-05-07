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

from typing import Dict, Union, List, Tuple
from collections import OrderedDict

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import transformers
from PIL import Image


class TextTransformerEncoder(nn.Sequential):

    def __init__(self, model_name_or_path, device=None, max_seq_len=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert = SentenceTransformer(model_name_or_path, device=device)
        if max_seq_len is not None:
            sbert._first_module().max_seq_length = max_seq_len
        super().__init__(sbert._modules)
        self.to(device)  # to device
        self._device = device
        self._max_len = sbert._first_module().max_seq_length
        self._do_lower_case = sbert._first_module().do_lower_case
        self._tokenizer = sbert.tokenizer
        self._input_names = self._tokenizer.model_input_names
        #self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._base_config = AutoConfig.from_pretrained(model_name_or_path)
        #self._base_model = AutoModel.from_pretrained(model_name_or_path)

    @property
    def max_seq_len(self):
        return self._max_len

    @property
    def do_lower_case(self):
        return self._do_lower_case

    @property
    def config(self):
        return self._base_config

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return ['sentence_embedding', 'token_embeddings']

    @property
    def input_axes(self):
        dynamic_axes = {}
        for name in self.input_names:
            dynamic_axes[name] = {0: 'batch_size', 1: 'max_seq_len'}
        return dynamic_axes

    @property
    def output_axes(self):
        dynamic_axes = {}
        dynamic_axes['sentence_embedding'] = {0: 'batch_size'}
        dynamic_axes['token_embeddings'] = {0: 'batch_size', 1: 'max_seq_len'}
        return dynamic_axes

    def save(self, save_path):
        self._tokenizer.save_pretrained(save_path)
        self._base_config.save_pretrained(save_path)

    def get_dummy_inputs(self, dummy=None, batch_size=1, device='cpu', return_tensors="pt"):
        text = dummy if dummy is not None else (" ".join([self._tokenizer.unk_token]) * 128)
        dummy_input = [text] * batch_size
        features = self.tokenize(dummy_input)
        inputs = {}
        for name in self.input_names:
            if return_tensors == "pt":
                inputs[name] = features[name].to(device)
            else:
                inputs[name] = features[name].cpu().numpy()
        return inputs

    def tokenize(self, texts: List[str]):
        if self._do_lower_case:
            texts = [s.lower() for s in texts]
        return self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self._max_len)

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
        for module in self:
            inputs = module(inputs)
        ret = OrderedDict()
        for name in self.output_names:
            ret[name] = inputs[name].detach()
        # normalize the sentence embedding
        ret['sentence_embedding'] = torch.nn.functional.normalize(ret['sentence_embedding'], p=2, dim=1)
        return ret

    def encode(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        features = self.tokenize(texts)
        features = {k:v.to(self._device) for k,v in features.items()}
        return self.forward(**features)


class CLIPImageEncoder(nn.Module):

    def __init__(self, model_name='openai/clip-vit-base-patch32', processor_name=None):
        super(CLIPImageEncoder, self).__init__()
        if processor_name is None:
            processor_name = model_name
        #self.model = transformers.CLIPModel.from_pretrained(model_name)
        #self.processor = transformers.CLIPProcessor.from_pretrained(processor_name)
        clip = transformers.CLIPModel.from_pretrained(model_name)
        self.model = clip.vision_model
        self.visual_projection = clip.visual_projection
        self.extractor = transformers.CLIPFeatureExtractor.from_pretrained(processor_name)

    def extract(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        return self.extractor(images, return_tensors="pt")

    def forward(self, pixel_values: Tensor=None, **kwargs):
        vision_outputs = self.model(pixel_values=pixel_values)
        image_embeds = self.visual_projection(vision_outputs[1])
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1) # norm
        ret = OrderedDict()
        ret['image_embedding'] = image_embeds
        return ret

    def _encode(self, batch, device=None, **kwargs):
        #features = self.processor(text=None, images=batch, return_tensors="pt", padding=True) 
        #vision_outputs = self.model.vision_model(pixel_values=features['pixel_values'])
        #image_embeds = self.model.visual_projection(vision_outputs[1])
        features = self.extractor(batch, return_tensors="pt")
        vision_outputs = self.model(pixel_values=features['pixel_values'].to(device=device))
        image_embeds = self.visual_projection(vision_outputs[1])
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1) # norm
        return image_embeds if device is None else image_embeds.to(device)

    def encode(self, images, batch_size=32, device=None):
        if isinstance(images, Image.Image):
            images = [images]

        embs = []
        batch = []
        for img in images:
            batch.append(img)
            if len(batch) == batch_size:
                embs.append(self._encode(batch, device))
                batch = []
        if batch:
            embs.append(self._encode(batch, device))

        ret = OrderedDict()
        ret["image_embedding"] = torch.cat(embs).float()
        return ret


if __name__ == '__main__':
    encoder = TextTransformerEncoder('bert-base-chinese', device='cuda:0')
    embs = encoder.encode('hello world!')['sentence_embedding']
    norm_embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    print(embs.size())
    print(embs, norm_embs)

    encoder = CLIPImageEncoder()
    img = Image.new('RGB', (256, 256))
    embs = encoder.encode(img)['image_embedding']
    norm_embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    print(embs, norm_embs)
    print(embs.size(), norm_embs.size())
