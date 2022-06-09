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
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import transformers
from PIL import Image


class CLIPTextEncoder(nn.Module):

    def __init__(self, model_name_or_path='openai/clip-vit-base-patch32', device=None, 
            max_seq_len=512, do_lower_case=True):
        super(CLIPTextEncoder, self).__init__()
        device = 'cpu' if device is None else device
        #self.config = transformers.CLIPTextConfig.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path).text_config
        #self.tokenizer = transformers.CLIPTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        clip_model = AutoModel.from_pretrained(model_name_or_path)
        self.model = clip_model.text_model
        self.text_projection = clip_model.text_projection
        self.to(device)
        self._device = device 
        self._max_len = max_seq_len
        self._do_lower_case = do_lower_case
        self._input_names = self.tokenizer.model_input_names

    @property
    def max_seq_len(self):
        return self._max_len

    @property
    def do_lower_case(self):
        return self._do_lower_case

    @property
    def preprocessor_kwargs(self):
        return {'max_seq_len': self.max_seq_len, 'do_lower_case': self.do_lower_case}

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return ['sentence_embedding']

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
        return dynamic_axes

    def save(self, save_path):
        self.tokenizer.save_pretrained(save_path)
        self.config.save_pretrained(save_path)

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

    def tokenize(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        if self.do_lower_case:
            texts = [s.lower() for s in texts]
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_seq_len)

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

        text_outputs = self.model(**inputs)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=1)
        ret = OrderedDict()
        ret['sentence_embedding'] = text_embeds
        return ret

    def encode(self, texts: List[str]):
        features = self.tokenize(texts)
        features = {k:v.to(self._device) for k,v in features.items()}
        return self.forward(**features)


class CLIPImageEncoder(nn.Module):

    def __init__(self, model_name_or_path='openai/clip-vit-base-patch32', device=None):
        super(CLIPImageEncoder, self).__init__()
        device = 'cpu' if device is None else device
        #self.config = transformers.CLIPVisionConfig.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path).vision_config
        #self.extractor = transformers.CLIPFeatureExtractor.from_pretrained(model_name_or_path)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        clip_model = AutoModel.from_pretrained(model_name_or_path)
        self.model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection
        self.to(device)
        self._input_names = self.extractor.model_input_names

    @property
    def preprocessor_kwargs(self):
        return {}

    @property
    def input_names(self):
        #return self._input_names
        return ['pixel_values']

    @property
    def output_names(self):
        return ['image_embedding']

    @property
    def input_axes(self):
        dynamic_axes = {
            'pixel_values': {0: 'batch_size'}
        }
        return dynamic_axes

    @property
    def output_axes(self):
        dynamic_axes = {
            'image_embedding': {0: 'batch_size'}
        }
        return dynamic_axes

    def get_dummy_inputs(self, dummy=None, batch_size=1, return_tensors="pt", device='cpu'):
        imgs = []
        for i in range(batch_size):
            if dummy is not None:
                imgs.append(dummy)
            else:
                imgs.append(Image.new('RGB', (256, 256)))
        features = self.extract(imgs)
        inputs = {}
        for name in self.input_names:
            if return_tensors == "pt":
                inputs[name] = features[name].to(device)
            else:
                inputs[name] = features[name].cpu().numpy()
        return inputs

    def save(self, save_path):
        self.config.save_pretrained(save_path)
        self.extractor.save_pretrained(save_path)

    def extract(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        return self.extractor(images, return_tensors="pt")

    def forward(self, pixel_values: Tensor=None, *args, **kwargs):
        vision_outputs = self.model(pixel_values=pixel_values)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1) # norm
        ret = OrderedDict()
        ret['image_embedding'] = image_embeds
        return ret

    def _encode(self, batch, device=None, **kwargs):
        features = self.extractor(batch, return_tensors="pt")
        vision_outputs = self.model(pixel_values=features['pixel_values'].to(device=device))
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
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
    text_model = CLIPTextEncoder()
    text_embs = text_model.encode('a b c')
    print(text_embs['sentence_embedding'].size())

    image_model = CLIPImageEncoder()
    image_embs = image_model.encode(Image.new('RGB', (256, 256)))
    print(image_embs['image_embedding'].size())
