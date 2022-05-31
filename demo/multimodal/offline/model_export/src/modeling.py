from typing import Dict, Union, List, Tuple
from collections import OrderedDict

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer
import transformers
from PIL import Image

from modeling_clip import CLIPTextEncoder, CLIPImageEncoder


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
    def preprocessor_kwargs(self):
        return {'max_seq_len': self.max_seq_len, 'do_lower_case': self.do_lower_case}

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


class ImageTransformerEncoder(nn.Module):

    def __init__(self, model_name_or_path, device=None):
        super(ImageTransformerEncoder, self).__init__()
        device = 'cpu' if device is None else device
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.model.to(device)

    @property
    def preprocessor_kwargs(self):
        return {}

    @property
    def input_names(self):
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
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=1) # norm
        ret = OrderedDict()
        ret['image_embedding'] = image_embeds
        return ret

    def _encode(self, batch, device=None, **kwargs):
        features = self.extractor(batch, return_tensors="pt")
        vision_outputs = self.model(pixel_values=features['pixel_values'].to(device=device))
        image_embeds = vision_outputs[1]
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

def test_text_encoder():
    #model_name = 'sentence-transformers/clip-ViT-B-32-multilingual-v1'
    model_name = 'DMetaSoul/sbert-chinese-qmc-domain-v1'
    encoder = TextTransformerEncoder(model_name, device='cpu')
    inputs = encoder.tokenize('a b c')
    print(inputs, inputs['input_ids'].dtype)

    embs = encoder.encode('hello world!')
    embs = {k:v.numpy() for k,v in embs.items()}
    for name in embs:
        print(name, embs[name].shape)
    #print(embs, norm_embs)
    #import pickle
    #with open("embs.pkl", "wb") as f:
    #    pickle.dump(embs, f)

def test_image_encoder():
    model_name = 'google/vit-base-patch16-224-in21k'
    encoder = ImageTransformerEncoder(model_name)
    img = Image.new('RGB', (256, 256))
    print(encoder.extract(img))

    embs = encoder.encode(img)['image_embedding']
    print(embs, embs.size())

    inputs = encoder.get_dummy_inputs()
    print(inputs)

    from torch.onnx.utils import _decide_input_format as get_tuple_model_args
    args = (encoder.get_dummy_inputs(), )
    print(get_tuple_model_args(encoder, args))


if __name__ == '__main__':
    test_text_encoder()
    test_image_encoder()
