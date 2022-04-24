import os
import sys
import json
import argparse
from collections import OrderedDict
from typing import Dict, Union, List, Tuple

import onnx
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import onnxruntime

from sentence_transformers import SentenceTransformer


class TransformerEncoder(nn.Sequential):

    def __init__(self, model_name_or_path, device=None, max_seq_len=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sbert = SentenceTransformer(model_name_or_path, device=device)
        if max_seq_len is not None:
            sbert._first_module().max_seq_length = max_seq_len
        super().__init__(sbert._modules)
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

    def get_dummy_inputs(self, text='', batch_size=1, seq_length=128, device='cpu', return_tensors="pt"):
        text = text if text else (" ".join([self._tokenizer.unk_token]) * seq_length)
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
        #print(texts)
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
        return ret

    def encode(self, texts: List[str]):
        if isinstance(texts, str):
            texts = [texts]
        features = self.tokenize(texts)
        return self.forward(**features)


def validate_onnx_model(model, onnx_path, device='cpu', print_model=False, rtol=1e-03, atol=1e-05):
    # Check that the exported model is well formed
    print("Checking ONNX model format...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("Checking done!")

    # Print a human readable representation of the graph
    if print_model:
        print(onnx.helper.printable_graph(onnx_model.graph))

    # Verify that ONNX Runtime and PyTorch are computing the same value for the network
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    #print([x.name for x ort_session.get_inputs()])

    output_names = model.output_names
    model = model.to(device)
    model.eval()
    tensor_inputs = model.get_dummy_inputs(batch_size=1, return_tensors="pt", device=device)
    torch_outs = model(**tensor_inputs)
    torch_out_keys = set(torch_outs.keys())

    print("Validating ONNX model...")
    ort_inputs = {k:v.cpu().numpy() for k, v in tensor_inputs.items()}
    ort_out_keys = set(output_names)
    ort_outs = ort_session.run(output_names, ort_inputs)
    #ort_outs = ort_session.run(None, ort_inputs)
    if not ort_out_keys.issubset(torch_out_keys):
        print(f"\t-[x] ONNX model output names {ort_out_keys} do not match reference model {ort_out_keys}")
        raise ValueError("Model validation failed!")
    else:
        print(f"\t-[✓] ONNX model output names match reference model ({ort_out_keys})")

    for name, ort_value in zip(output_names, ort_outs):
        print(f'\t- Validating ONNX Model output "{name}":')
        ref_value = torch_outs[name].numpy()

        if not ort_value.shape == ref_value.shape:
            print(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError("Model validation failed!")
        else:
            print(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        if not np.allclose(ref_value, ort_value, atol=atol, rtol=rtol):
            print(f"\t\t-[x] values not close enough (atol: {atol}, rtol: {rtol})")
            raise ValueError("Model validation failed!")
        else:
            print(f"\t\t-[✓] all values close (atol: {atol}, rtol: {rtol})")

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def onnx_export(model_name_or_path, export_path, onnx_name='model.onnx', config_name='onnx_config.json', device='cpu', validate=True, save_pretrained=True, dummy_text='hello world!'):
    onnx_path = os.path.join(export_path, onnx_name)
    config_path = os.path.join(export_path, config_name)
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
    # load transformer model
    model = TransformerEncoder(model_name_or_path)
    if save_pretrained:
        model.save(export_path)
    # export model via onnx
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        model_inputs = model.get_dummy_inputs(device=device, text=dummy_text)
        #args = []
        #for name in model.input_names:
        #    args.append(model_inputs[name])

        dynamic_axes = {}
        dynamic_axes.update(model.input_axes)
        dynamic_axes.update(model.output_axes)

        torch.onnx.export(model,
            #args=tuple(args),
            (model_inputs,),
            f=onnx_path,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes=dynamic_axes,
            #verbose=True,
            do_constant_folding=True,  # whether to execute constant folding for optimization
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=11
        )
    # validate the exported model
    if validate:
        validate_onnx_model(model, onnx_path, print_model=True, device=device)
    # dump config
    config = {}
    config['export_path'] = export_path
    config['onnx_path'] = onnx_path
    config['model_name_or_path'] = model_name_or_path
    config['do_lower_case'] = model.do_lower_case
    config['max_seq_len'] = model.max_seq_len
    config['onnx_inputs'] = model.input_names
    config['onnx_outputs'] = model.output_names
    with open(config_path, 'w', encoding='utf8') as fout:
        json.dump(config, fout, indent=4)
    return model.input_names, model.output_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="The sbert model name or path to be exported.")
    parser.add_argument("--onnx-path", required=True, help="The path of onnx exported directory.")
    args = parser.parse_args()

    model_name, onnx_path = args.model_name, args.onnx_path
    print('model export...')
    onnx_inputs, onnx_outputs = onnx_export(model_name, onnx_path, validate=True)
    print('onnx inputs', onnx_inputs)
    print('onnx outputs', onnx_outputs)
