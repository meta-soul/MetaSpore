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
import json

import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer, AutoFeatureExtractor

PROCESSORS = {'tokenizer': AutoTokenizer, 'extractor': AutoFeatureExtractor}

class HfBasePreprocessor(object):

    def __init__(self, name, config_dir):
        self.processor = PROCESSORS[name].from_pretrained(config_dir)

        export_conf = json.load(open(os.path.join(config_dir, 'export_config.json'), 'r'))
        self.raw_encoding = export_conf['raw_encoding']
        self.raw_decoding = export_conf['raw_decoding']
        self._input_names = export_conf['raw_inputs']
        self._output_names = export_conf['onnx_inputs']  # preprocessor's output as onnx model's input

    @property
    def input_names(self):
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    @staticmethod
    def decode(params, encoding='json'):
        """decode the input of preprocess"""
        if encoding == 'json':
            return {k:json.loads(v.decode('utf8')) for k,v in params.items()}
        else:
            raise RuntimeError("Not support raw decoding method!")

    @staticmethod
    def encode(params, encoding='numpy'):
        """
        :param params: a dict of numpy.array
        """
        if encoding == 'list':
            return {k:v.tolist() for k,v in params.items()}
        elif encoding == 'numpy':
            return {k:v.tobytes() for k,v in params.items()}
        elif encoding == 'arrow':
            params = {k:v.tolist() for k,v in params.items()}
            payload_map = {}
            for name, value in params.items():
                t = pa.Tensor.from_numpy(np.array(value))
                sink = pa.BufferOutputStream()
                buffer = pa.ipc.write_tensor(t, sink)
                payload_map[name] = sink.getvalue().to_pybytes()
            return payload_map
        else:
            raise RuntimeError("Not support raw encoding method!")

    def predict(self, inputs):
        inputs = self.decode(inputs, self.raw_decoding)
        for name in self.input_names:
            if name in inputs:
                continue
            raise RuntimeError(f"The input {name} field not exists!")
        raise NotImplementedError

