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

import pyarrow as pa
import numpy as np
from transformers import AutoTokenizer

class HfTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=256, do_lower_case=False, *args, **kwargs):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.do_lower_case = do_lower_case

    @staticmethod
    def encode(params):
        """encode the output of preprocess"""
        # enccode by json
        #return {k:json.dumps(v, ensure_ascii=False).encode('utf8') for k,v in params.items()}
        # encode by pyarrow
        payload_map = {}
        for name, value in params.items():
            t = pa.Tensor.from_numpy(np.array(value))
            sink = pa.BufferOutputStream()
            buffer = pa.ipc.write_tensor(t, sink)
            payload_map[name] = sink.getvalue().to_pybytes()
        return payload_map

    @staticmethod
    def decode(params):
        """decode the input of preprocess"""
        return {k:json.loads(v.decode('utf8')) for k,v in params.items()}

    @classmethod
    def load(cls, model_name_or_config_dir):
        onnx_conf = os.path.join(model_name_or_config_dir, 'onnx_config.json')
        if os.path.isfile(onnx_conf):
            kwargs = json.load(open(onnx_conf, 'r'))['tokenizer']
            model_dir = os.path.join(model_name_or_config_dir, 'pretrained')
            return cls(AutoTokenizer.from_pretrained(model_dir), **kwargs)
        return cls(AutoTokenizer.from_pretrained(model_name_or_config_dir))

    def predict(self, inputs):
        inputs = self.decode(inputs)
        texts = inputs['texts']
        if isinstance(texts, str):
            texts = [texts]
        if self.do_lower_case:
            texts = [s.lower() for s in texts]
        encoding = self.tokenizer(texts, add_special_tokens=True, 
            padding=True, truncation=True, return_tensors="np", max_length=self.max_seq_len)
        outputs = encoding.data
        outputs = {k:v.tolist() for k,v in outputs.items()}
        return self.encode(outputs)


if __name__ == '__main__':
    tokenizer = HfTokenizer.load('bert-base-chinese')
    outputs = tokenizer.predict('a b c')
    print(outputs)
    print(tokenizer.decode(outputs))
