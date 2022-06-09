import os
import json

import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer

# for arrowtensor encoder

def json_encode(params):
    return {k:json.dumps(v, ensure_ascii=False).encode('utf8') for k,v in params.items()}

def json_decode(params):
    return {k:json.loads(v.decode('utf8')) for k,v in params.items()}


class HfTokenizerPreprocessor(object):

    def __init__(self, config_dir):
        # assume the data and files in the `config_dir`
        self.tokenizer = AutoTokenizer.from_pretrained(config_dir)
        # load from export config
        export_conf = json.load(open(os.path.join(config_dir, 'export_config.json'), 'r'))
        tokenizer_kwargs = {} if 'preprocessor_kwargs' not in export_conf else export_conf['preprocessor_kwargs']
        self.max_seq_len = tokenizer_kwargs.get('max_seq_len', 256)
        self.do_lower_case = tokenizer_kwargs.get('do_lower_case', False)
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
        texts_byte = params['texts']
        if encoding == 'json':
            texts = json.loads(texts_byte.decode('utf8'))
            texts = [texts] if isinstance(texts, str) else texts
            return {'texts': texts}
        else:
            raise RuntimeError("Not support raw decoding method!")

    @staticmethod
    def encode(params, encoding='numpy'):
        """
        :param params: a dict of numpy.array
        """
        if encoding == "ndarray":
            return {k:v for k,v in params.items()}
        elif encoding == 'list':
            return {k:v.tolist() for k,v in params.items()}
        elif encoding == 'numpy':
            return {k:v.tobytes() for k,v in params.items()}
        elif encoding == 'arrow':
            payload_map = {}
            for name, value in params.items():
                t = pa.Tensor.from_numpy(value)
                sink = pa.BufferOutputStream()
                buffer = pa.ipc.write_tensor(t, sink)
                payload_map[name] = sink.getvalue().to_pybytes()
            return payload_map
        else:
            raise RuntimeError("Not support raw encoding method!")

    def predict(self, inputs):
        for name in self.input_names:
            if name in inputs:
                continue
            raise RuntimeError(f"The input {name} field not exists!")
        # decode the raw data
        inputs = self.decode(inputs, self.raw_decoding)
        texts = inputs['texts']
        texts_pair = None
        if len(texts) > 0 and isinstance(texts[0], list):
            texts, texts_pair = zip(*texts)
            texts, texts_pair = list(texts), list(texts_pair)
        if self.do_lower_case:
            texts = [s.lower() for s in texts]
            texts_pair = [s.lower() for s in texts_pair] if texts_pair is not None else None
        # preprocess, the dtype must be same with the input of onnx model
        encoding = self.tokenizer(texts, text_pair=texts_pair, add_special_tokens=True, 
            padding=True, truncation=True, return_tensors="np", max_length=self.max_seq_len)
        # encode the processed data
        outputs = self.encode(encoding.data, self.raw_encoding)
        return outputs


if __name__ == '__main__':
    tokenizer = HfTokenizerPreprocessor('../../export/preprocess')
    tokenizer.raw_encoding = 'list'
    params1 = json_encode({'texts': 'a b c'})
    params2 = json_encode({'texts': ['a b c', ' a c']})
    params2 = json_encode({'texts': [['a b c', ' a c'], ['xyz', 'mpq']]})
    outputs = tokenizer.predict(params1)
    print(outputs)
    outputs = tokenizer.predict(params2)
    print(outputs)
