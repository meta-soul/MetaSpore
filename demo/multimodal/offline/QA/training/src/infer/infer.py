import os
import sys
import json
import time
from collections import OrderedDict
from typing import List, Dict, Tuple, Any

import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer


class Inference(object):
    
    def __init__(self, onnx_path: str, onnx_inputs: List[str], onnx_outputs: List[str], device=None, logger=None):
        options = SessionOptions()
        if device is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # use cuda if avaiable
        elif device == 'cuda':
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self._ort_session = InferenceSession(onnx_path, options, providers=providers)
        self._input_names = onnx_inputs[:]
        self._output_names = onnx_outputs[:]
        self._logger = logger
        self._perf_info = {}

    @property
    def input_names(self):
        return self._input_names[:]

    @property
    def output_names(self):
        return self._output_names[:]

    @property
    def perf(self):
        return self._perf_info.copy()

    def preprocess(self, inputs: Tuple[Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, features: Dict[str, Any]) -> Dict[str, np.ndarray]:
        ins = {name:features[name] for name in self._input_names}  # inputs should be np.ndarray
        outs = self._ort_session.run(self._output_names, ins)
        outputs = OrderedDict()
        for name, value in zip(self._output_names, outs):
            outputs[name] = value
        return outputs

    def postprocess(self, features: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray]) -> Any:
        return outputs

    def encode(self, *inputs):
        return self.__call__(*inputs)

    def __call__(self, *inputs):
        start_time = time.time()
        features = self.preprocess(*inputs)
        pre_time = time.time()-start_time
        start_time = time.time()
        assert isinstance(features, dict), "preprocess must return a dict!"
        outputs = self.predict(features)
        pred_time = time.time()-start_time
        start_time = time.time()
        results = self.postprocess(features, outputs)
        post_time = time.time()-start_time
        self._perf_info['pre_process_latency'] = pre_time*1000
        self._perf_info['prediction_latency'] = pred_time*1000
        self._perf_info['post_process_latency'] = post_time*1000
        if self._logger is not None:
            self._logger.info("Pre-process time: {}ms".format(pre_time*1000))
            self._logger.info("Prediction time: {}ms".format(pred_time*1000))
            self._logger.info("Post-process time: {}ms".format(post_time*1000))
        return results


class TextEncoderInference(Inference):
    
    def __init__(self, onnx_path, device=None, logger=None, max_len=256, do_lower_case=False):
        tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(onnx_path))
        onnx_inputs = tokenizer.model_input_names
        onnx_outputs = ['sentence_embedding']
        super(TextEncoderInference, self).__init__(onnx_path, onnx_inputs, onnx_outputs, device=device, logger=logger)
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._do_lower_case = do_lower_case
        self._logger = logger

    @property
    def tokenizer(self):
        return self._tokenizer

    @classmethod
    def create_from_config(cls, config_file, **kwargs):
        with open(config_file, 'r', encoding='utf8') as fin:
            cfg = json.load(fin)
        max_len = 256 if 'max_seq_len' not in cfg else cfg['max_seq_len']
        do_lower_case = False if 'do_lower_case' not in cfg else cfg['do_lower_case']
        return cls(cfg['onnx_path'], max_len=max_len, do_lower_case=do_lower_case, **kwargs)

    def preprocess(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if self._do_lower_case:
            texts = [s.lower() for s in texts]
        encoding = self._tokenizer(texts, padding=True, truncation=True, return_tensors="np", max_length=self._max_len)
        return encoding.data

    def postprocess(self, features, outputs):
        """Return a numpy array of batch_size x emb_dim"""
        return outputs['sentence_embedding']

if __name__ == '__main__':
    config_file = sys.argv[1]
    encoder = TextEncoderInference.create_from_config(config_file)
    text = 'hello world!'
    print(f'text: {text}')
    print('embedding: {}'.format(encoder(text)))
    print('perf: {}'.format(encoder.perf))

