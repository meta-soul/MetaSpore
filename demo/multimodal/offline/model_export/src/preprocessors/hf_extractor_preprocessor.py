import os
import json
import base64
from io import BytesIO

from PIL import Image
import numpy as np
import pyarrow as pa
from transformers import AutoFeatureExtractor


class HfExtractorPreprocessor(object):

    def __init__(self, config_dir):
        self.extractor = AutoFeatureExtractor.from_pretrained(config_dir)

        export_conf = json.load(open(os.path.join(config_dir, 'export_config.json'), 'r'))
        # we can get kwargs from `preprocessor_kwargs` 
        #extractor_kwargs = {} if 'preprocessor_kwargs' not in export_conf else export_conf['preprocessor_kwargs']
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
    def decode(params, encoding='bytes'):
        """decode the input of preprocess"""
        decoded = {}
        images_byte = params['images']
        if encoding == 'bytes':
            # the input image is encoded as bytes 
            # encode via `open('img.png', 'rb').read()`, a single image for now
            return {'images': [Image.open(BytesIO(images_byte))]}
        elif encoding == 'base64':
            # the input image is encoded as base64 string
            # encode via `data = open('img.png', 'rb').read(); base64.urlsafe_b64encode(data)`, a single image for now
            return {'images': [Image.open(BytesIO(base64.urlsafe_b64decode(images_byte)))]}
        elif encoding == 'json':
            # the input image is encoded as json list of base64 string
            # encode via `data = open('img.png', 'rb').read(); bs64 = base64.urlsafe_b64encode(data).decode('utf8'); json.dumps([bs64]).encode('utf8')
            images = json.loads(images_byte.decode('utf8'))
            if isinstance(images, str):
                images [images]
            return {'images': [Image.open(BytesIO(base64.urlsafe_b64decode(img.encode('utf8')))) for img in images]}
        else:
            raise RuntimeError("Not support raw decoding method!")

    @staticmethod
    def encode(params, encoding='numpy'):
        """
        :param params: a dict of numpy.array
        """
        if encoding == 'ndarray':
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
        images = inputs['images']  # a list of PIL.Image.Image
        # preprocess, the dtype must be same with the input of onnx model
        encoding = self.extractor(images, return_tensors="np")  # pixel_values of shape (batch_size, num_channels, height, width)
        # encode the processed data
        outputs = self.encode(encoding.data, self.raw_encoding)
        return outputs


if __name__ == '__main__':
    import requests

    #feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    #feature_extractor.save_pretrained('./export')
    #conf = {
    #    'raw_encoding': 'arrow',
    #    'raw_decoding': 'bytes',
    #    'raw_inputs': ['images'],
    #    'onnx_inputs': ['pixel_values']
    #}
    #with open('./export/export_config.json', 'w') as fout:
    #    json.dump(conf, fout)

    processor = HfExtractorPreprocessor('../../export/preprocess')
    processor.raw_encoding = 'ndarray'  # just for debug
    
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    data = requests.get(url, stream=True).raw.read()

    # input as bytes
    #inputs = {'images': data}
    #processor.raw_decoding = 'bytes'
    #res = processor.predict(inputs)
    #print(res['pixel_values'].shape, res['pixel_values'].dtype)

    # input as base64
    #inputs = {'images': base64.urlsafe_b64encode(data)}
    #processor.raw_decoding = 'base64'
    #res = processor.predict(inputs)
    #print(res['pixel_values'].shape, res['pixel_values'].dtype)

    # input as json
    inputs = {'images': json.dumps([base64.urlsafe_b64encode(data).decode('utf8')]).encode('utf8')}
    processor.raw_decoding = 'base64'
    res = processor.predict(inputs)
    print(res['pixel_values'].shape, res['pixel_values'].dtype)
