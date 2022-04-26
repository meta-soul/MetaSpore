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

from __future__ import print_function

import sys
import json
import logging

import grpc
from hf_preprocessor import hf_preprocessor_pb2
from hf_preprocessor import hf_preprocessor_pb2_grpc
import pyarrow as pa


def run_tokenize(model_key, text, port=60051):
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        stub = hf_preprocessor_pb2_grpc.HfPreprocessorStub(channel)
        # req encode
        payload = {'texts': json.dumps([text]).encode('utf8')}
        req = hf_preprocessor_pb2.HfTokenizerRequest(model_name=model_key, payload=payload)
        response = stub.HfTokenizer(req)
        # res decode via json
        #payload = {k:json.loads(v.decode('utf8')) for k,v in response.payload.items()}
        # res decode via pyarrow
        payload = {}
        for name in response.payload:
            with pa.BufferReader(response.payload[name]) as reader:
                payload[name] = pa.ipc.read_tensor(reader).to_numpy().tolist()
    print("Client received: payload={}, extras={}".format(payload, response.extras))


def run_push(model_key, model_url, port=60051):
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        stub = hf_preprocessor_pb2_grpc.HfPreprocessorStub(channel)
        req = hf_preprocessor_pb2.HfTokenizerPushRequest(model_name=model_key, model_url=model_url)
        response = stub.HfTokenizerPush(req)
    print("Client received: status={}, message={}".format(response.status, response.msg))


if __name__ == '__main__':
    logging.basicConfig()
    action = sys.argv[1]
    if action == 'push':
        key, url = sys.argv[2], sys.argv[3]
        run_push(key, url)
    elif action == 'tokenize':
        key, text = sys.argv[2], sys.argv[3]
        run_tokenize(key, text)
    else:
        print('invalid action!')
