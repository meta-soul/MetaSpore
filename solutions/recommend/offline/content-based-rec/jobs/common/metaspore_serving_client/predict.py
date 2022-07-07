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

import json

import grpc
import pyarrow as pa
import requests

from . import metaspore_pb2
from . import metaspore_pb2_grpc

def make_payload(items, item_type='text'):
    if item_type == 'text':
        return {'texts': json.dumps(items, separators=(',', ':')).encode('utf-8')}
    elif item_type == 'image':
        return {'images': items}  # image must be bytes
    else:
        return {}

def request(host, port, model_name, payload):
    res = {}
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = metaspore_pb2_grpc.PredictStub(channel)
        req = metaspore_pb2.PredictRequest(model_name=model_name, payload=payload)
        reply = stub.Predict(req)
        for name in reply.payload:
            with pa.BufferReader(reply.payload[name]) as reader:
                tensor = pa.ipc.read_tensor(reader)
                res[name] = tensor.to_numpy()
    return res
