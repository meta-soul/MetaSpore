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

import grpc

import metaspore_pb2
import metaspore_pb2_grpc

import pyarrow as pa

with grpc.insecure_channel('0.0.0.0:50051') as channel:
    stub = metaspore_pb2_grpc.PredictStub(channel)
    row = []
    values = [0.6558618,0.13005558,0.03510657,0.23048967,0.63329154,0.43201634,0.5795548,0.5384891,0.9612295,0.39274803]
    for i in range(10):
        row.append(pa.array([values[i]], type=pa.float32()))
    rb = pa.RecordBatch.from_arrays(row, [f'field_{i}' for i in range(10)])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, rb.schema) as writer:
        writer.write_batch(rb)
    payload_map = {"input": sink.getvalue().to_pybytes()}
    request = metaspore_pb2.PredictRequest(model_name="xgboost_model", payload=payload_map)
    reply = stub.Predict(request)
    for name in reply.payload:
        with pa.BufferReader(reply.payload[name]) as reader:
            tensor = pa.ipc.read_tensor(reader)
            print(f'Tensor: {tensor.to_numpy()}')