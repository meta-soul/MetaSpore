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
from pyarrow.csv import read_csv, ReadOptions, ParseOptions, ConvertOptions

column_names = []
column_types = {}

with open('schema/wdl/column_name_demo.txt', 'r') as f:
    for line in f:
        line = line.rstrip('\n')
        column_names.append(line)
        column_types[line] = pa.string()


def read_csv_as_rb(path):
    return read_csv(path, ReadOptions(use_threads=False, block_size=1024 * 1024, column_names=column_names),
                    ParseOptions(delimiter='\t'),
                    ConvertOptions(column_types=column_types))


with grpc.insecure_channel('0.0.0.0:50051') as channel:
    stub = metaspore_pb2_grpc.PredictStub(channel)

    tb = read_csv_as_rb('data/day_0_0.001_test_head10.csv')
    rbs = tb.to_batches(1024 * 1024)
    print(len(rbs))
    rb = rbs[0]
    print(rb.to_pandas())


    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, rb.schema) as writer:
        writer.write_batch(rb)
    bytes = sink.getvalue().to_pybytes()
    payload_map = {"_sparse": bytes, "lr_layer": bytes}
    request = metaspore_pb2.PredictRequest(
        model_name="wide_and_deep", payload=payload_map)
    reply = stub.Predict(request)
    for name in reply.payload:
        print(f'reply tensor {name}, buffer len: {len(reply.payload[name])}')
        print(f'payload hex: {reply.payload[name].hex()}')
        with pa.BufferReader(reply.payload[name]) as reader:
            tensor = pa.ipc.read_tensor(reader)
            print(f'Tensor: {tensor.to_numpy()}')
