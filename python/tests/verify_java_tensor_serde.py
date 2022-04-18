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

import metaspore_pb2
import pyarrow as pa

with open('arrow_tensor_java_ser.bin', 'rb') as f:
    message = metaspore_pb2.PredictRequest()
    message.ParseFromString(f.read())
    for name in message.payload:
        print(f'{name}')
        with pa.BufferReader(message.payload[name]) as reader:
            tensor = pa.ipc.read_tensor(reader).to_numpy()
            print(f'Tensor: {tensor}, shape: {tensor.shape}')