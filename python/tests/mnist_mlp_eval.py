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

# Modified from https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

validation_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                batch_size=10, 
                                                shuffle=False)

import onnxruntime

ort_session = onnxruntime.InferenceSession("output/model_export/mnist/model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_pth = torch.jit.load('output/model_export/mnist/model.pth')

for data, target in validation_loader:
    data = data.to(device)
    numpy_input = to_numpy(data)
    print(f'torchscript data: {numpy_input}')
    target = target.to(device)
    # torch script inference
    output = model_pth(data)
    print(f'Predict output from torch script: {output}')

    # onnx runtime inference
    ort_inputs = {ort_session.get_inputs()[0].name: numpy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f'Predict output from python onnxruntime: {ort_outs}')

    # metaspore serving inference
    import pyarrow as pa
    t = pa.Tensor.from_numpy(numpy_input)
    sink = pa.BufferOutputStream()
    buffer = pa.ipc.write_tensor(t, sink)
    payload_map = {"input": sink.getvalue().to_pybytes()}

    import grpc
    import metaspore_pb2
    import metaspore_pb2_grpc

    with grpc.insecure_channel('0.0.0.0:50051') as channel:
        stub = metaspore_pb2_grpc.PredictStub(channel)
        request = metaspore_pb2.PredictRequest(model_name="mnist", payload=payload_map)
        reply = stub.Predict(request)
        for name in reply.payload:
            with pa.BufferReader(reply.payload[name]) as reader:
                tensor = pa.ipc.read_tensor(reader)
                print(f'Predict output from metaspore serving with onnxruntime, name {name}: {tensor.to_numpy()}')
    break