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

# torch script inference
model_pth = torch.jit.load('mnist_model/model.pth')

for data, target in validation_loader:
    data = data.to(device)
    print(f'torchscript data: {data.numpy()}')
    target = target.to(device)
    output = model_pth(data)
    print(f'Predict output from torch script: {output}')
    break

import onnxruntime

ort_session = onnxruntime.InferenceSession("mnist_model/model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

for data, target in validation_loader:
    data = data.to(device)
    print(f'onnx data: {data.numpy()}')
    target = target.to(device)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(f'Predict output from onnxruntime: {ort_outs}')
    break