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

import torch
import metaspore as ms

# Logistic regression layer
class LRLayer(torch.nn.Module):
    def __init__(self,
                embedding_size,
                feature_dim):
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_dim = feature_dim
    
    def forward(self, inputs):
        out = torch.sum(inputs, dim=1, keepdim=True)
        return out


# Fully connected layers
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/deep.py
class MLPLayer(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 final_activation=None, 
                 dropout_rates=[], 
                 batch_norm=False, 
                 use_bias=True,
                 input_norm=False):
        super().__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)        
        hidden_activations = [self.set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        
        if input_norm:
            dense_layers.append(ms.nn.Normalization(input_dim))
        
        ## batchnorm + linear + activation + dropout...
        for idx in range(len(hidden_units) - 1):
            if batch_norm:
                dense_layers.append(torch.nn.BatchNorm1d(hidden_units[idx]))
            dense_layers.append(torch.nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias = use_bias))
            if hidden_activations[idx] is not None \
                and (idx < len(hidden_units) - 2 or output_dim is not None):
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] is not None and dropout_rates[idx] > 0 \
                and (idx < len(hidden_units) - 2 or output_dim is not None):
                dense_layers.append(torch.nn.Dropout(p=dropout_rates[idx]))
        ## final layer
        if output_dim is not None:
            dense_layers.append(torch.nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        ## final activation
        if final_activation is not None:
            dense_layers.append(self.set_activation(final_activation))
        ## all in one
        self.dnn = torch.nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)
    
    @staticmethod
    def set_activation(activation):
        if isinstance(activation, str):
            if activation.lower() == "relu":
                return torch.nn.ReLU()
            elif activation.lower() == "sigmoid":
                return torch.nn.Sigmoid()
            elif activation.lower() == "tanh":
                return torch.nn.Tanh()
            else:
                return torch.nn.ReLU() ## defalut relu
        else:
            return torch.nn.ReLU() ## defalut relu


# Factorization Machine layer
class FMLayer(torch.nn.Module):
    def __init__(self,
                feature_count,
                embedding_dim):
        super().__init__()
        self._feature_count = feature_count
        self._embedding_dim = embedding_dim
    
    def forward(self, inputs):
        inputs = inputs.reshape(-1, self._feature_count, self._embedding_dim)
        square_of_sum = torch.pow(torch.sum(inputs, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(inputs * inputs, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


# Cross Net layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/interaction.py
class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.cross_net = torch.nn.ModuleList(CrossInteractionLayer(input_dim) 
                                       for _ in range(self.num_layers))
    
    def forward(self, X_0):
        X_i = X_0
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0,X_i)
        return X_i   

# Cross Interaction Layer
# This code is adapted from github repository:  https://github.com/xue-pai/FuxiCTR/blob/main/fuxictr/pytorch/layers/interaction.py
class CrossInteractionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.weight = torch.nn.Linear(input_dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out