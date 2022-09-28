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
import sys

from ...layers import MLPLayer

class MMoE(torch.nn.Module):
    def __init__(self,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 expert_numb=2,
                 task_numb=2,
                 expert_hidden_units=[],
                 expert_out_dim=10,
                 gate_hidden_units=[],
                 tower_hidden_units=[],
                 dnn_activations='ReLU',
                 use_bias=True,
                 input_norm=False,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 sparse_init_var=1e-2,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        self.expert_numb = expert_numb
        self.task_numb = task_numb
        self.expert_out_dim = expert_out_dim

        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, \
                                                    l2 = ftrl_l2, \
                                                    alpha = ftrl_alpha, \
                                                    beta = ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count*self.embedding_dim)

        self.experts = []
        for i in range(0, self.expert_numb):
            mlp = MLPLayer(input_dim = self.input_dim,
                            output_dim = self.expert_out_dim,
                            hidden_units = expert_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            input_norm=input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.experts.append(mlp)

        self.gates = []
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim = self.input_dim,
                            output_dim = self.expert_numb,
                            hidden_units = gate_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            input_norm=input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.gates.append(mlp)
        self.gate_softmax = torch.nn.Softmax(dim=1)

        self.towers = []
        for i in range(0, self.task_numb):
            mlp = MLPLayer(input_dim = self.expert_out_dim,
                            output_dim = 1,
                            hidden_units = tower_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = 'Sigmoid',
                            dropout_rates = net_dropout,
                            input_norm=input_norm,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
            self.towers.append(mlp)

    def forward(self, x):
        x = self.sparse(x)
        expert_outputs = []
        for i in range(0, self.expert_numb):
            expert_out = self.experts[i](x)
            expert_outputs.append(expert_out)
        expert_cat = torch.cat(expert_outputs, dim=1)
        expert_cat = expert_cat.reshape(-1, self.expert_numb, self.expert_out_dim)

        predictions = []
        for i in range(0, self.task_numb):
            gate_out = self.gates[i](x)
            gate_out = self.gate_softmax(gate_out)
            gate_out = gate_out.reshape(-1, self.expert_numb, 1)
            tower_input = torch.mul(expert_cat, gate_out)
            tower_input = torch.sum(tower_input, 1)
            tower_out = self.towers[i](tower_input)
            predictions.append(tower_out)
        prediction = torch.cat(predictions, dim=1)
        return prediction
