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

class ESMM(torch.nn.Module):
    def __init__(self,
                 embedding_dim=18,
                 column_name_path=None,
                 combine_schema_path=None,
                 sparse_init_var=1e-2,
                 dnn_hidden_units=[360, 200, 80, 2],
                 dnn_hidden_activations="ReLU",
                 use_bias=True,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 **kwargs):
        super().__init__()
        # embedding dim for every field
        self.embedding_dim = embedding_dim
        self.embedding_dim = embedding_dim
        # embedding layer settings
        self.embedding_layer = ms.EmbeddingSumConcat(embedding_dim, column_name_path, combine_schema_path)
        self.embedding_layer.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.embedding_layer.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        # embdding layer output
        self.dnn_input_dim = self.embedding_layer.feature_count * embedding_dim
        # ctr & cvr nn net
        self.ctr_tower = MLPLayer(input_dim = self.dnn_input_dim,
                                  output_dim = 1,
                                  hidden_units = dnn_hidden_units,
                                  hidden_activations = dnn_hidden_activations,
                                  final_activation = None,
                                  dropout_rates = net_dropout,
                                  batch_norm = batch_norm,
                                  use_bias = use_bias)
        self.cvr_tower = MLPLayer(input_dim = self.dnn_input_dim,
                                  output_dim = 1,
                                  hidden_units = dnn_hidden_units,
                                  hidden_activations = dnn_hidden_activations,
                                  final_activation = None,
                                  dropout_rates = net_dropout,
                                  batch_norm = batch_norm,
                                  use_bias = use_bias)
        # ctr & cvr activation
        self.ctr_activation = torch.nn.Sigmoid()
        self.cvr_activation = torch.nn.Sigmoid()

    def forward(self, x):
        nn_feature_map = self.embedding_layer(x)
        ctr_tower_out = self.ctr_tower(nn_feature_map)
        cvr_tower_out = self.cvr_tower(nn_feature_map)
        ctr_logits = self.ctr_activation(ctr_tower_out)
        cvr_logits = self.cvr_activation(cvr_tower_out)
        return ctr_logits, cvr_logits


