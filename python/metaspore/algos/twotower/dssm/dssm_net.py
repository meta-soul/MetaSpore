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
import torch.nn.functional as F

from ...layers import MLPLayer

class SimilarityModule(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x, y):
        z = torch.sum(x * y, dim=1).reshape(-1, 1)
        s = torch.sigmoid(z/self.tau)
        return s

class UserModule(torch.nn.Module):
    def __init__(self,
                 column_name_path,
                 combine_schema_path,
                 embedding_dim,
                 sparse_init_var=1e-2,
                 dnn_hidden_units=[1024, 512, 256],
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
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        ## sparse layers
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.sparse.output_batchsize1_if_only_level0 = True
        ## sparse normalization
        self.sparse_output_dim = self.sparse.feature_count * self.embedding_dim
        self.sparse_embedding_bn = ms.nn.Normalization(self.sparse_output_dim, momentum=0.01, eps=1e-5)
        ## dense layers
        self.dense = MLPLayer(input_dim = self.sparse_output_dim,
                              output_dim = None,
                              hidden_units = dnn_hidden_units,
                              hidden_activations = dnn_hidden_activations,
                              final_activation = None,
                              dropout_rates = net_dropout,
                              batch_norm = batch_norm,
                              use_bias = use_bias)

    def forward(self, x):
        x = self.sparse(x)
        x = self.sparse_embedding_bn(x)
        x = self.dense(x)
        return x

class ItemModule(torch.nn.Module):
    def __init__(self,
                 column_name_path,
                 combine_schema_path,
                 embedding_dim,
                 sparse_init_var=1e-2,
                 dnn_hidden_units=[1024, 512, 256],
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
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        ## sparse layers
        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        ## sparse normalization
        self.sparse_output_dim = self.sparse.feature_count * self.embedding_dim
        self.sparse_embedding_bn = ms.nn.Normalization(self.sparse_output_dim, momentum=0.01, eps=1e-5)
        ## dense layers
        self.dense = MLPLayer(input_dim = self.sparse_output_dim,
                              output_dim = None,
                              hidden_units = dnn_hidden_units,
                              hidden_activations = dnn_hidden_activations,
                              final_activation = None,
                              dropout_rates = net_dropout,
                              batch_norm = batch_norm,
                              use_bias = use_bias)

    def forward(self, x):
        x = self.sparse(x)
        x = self.sparse_embedding_bn(x)
        x = self.dense(x)
        return x
