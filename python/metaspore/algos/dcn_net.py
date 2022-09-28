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

from .layers import LRLayer, MLPLayer, CrossNet

class DCN(torch.nn.Module):
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=10,
                 deep_embedding_dim=10,
                 wide_column_name_path=None,
                 deep_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_combine_schema_path=None,
                 sparse_init_var=1e-2,
                 dnn_hidden_units=[1024, 1024, 512, 512, 512],
                 dnn_activations='ReLU',
                 use_bias=True,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 crossing_layers=3):
        super().__init__()
        self.use_wide = use_wide
        self.deep_embedding_dim = deep_embedding_dim
        self.crossing_layers = crossing_layers

        ## lr layer
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_dim, self.lr_sparse.feature_count)

        ## dnn sparse layers, initalization, and updater
        self.deep_column_name_path = deep_column_name_path
        self.deep_combine_schema_path = deep_combine_schema_path
        self.dnn_sparse = ms.EmbeddingSumConcat(self.deep_embedding_dim, self.deep_column_name_path, self.deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1 = ftrl_l1, \
                                                    l2 = ftrl_l2, \
                                                    alpha = ftrl_alpha, \
                                                    beta = ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.dnn_input_dim = int(self.dnn_sparse.feature_count*self.deep_embedding_dim)
        self.dnn_sparse_bn = ms.nn.Normalization(self.dnn_input_dim)

        ## crossing layers
        self.cross_net = CrossNet(self.dnn_input_dim, self.crossing_layers)

        ## dnn layers
        self.final_dim = self.dnn_input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0:
            self.final_dim += dnn_hidden_units[-1]
        self.dnn = MLPLayer(input_dim = self.dnn_input_dim,
                            output_dim = None,
                            hidden_units = dnn_hidden_units,
                            hidden_activations = dnn_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            batch_norm = batch_norm,
                            use_bias = use_bias) \
                   if dnn_hidden_units else None
        self.fc = torch.nn.Linear(self.final_dim, 1)

        ## logit
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.dnn_sparse(x)
        x_bn = self.dnn_sparse_bn(x)
        dnn_output = self.dnn(x_bn)
        x = x.contiguous().view(-1, self.dnn_sparse.feature_count, self.deep_embedding_dim)
        flat_feature = x.flatten(start_dim = 1)
        cross_out = self.cross_net(flat_feature)
        logit = torch.cat([cross_out, dnn_output], dim = -1)
        logit = self.fc(logit)

        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_logit = self.lr(lr_feature_map)
            logit += lr_logit

        prediction = self.final_activation(logit)
        return prediction
