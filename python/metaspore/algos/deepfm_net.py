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

from .layers import LRLayer, MLPLayer, FMLayer

class DeepFM(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                use_dnn=True,
                use_fm=True,
                wide_embedding_dim=16,
                deep_embedding_dim=16,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
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
                **kwargs
                ):
        super().__init__()
        self.use_wide = use_wide
        self.use_dnn = use_dnn
        self.use_fm = use_fm
        self.wide_embedding_dim = wide_embedding_dim
        self.deep_embedding_dim = deep_embedding_dim
        ## lr layer
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_dim, self.lr_sparse.feature_count)
        ## nn layers
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, deep_column_name_path, deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        if self.use_dnn:
            self.dnn = MLPLayer(input_dim = self.dnn_sparse.feature_count * deep_embedding_dim,
                                output_dim = 1,
                                hidden_units = dnn_hidden_units,
                                hidden_activations = dnn_hidden_activations,
                                final_activation = None,
                                dropout_rates = net_dropout,
                                batch_norm = batch_norm,
                                use_bias = use_bias)
        self.fm = FMLayer(self.dnn_sparse.feature_count, self.deep_embedding_dim)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        nn_feature_map = self.dnn_sparse(x)
        fm_out = self.fm(nn_feature_map)
        if self.use_wide:
            lr_feature_map = self.lr_sparse(x)
            lr_out = self.lr(lr_feature_map)
            fm_out += lr_out
        if self.use_dnn:
            dnn_out = self.dnn(nn_feature_map)
            fm_out += dnn_out
        return self.final_activation(fm_out)

