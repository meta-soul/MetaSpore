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

from .layers import LRLayer, MLPLayer, InnerProductLayer, OuterProductLayer

class PNN(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                wide_embedding_dim=16,
                deep_embedding_dim=16,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                sparse_init_var=1e-2,
                dnn_hidden_units=[512, 512, 512],
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
                product_type="inner",
                **kwargs):
        super().__init__()
        self._use_wide = use_wide
        self._wide_embedding_dim = wide_embedding_dim
        self._deep_embedding_dim = deep_embedding_dim
        ## lr layer
        if self._use_wide:
            self._lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self._lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
            self._lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self._lr = LRLayer(wide_embedding_dim, self._lr_sparse.feature_count)
        ## nn sparse embedding
        self._dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, deep_column_name_path, deep_combine_schema_path)
        self._dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self._dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        ## pnn layer
        if product_type == 'inner':
            self._pnn = InnerProductLayer(self._dnn_sparse.feature_count, deep_embedding_dim)
            dnn_input_dim = int(self._dnn_sparse.feature_count * (self._dnn_sparse.feature_count - 1)/2 \
                            + self._dnn_sparse.feature_count * deep_embedding_dim)
        elif product_type == 'outer':
            self._pnn = OuterProductLayer(self._dnn_sparse.feature_count, deep_embedding_dim)
            dnn_input_dim = int(deep_embedding_dim * (deep_embedding_dim - 1)/2 * self._dnn_sparse.feature_count
                            + self._dnn_sparse.feature_count * deep_embedding_dim)
        else:
            raise ValueError('product_type should be inner or outer')
        ## dnn layer
        self._dnn = MLPLayer(input_dim = dnn_input_dim,
                            output_dim = 1,
                            hidden_units = dnn_hidden_units,
                            hidden_activations = dnn_hidden_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            batch_norm = batch_norm,
                            use_bias = use_bias)
        ## final output
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        feature_map = self._dnn_sparse(inputs)
        inner_product_vec = self._pnn(feature_map)
        dense_input = torch.cat([feature_map.flatten(start_dim=1), inner_product_vec], dim=1)
        logit = self._dnn(dense_input)
        if self._use_wide:
            lr_feature_map = self._lr_sparse(inputs)
            lr_logit = self._lr(lr_feature_map)
            logit += lr_logit
        return self.final_activation(logit)
