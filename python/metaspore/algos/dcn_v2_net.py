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

from .layers import LRLayer, MLPLayer, CrossNetMix, CrossNetV2

class DCN(torch.nn.Module):
    def __init__(self,
                 use_wide=True,
                 wide_embedding_dim=10,
                 deep_embedding_dim=10,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 deep_column_name_path=None,
                 deep_combine_schema_path=None,
                 sparse_init_var=1e-2,
                 dnn_activations='ReLU',
                 use_bias=True,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 stacked_dnn_hidden_units=[512, 512, 512],
                 parallel_dnn_hidden_units=[512, 512, 512],
                 model_structure='parallel',
                 use_low_rank_mixture=False,
                 low_rank=32,
                 num_experts=4,
                 num_crossing_layers=3,
                 **kwargs):
        super().__init__()
        self.use_wide = use_wide
        self.wide_embedding_dim = wide_embedding_dim
        self.deep_embedding_dim = deep_embedding_dim
        self.num_crossing_layers = num_crossing_layers
        self.model_structure = model_structure

        ## lr layer
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim, wide_column_name_path, wide_combine_schema_path)
            self.lr_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
            self.lr_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
            self.lr = LRLayer(wide_embedding_dim, self.lr_sparse.feature_count)

        ## nn sparse embedding
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim, deep_column_name_path, deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, l2=ftrl_l2, alpha = ftrl_alpha, beta=ftrl_beta)
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.dnn_input_dim = int(self.dnn_sparse.feature_count * self.deep_embedding_dim)

        ## crossing layers
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(self.dnn_input_dim, self.num_crossing_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(self.dnn_input_dim, self.num_crossing_layers)

        ## crossing structure
        if self.model_structure in ["stacked", "stacked_parallel"]:
            self.stacked_dnn = MLPLayer(input_dim=self.dnn_input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=stacked_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        final_activation=None,
                                        dropout_rates=net_dropout,
                                        batch_norm=batch_norm,
                                        use_bias=True)
            final_dim = stacked_dnn_hidden_units[-1]
        if self.model_structure in ["parallel", "stacked_parallel"]:
            self.parallel_dnn = MLPLayer(input_dim=self.dnn_input_dim,
                                        output_dim=None, # output hidden layer
                                        hidden_units=parallel_dnn_hidden_units,
                                        hidden_activations=dnn_activations,
                                        final_activation=None,
                                        dropout_rates=net_dropout,
                                        batch_norm=batch_norm,
                                        use_bias=True)
            final_dim = self.dnn_input_dim + parallel_dnn_hidden_units[-1]
        if self.model_structure == "stacked_parallel":
            final_dim = stacked_dnn_hidden_units[-1] + parallel_dnn_hidden_units[-1]
        if self.model_structure == "crossnet_only": # only CrossNet
            final_dim = self.dnn_input_dim

        ## model output
        self.fc = torch.nn.Linear(final_dim, 1)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, inputs):
        feature_map = self.dnn_sparse(inputs)
        cross_out = self.crossnet(feature_map)
        if self.model_structure == 'crossnet_only':
            final_out = cross_out
        elif self.model_structure == 'stacked':
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(feature_map)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
        elif self.model_structure == "stacked_parallel":
            final_out = torch.cat([self.stacked_dnn(cross_out), self.parallel_dnn(feature_map)], dim=-1)
        logit = self.fc(final_out)
        if self.use_wide:
            lr_feature_map = self.lr_sparse(inputs)
            lr_logit = self.lr(lr_feature_map)
            logit += lr_logit
        return self.final_activation(logit)

