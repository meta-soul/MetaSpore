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

from .layers import MLPLayer, MultiHeadSelfAttention

class AutoInt(torch.nn.Module):
    def __init__(self,
                 use_wide=False,
                 use_residual=True,
                 use_scale=False,
                 embedding_dim=10,
                 column_name_path=None,
                 combine_schema_path=None,
                 sparse_init_var=1e-2,
                 dnn_hidden_units=[1024, 1024, 512, 256, 128],
                 dnn_hidden_activations='ReLU',
                 use_bias=True,
                 batch_norm=False,
                 net_dropout=None,
                 net_regularizer=None,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0,
                 attention_layers=1,
                 num_heads=1,
                 attention_dim=8,
                 layer_norm=False,
                 **kwargs):
        super().__init__()
        self.use_wide = use_wide
        self.use_residual = use_residual
        self.embedding_dim = embedding_dim
        self.column_name_path = column_name_path
        self.combine_schema_path = combine_schema_path
        self.attention_layers = attention_layers
        self.num_heads = num_heads
        self.attention_dim=attention_dim
        self.net_dropout = net_dropout
        self.batch_norm = batch_norm
        self.layer_norm=layer_norm
        self.use_scale = use_scale
        self.net_regularizer= net_regularizer

        self.sparse = ms.EmbeddingSumConcat(self.embedding_dim, self.column_name_path, self.combine_schema_path)
        self.sparse.updater = ms.FTRLTensorUpdater(l1=ftrl_l1, \
                                                    l2=ftrl_l2, \
                                                    alpha=ftrl_alpha, \
                                                    beta=ftrl_beta)
        self.sparse.initializer = ms.NormalTensorInitializer(var=sparse_init_var)
        self.input_dim = int(self.sparse.feature_count*self.embedding_dim)
        self.sparse_bn = ms.nn.Normalization(self.input_dim)

        self.lr = torch.nn.Linear(self.input_dim, 1, bias=use_bias) if use_wide else None

        self.dnn = MLPLayer(input_dim = self.input_dim,
                            output_dim = 1,
                            hidden_units = dnn_hidden_units,
                            hidden_activations = dnn_hidden_activations,
                            final_activation = None,
                            dropout_rates = net_dropout,
                            batch_norm = batch_norm,
                            use_bias = use_bias) \
                   if dnn_hidden_units else None

        self.self_attention=torch.nn.Sequential(
            *[MultiHeadSelfAttention(input_dim=self.input_dim if i == 0 else self.num_heads * self.attention_dim,
                                    dim_per_head=self.attention_dim,
                                    num_heads=self.num_heads,
                                    dropout_rate=self.net_dropout,
                                    use_residual=self.use_residual,
                                    layer_norm=self.layer_norm)
                 for i in range(self.attention_layers)])
        self.fc = torch.nn.Linear(self.attention_dim * self.num_heads, 1)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self,x):
        feature_map = self.sparse(x)
        feature_map_bn = self.sparse_bn(feature_map)
        dnn_out = self.dnn(feature_map_bn.flatten(start_dim=1))
        attention_out = self.self_attention(feature_map)
        attention_out = torch.flatten(attention_out, start_dim=1)
        attention_out = self.fc(attention_out)
        y_pred = attention_out + dnn_out
        if self.lr is not None:
            y_pred = y_pred + self.lr(feature_map_bn)
        y_pred = self.final_activation(y_pred)
        return y_pred
