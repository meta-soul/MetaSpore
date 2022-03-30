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

class WideDeep(torch.nn.Module):
    def __init__(self,
                use_wide=True,
                wide_embedding_dim=10,
                deep_embedding_dim=10,
                wide_column_name_path=None,
                wide_combine_schema_path=None,
                deep_column_name_path=None,
                deep_combine_schema_path=None,
                dnn_hidden_units=[1024,512,256,128,1],
                dnn_hidden_activations="ReLU", 
                dropout=0, 
                batch_norm=False,
                **kwargs):
        super().__init__()
        self.use_wide = use_wide
        if self.use_wide:
            self.lr_sparse = ms.EmbeddingSumConcat(wide_embedding_dim,
                                                wide_column_name_path,
                                                wide_combine_schema_path)
        self.dnn_sparse = ms.EmbeddingSumConcat(deep_embedding_dim,
                                           deep_column_name_path,
                                           deep_combine_schema_path)
        self.dnn_sparse.updater = ms.FTRLTensorUpdater()
        self.dnn_sparse.initializer = ms.NormalTensorInitializer(var=0.01)
        self.dnn = MLPLayer(dnn_hidden_units, deep_embedding_dim, self.dnn_sparse.feature_count)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.final_activation = torch.nn.Sigmoid()

    def forward(self, x):
        if self.use_wide:      
            wide_out = self.lr_sparse(x)
            wide_out = torch.sum(wide_out, dim=1, keepdim=True)
        dnn_out = self.dnn_sparse(x)
        dnn_out = self.dnn(dnn_out)
        final_out = torch.add(wide_out, dnn_out) if self.use_wide else dnn_out + self.bias
        return self.final_activation(final_out)

class MLPLayer(torch.nn.Module):
    def __init__(self,
                 hidden_lay_unit,
                 embedding_size,
                 feature_dim):
        super().__init__()
        dense_layers=[]
        dnn_linear_num=len(hidden_lay_unit)
        dense_layers.append(ms.nn.Normalization(feature_dim*embedding_size))
        dense_layers.append(torch.nn.Linear(feature_dim*embedding_size, hidden_lay_unit[0]))
        dense_layers.append(torch.nn.ReLU())
        for i in range(dnn_linear_num - 2):
            dense_layers.append(torch.nn.Linear(hidden_lay_unit[i], hidden_lay_unit[i + 1]))
            dense_layers.append(torch.nn.ReLU())
        dense_layers.append(torch.nn.Linear(hidden_lay_unit[-2], hidden_lay_unit[-1]))
        self.dnn = torch.nn.Sequential(*dense_layers)

    def forward(self,inputs):
        return self.dnn(inputs)
