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

class HRMSimilarityModule(torch.nn.Module):
    def __init__(self, tau=1.0, net_dropout=0.0):
        super().__init__()
        self.tau = tau
        self.user_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None

    def forward(self, x, y):
        z = torch.sum(x * y, dim=1).reshape(-1, 1)
        return torch.sigmoid(z/self.tau)

class HRMUserModule(torch.nn.Module):
    def __init__(self,
                 column_name_path,
                 user_combine_schema_path,
                 seq_combine_schema_path,
                 embedding_dim,
                 pooling_type_layer_1='sum', # in ['mean', 'sum', 'max']
                 pooling_type_layer_2='sum', # in ['mean', 'sum', 'max']
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 sparse_init_var=1e-4,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        # pooling type checks
        if pooling_type_layer_1 not in ('mean', 'sum', 'max'):
            raise ValueError(f"Pooling type must be one of: 'mean', 'sum', 'max'; {pooling_type_layer_1!r} is invalid")
        # user embedding and pooling layer1
        self.user_sparse = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            user_combine_schema_path,
            embedding_bag_mode=pooling_type_layer_1
        )
        self.user_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2,
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self.user_sparse.initializer = ms.NormalTensorInitializer(var = sparse_init_var)
        self.user_sparse.output_batchsize1_if_only_level0 = True
        # behaviour sequence embedding and pooling layer1
        self.seq_sparse = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            seq_combine_schema_path,
            embedding_bag_mode=pooling_type_layer_1
        )
        self.seq_sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2,
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self.seq_sparse.initializer = ms.NormalTensorInitializer(var = sparse_init_var)
        self.seq_sparse.output_batchsize1_if_only_level0 = True

        # pooling layer2
        if pooling_type_layer_2 == 'mean':
            self.pooling_layer2 = torch.mean
        elif pooling_type_layer_2 == 'sum':
            self.pooling_layer2 = torch.sum
        elif pooling_type_layer_2 == 'max':
            self.pooling_layer2 = torch.amax

        # dropout
        self.embedding_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None

    def forward(self, x):
        hybrid_user_embedding = torch.cat([
            self.user_sparse(x).unsqueeze(dim=1),
            self. seq_sparse(x).unsqueeze(dim=1)],
            dim=1
        )
        if self.embedding_dropout:
            hybrid_user_embedding = self.embedding_dropout(hybrid_user_embedding)
        hybrid_user_embedding = self.pooling_layer2(hybrid_user_embedding, dim=1)
        return F.normalize(hybrid_user_embedding)

class HRMItemModule(torch.nn.Module):
    def __init__(self,
                 column_name_path,
                 combine_schema_path,
                 embedding_dim,
                 pooling_type_layer_1='sum', # in ['mean', 'sum', 'max']
                 net_dropout=0,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 sparse_init_var=1e-4,
                 ftrl_l1=1.0,
                 ftrl_l2=120.0,
                 ftrl_alpha=0.5,
                 ftrl_beta=1.0):
        super().__init__()
        # pooling type checks
        if pooling_type_layer_1 not in ('mean', 'sum', 'max'):
            raise ValueError(f"Pooling type must be one of: 'mean', 'sum', 'max'; {pooling_type_layer_1!r} is invalid")
         # user embedding and pooling layer1
        self.sparse = ms.EmbeddingSumConcat(
            embedding_dim,
            column_name_path,
            combine_schema_path,
            embedding_bag_mode=pooling_type_layer_1
        )
        self.sparse.updater = ms.FTRLTensorUpdater(
            l1=ftrl_l1,
            l2=ftrl_l2,
            alpha=ftrl_alpha,
            beta=ftrl_beta
        )
        self.sparse.initializer = ms.NormalTensorInitializer(var = sparse_init_var)
        # dropout
        self.embedding_dropout = torch.nn.Dropout(net_dropout) if net_dropout > 0 else None

    def forward(self, x):
        x = self.sparse(x)
        if self.embedding_dropout:
            x = self.embedding_dropout(x)
        x = F.normalize(x)
        return x
