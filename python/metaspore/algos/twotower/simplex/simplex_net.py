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

from torch import nn

class SimpleXDense(torch.nn.Module):
    def __init__(self, emb_out_size, g):
        super().__init__()
        self._g = g
        self._v = torch.nn.Linear(emb_out_size, emb_out_size, bias=False)
        nn.init.xavier_uniform_(self._v.weight)

    def forward(self, x, y):
        y = self.average_pooling(y)
        z = self._g*x + (1-self._g)*y
        return z

    def average_pooling(self, sequence_emb):
        return self._v(sequence_emb)

class SimilarityModule(torch.nn.Module):
    def __init__(self, net_dropout=0.0):
        super().__init__()
        self.user_dropout = nn.Dropout(net_dropout) if net_dropout > 0 else None

    def forward(self, x, y):
        if self.user_dropout is not None:
            x = self.user_dropout(x)
        cosine_similarities = F.cosine_similarity(x, y, dim=1).reshape(-1,1)
        return cosine_similarities

class UserModule(torch.nn.Module):
    def __init__(self, column_name_path, user_combine_schema_path, interacted_items_combine_schema_path, emb_size, g, alpha, beta, l1, l2):
        super().__init__()
        self._embedding_size = emb_size
        self._g = g
        self._column_name_path = column_name_path
        self._user_combine_schema_path = user_combine_schema_path
        self._interacted_items_combine_schema_path = interacted_items_combine_schema_path

        self._sparse_user = ms.EmbeddingSumConcat(self._embedding_size, self._column_name_path, self._user_combine_schema_path)
        self._sparse_user.updater = ms.FTRLTensorUpdater(l1=l1, l2=l2, alpha = alpha, beta=beta)
        self._sparse_user.initializer = ms.NormalTensorInitializer(var = 0.0001)
        self._sparse_user.output_batchsize1_if_only_level0 = True

        self._sparse_interacted_items = ms.EmbeddingSumConcat(self._embedding_size, self._column_name_path, self._interacted_items_combine_schema_path, embedding_bag_mode='mean')
        self._sparse_interacted_items.updater = ms.FTRLTensorUpdater(l1=l1, l2=l2, alpha = alpha, beta=beta)
        self._sparse_interacted_items.initializer = ms.NormalTensorInitializer(var = 0.0001)
        self._sparse_interacted_items.output_batchsize1_if_only_level0 = True

        self._emb_out_size = self._sparse_user.feature_count * self._embedding_size
        self._dense = SimpleXDense(self._emb_out_size, self._g)

    def forward(self, x):
        a = self._sparse_user(x)
        b = self._sparse_interacted_items(x)
        x = self._dense(a, b)
        x = F.normalize(x)
        return x

class ItemModule(torch.nn.Module):
    def __init__(self, column_name_path, combine_schema_path, emb_size, alpha, beta, l1, l2):
        super().__init__()
        self._embedding_size = emb_size
        self._column_name_path = column_name_path
        self._combine_schema_path = combine_schema_path

        self._sparse = ms.EmbeddingSumConcat(self._embedding_size, self._column_name_path, self._combine_schema_path)
        self._sparse.updater = ms.FTRLTensorUpdater(l1=l1, l2=l2, alpha = alpha, beta=beta)
        self._sparse.initializer = ms.NormalTensorInitializer(var = 0.0001)

        self._emb_out_size = self._sparse.feature_count * self._embedding_size

    def forward(self, x):
        x = self._sparse(x)
        x = F.normalize(x)
        return x
