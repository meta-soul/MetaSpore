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
from ..updater import FTRLTensorUpdater
from ..initializer import NormalTensorInitializer
from ..embedding import EmbeddingSumConcat
from .normalization import Normalization

class WideAndDeepModule(torch.nn.Module):
    def __init__(self,
                 wide_embedding_size=16,
                 wide_column_name_path=None,
                 wide_combine_schema_path=None,
                 wide_updater=None,
                 wide_initializer=None,
                 deep_sparse_embedding_size=16,
                 deep_sparse_column_name_path=None,
                 deep_sparse_combine_schema_path=None,
                 deep_sparse_updater=None,
                 deep_sparse_initializer=None,
                 deep_dense_hidden_units=(1024, 512),
                ):
        super().__init__()
        if wide_column_name_path is None:
            raise ValueError("wide_column_name_path is required")
        if wide_combine_schema_path is None:
            raise ValueError("wide_combine_schema_path is required")
        if wide_updater is None:
            wide_updater = FTRLTensorUpdater()
        if wide_initializer is None:
            wide_initializer = NormalTensorInitializer(var=0.01)
        if deep_sparse_column_name_path is None:
            raise ValueError("deep_sparse_column_name_path is required")
        if deep_sparse_combine_schema_path is None:
            raise ValueError("deep_sparse_combine_schema_path is required")
        if deep_sparse_updater is None:
            deep_sparse_updater = FTRLTensorUpdater()
        if deep_sparse_initializer is None:
            deep_sparse_initializer = NormalTensorInitializer(var=0.01)
        if not deep_dense_hidden_units:
            raise ValueError("deep_dense_hidden_units can not be empty")
        self._wide_embedding_size = wide_embedding_size
        self._wide_column_name_path = wide_column_name_path
        self._wide_combine_schema_path = wide_combine_schema_path
        self._wide = EmbeddingSumConcat(self._wide_embedding_size,
                                        self._wide_column_name_path,
                                        self._wide_combine_schema_path)
        self._wide.updater = wide_updater
        self._wide.initializer = wide_initializer
        self._deep_sparse_embedding_size = deep_sparse_embedding_size
        self._deep_sparse_column_name_path = deep_sparse_column_name_path
        self._deep_sparse_combine_schema_path = deep_sparse_combine_schema_path
        self._deep_sparse = EmbeddingSumConcat(self._deep_sparse_embedding_size,
                                               self._deep_sparse_column_name_path,
                                               self._deep_sparse_combine_schema_path)
        self._deep_sparse.updater = deep_sparse_updater
        self._deep_sparse.initializer = deep_sparse_initializer
        modules = []
        deep_dense_input_units = self._deep_sparse.feature_count * self._deep_sparse_embedding_size
        modules.append(Normalization(deep_dense_input_units))
        modules.append(torch.nn.Linear(deep_dense_input_units, deep_dense_hidden_units[0]))
        modules.append(torch.nn.ReLU())
        for i in range(len(deep_dense_hidden_units)):
            input_units = deep_dense_hidden_units[i]
            if i != len(deep_dense_hidden_units) - 1:
                output_units = deep_dense_hidden_units[i + 1]
            else:
                output_units = 1
            modules.append(torch.nn.Linear(input_units, output_units))
            if i != len(deep_dense_hidden_units) - 1:
                modules.append(torch.nn.ReLU())
        self._deep_dense = torch.nn.Sequential(*modules)

    def forward(self, inputs):
        wide_outputs = self._wide(inputs)
        wide_outputs = torch.sum(wide_outputs, dim=1, keepdim=True)
        deep_sparse_outputs = self._deep_sparse(inputs)
        deep_outputs = self._deep_dense(deep_sparse_outputs)
        return torch.sigmoid(wide_outputs + deep_outputs)
