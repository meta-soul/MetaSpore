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
from .fm import FMModule
from .wide_and_deep import WideAndDeepModule

class DeepFMModule(WideAndDeepModule):
    def __init__(self,
                 cross_sparse_embedding_size=16,
                 cross_sparse_column_name_path=None,
                 cross_sparse_combine_schema_path=None,
                 cross_sparse_updater=None,
                 cross_sparse_initializer=None,
                 **kwargs
                ):
        super().__init__(**kwargs)
        if cross_sparse_column_name_path is None:
            raise ValueError("cross_sparse_column_name_path is required")
        if cross_sparse_combine_schema_path is None:
            raise ValueError("cross_sparse_combine_schema_path is required")
        if cross_sparse_updater is None:
            cross_sparse_updater = FTRLTensorUpdater()
        if cross_sparse_initializer is None:
            cross_sparse_initializer = NormalTensorInitializer(var=0.01)
        self._cross_sparse_embedding_size = cross_sparse_embedding_size
        self._cross_sparse_column_name_path = cross_sparse_column_name_path
        self._cross_sparse_combine_schema_path = cross_sparse_combine_schema_path
        self._cross_sparse = EmbeddingSumConcat(self._cross_sparse_embedding_size,
                                                self._cross_sparse_column_name_path,
                                                self._cross_sparse_combine_schema_path)
        self._cross_sparse.updater = cross_sparse_updater
        self._cross_sparse.initializer = cross_sparse_initializer
        self._cross_sparse_feature_count = self._cross_sparse.feature_count
        self._fm = FMModule()

    def forward(self, inputs):
        wide_outputs = self._wide(inputs)
        wide_outputs = torch.sum(wide_outputs, dim=1, keepdim=True)
        cross_sparse_outputs = self._cross_sparse(inputs)
        cross_sparse_outputs = cross_sparse_outputs.reshape(
                                    -1,
                                    self._cross_sparse_feature_count,
                                    self._cross_sparse_embedding_size)
        fm_outputs = self._fm(cross_sparse_outputs)
        deep_sparse_outputs = self._deep_sparse(inputs)
        deep_outputs = self._deep_dense(deep_sparse_outputs)
        return torch.sigmoid(wide_outputs + fm_outputs + deep_outputs)
