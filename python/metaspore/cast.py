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

import numpy
import torch
from ._metaspore import CombineSchema
from .url_utils import use_s3
from .file_utils import file_exists

class Cast(torch.nn.Module):
    def __init__(self, selected_columns, column_name_file_path=None, dtype=torch.float32):
        if not isinstance(selected_columns, (list, tuple)):
            raise TypeError(f"list or tuple of strings expected; {selected_columns!r} is invalid")
        if not all(isinstance(item, str) for item in selected_columns):
            raise TypeError(f"list or tuple of strings expected; {selected_columns!r} is invalid")
        if not selected_columns:
            raise ValueError("selected_columns must not be empty")
        if column_name_file_path is not None:
            if not isinstance(column_name_file_path, str) or not file_exists(column_name_file_path):
                raise RuntimeError(f"column name file {column_name_file_path!r} not found")
        if dtype not in (torch.int32, torch.int64, torch.float32, torch.float64):
            raise TypeError(f"dtype must be one of: torch.int32, torch.int64, "
                            f"torch.float32, torch.float64; {dtype!r} is invalid")
        super().__init__()
        self._selected_columns = tuple(selected_columns)
        self._column_name_file_path = column_name_file_path
        self._dtype = dtype
        self._dtype_name = str(self._dtype).rpartition('.')[-1]
        self._column_name_map = None
        if self._column_name_file_path is not None:
            self._load_column_name_map()
        self._clean()

    @torch.jit.unused
    def _load_column_name_map(self):
        if self._column_name_map is not None:
            raise RuntimeError("column map has been loaded")
        column_name_file_path = self._checked_get_column_name_file_path()
        combine_schema = CombineSchema()
        combine_schema.load_column_name_from_file(use_s3(column_name_file_path))
        self._column_name_map = combine_schema.get_column_name_map()
        string = f"\033[32mloaded column name map from \033[m{column_name_file_path!r}"
        print(string)
        keys = tuple(item for item in self._selected_columns if item not in self._column_name_map)
        if keys:
            raise ValueError(f"the following columns are not defined in {self._column_name_file_path!r}: "
                             f"{', '.join(keys)}")

    @torch.jit.unused
    def _ensure_column_name_map_loaded(self):
        if self._column_name_map is None:
            self._load_column_name_map()

    @torch.jit.unused
    def _checked_get_column_name_file_path(self):
        if self._column_name_file_path is None:
            raise RuntimeError("column_name_file_path is not set")
        return self._column_name_file_path

    @torch.jit.unused
    def _clean(self):
        self._output = torch.tensor(0.0)

    @torch.jit.unused
    def _do_cast(self, ndarrays):
        columns = []
        for name in self._selected_columns:
            index = self._column_name_map[name]
            if index >= len(ndarrays):
                raise ValueError(f"column {name}({index}) is out of range; only {len(ndarrays)} columns in ndarrays")
            ndarray = ndarrays[index]
            column = ndarray.astype(self._dtype_name)
            columns.append(column)
        output = numpy.stack(columns, axis=-1)
        output = torch.from_numpy(output)
        return output

    @torch.jit.unused
    def _cast(self, ndarrays):
        self._clean()
        self._ensure_column_name_map_loaded()
        self._output = self._do_cast(ndarrays)

    def forward(self, x):
        return self._output
