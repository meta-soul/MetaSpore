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

class Cast(torch.nn.Module):
    def __init__(self, selected_columns, dtype=torch.float32):
        if not isinstance(selected_columns, (list, tuple)):
            raise TypeError(f"list or tuple of strings expected; {selected_columns!r} is invalid")
        if not all(isinstance(item, str) for item in selected_columns):
            raise TypeError(f"list or tuple of strings expected; {selected_columns!r} is invalid")
        if not selected_columns:
            raise ValueError("selected_columns must not be empty")
        if dtype not in (torch.int32, torch.int64, torch.float32, torch.float64):
            raise TypeError(f"dtype must be one of: torch.int32, torch.int64, "
                            f"torch.float32, torch.float64; {dtype!r} is invalid")
        super().__init__()
        self._selected_columns = tuple(selected_columns)
        self._dtype = dtype
        self._dtype_name = str(self._dtype).rpartition('.')[-1]
        self._clean()

    @torch.jit.unused
    def _clean(self):
        self._output = torch.tensor(0.0)

    @torch.jit.unused
    def _do_cast(self, minibatch):
        columns = []
        for name in self._selected_columns:
            column = minibatch[name].values.astype(self._dtype_name)
            columns.append(column)
        output = numpy.stack(columns, axis=-1)
        output = torch.from_numpy(output)
        return output

    @torch.jit.unused
    def _cast(self, minibatch):
        self._clean()
        self._output = self._do_cast(minibatch)

    def forward(self, x):
        return self._output
