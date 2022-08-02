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

import asyncio
import os
import numpy
import torch
from . import _metaspore
from ._metaspore import SparseFeatureExtractor
from ._metaspore import HashUniquifier
from .feature_group import SparseFeatureGroup
from .url_utils import is_url
from .url_utils import use_s3
from .file_utils import file_exists
from .name_utils import is_valid_qualified_name
from .updater import TensorUpdater
from .initializer import TensorInitializer

#declare a class which we generate a onnx file to represent the sumconcat logic after Lookup
class EmbeddingBagModule(torch.nn.Module):
    def __init__(self, feature_count, emb_size, mode):
        super(EmbeddingBagModule, self).__init__()
        self.second_dim = feature_count * emb_size
        self.emb_size = emb_size
        self.feature_count = feature_count
        self.embedding_bag_mode = mode

    def forward(self, input, weight, offsets, batch_size):
        embs = torch.nn.functional.embedding_bag(input, weight, offsets, mode=self.embedding_bag_mode)
        return embs.reshape(batch_size, self.second_dim)

    def export_onnx(self, path, name):
        self.eval()
        torch_script = torch.jit.script(self)
        input = torch.tensor([i for i in range(self.feature_count)], dtype=torch.long)
        weight = torch.zeros(self.feature_count, self.emb_size)
        offsets = torch.tensor([i for i in range(self.feature_count)], dtype=torch.long)
        batch_size = torch.tensor(1, dtype=torch.long)
        class FakeStream(object):
            def write(self, data):
                _metaspore.stream_write_all(use_s3(path), data)
            def flush(self):
                pass
        if path.endswith('model.onnx'):
            dir = path[:-len('model.onnx')]
            _metaspore.ensure_local_directory(use_s3(dir))
        else:
            _metaspore.ensure_local_directory(use_s3(path))
            path = os.path.join(path, 'model.onnx')
        fout = FakeStream()

        torch.onnx.export(torch_script,
                        (input, weight, offsets, batch_size),
                        fout,
                        input_names=["input", "weight", "offsets", "batch_size"],
                        output_names= [name],
                        dynamic_axes={
                            "input": {0: "input_num"},
                            "weight": {0: "emb_vec_num"},
                            "offsets": {0: "offset_num"}
                        },
                        verbose=True,
                        opset_version=14)

class EmbeddingOperator(torch.nn.Module):
    def __init__(self,
                 embedding_size=None,
                 combine_schema_source=None,
                 combine_schema_file_path=None,
                 dtype=torch.float32,
                 requires_grad=True,
                 updater=None,
                 initializer=None,
                 output_batchsize1_if_only_level0=False,
                 use_nan_fill=False,
                 save_as_text=False,
                 embedding_bag_mode='sum'
                ):
        if embedding_size is not None:
            if not isinstance(embedding_size, int) or embedding_size <= 0:
                raise TypeError(f"embedding_size must be positive integer; {embedding_size!r} is invalid")
        if isinstance(combine_schema_source, SparseFeatureGroup):
            if combine_schema_file_path is not None:
                raise ValueError("can not specify combine_schema_file_path when SparseFeatureGroup is provided")
            combine_schema_source = combine_schema_source.schema_source
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"dtype must be one of: torch.float32, torch.float64; {dtype!r} is invalid")
        if updater is not None:
            if not isinstance(updater, TensorUpdater):
                raise TypeError(f"updater must be TensorUpdater; {updater!r} is invalid")
        if initializer is not None:
            if not isinstance(initializer, TensorInitializer):
                raise TypeError(f"initializer must be TensorInitializer; {initializer!r} is invalid")
        self._check_embedding_bag_mode(embedding_bag_mode)
        super().__init__()
        self._embedding_size = embedding_size
        self._combine_schema_source = None
        self._combine_schema_file_path = None
        if combine_schema_file_path is not None:
            self.combine_schema_file_path = combine_schema_file_path
        elif combine_schema_source is not None:
            self.combine_schema_source = combine_schema_source
        self._dtype = dtype
        self._requires_grad = requires_grad
        self._updater = updater
        self._initializer = initializer
        self._is_backing = False
        self._is_exported = True
        self._output_batchsize1_if_only_level0 = output_batchsize1_if_only_level0
        self._use_nan_fill = use_nan_fill
        self._save_as_text = save_as_text
        self._embedding_bag_mode = embedding_bag_mode
        self._distributed_tensor = None
        self._feature_extractor = None
        if self._combine_schema_source is not None:
            self._load_combine_schema()

        # init a embedding_bag
        self.sparse_embedding_bag = EmbeddingBagModule(self.feature_count, embedding_size, mode=self.embedding_bag_mode)
        self._clean()

    # the inerface to call the export_onnx
    @torch.jit.unused
    def export_sparse_embedding_bag(self, path, name):
        self.sparse_embedding_bag.export_onnx(path, name)

    @torch.jit.unused
    def _load_combine_schema(self):
        if self._feature_extractor is not None:
            raise RuntimeError("combine schema has been loaded")
        combine_schema_source = self._checked_get_combine_schema_source()
        # For offline, the table name does not matter. Use a placeholder value.
        table_name = '_sparse'
        self._feature_extractor = SparseFeatureExtractor(table_name, combine_schema_source)
        string = f"\033[32mloaded combine schema from\033[m "
        string += f"\033[32mcombine schema file \033[m{self.combine_schema_file_path!r}"
        print(string)

    @torch.jit.unused
    def _ensure_combine_schema_loaded(self):
        if self._feature_extractor is None:
            self._load_combine_schema()

    def __repr__(self):
        args = []
        if self._embedding_size is not None:
            args.append(f"{self._embedding_size}")
        if self._combine_schema_file_path is not None:
            args.append(f"combine_schema_file_path={self._combine_schema_file_path!r}")
        if self._dtype is not None and self._dtype is not torch.float32:
            args.append(f"dtype={self._dtype}")
        if not self._requires_grad:
            args.append("requires_grad=False")
        if self._updater is not None:
            args.append(f"updater={self._updater!r}")
        if self._initializer is not None:
            args.append(f"initializer={self._initializer!r}")
        if self._output_batchsize1_if_only_level0:
            args.append(f"output_batchsize1_if_only_level0={self._output_batchsize1_if_only_level0!r}")
        if self._use_nan_fill:
            args.append(f"use_nan_fill={self._use_nan_fill!r}")
        if self._save_as_text:
            args.append(f"save_as_text={self._save_as_text!r}")
        return f"{self.__class__.__name__}({', '.join(args)})"

    @property
    @torch.jit.unused
    def embedding_size(self):
        return self._embedding_size

    @embedding_size.setter
    @torch.jit.unused
    def embedding_size(self, value):
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise TypeError(f"embedding_size must be positive integer; {value!r} is invalid")
        if self._embedding_size is not None:
            raise RuntimeError(f"can not reset embedding_size {self._embedding_size} to {value!r}")
        self._embedding_size = value

    @torch.jit.unused
    def _checked_get_embedding_size(self):
        if self._embedding_size is None:
            raise RuntimeError("embedding_size is not set")
        return self._embedding_size

    @property
    @torch.jit.unused
    def is_backing(self):
        return self._is_backing

    @is_backing.setter
    @torch.jit.unused
    def is_backing(self, value):
        self._is_backing = value

    @property
    @torch.jit.unused
    def is_exported(self):
        return self._is_exported

    @is_exported.setter
    @torch.jit.unused
    def is_exported(self, value):
        self._is_exported = value

    @property
    @torch.jit.unused
    def combine_schema_source(self):
        return self._combine_schema_source

    @combine_schema_source.setter
    @torch.jit.unused
    def combine_schema_source(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(f"combine_schema_source must be string; {value!r} is invalid")
        if self._combine_schema_source is not None:
            raise RuntimeError(f"can not reset combine_schema_source {self._combine_schema_source!r} to {value!r}")
        self.combine_schema_file_path = value

    @torch.jit.unused
    def _checked_get_combine_schema_source(self):
        if self._combine_schema_source is None:
            raise RuntimeError("combine_schema_source is not set")
        return self._combine_schema_source

    @property
    @torch.jit.unused
    def combine_schema_file_path(self):
        return self._combine_schema_file_path

    @combine_schema_file_path.setter
    @torch.jit.unused
    def combine_schema_file_path(self, value):
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(f"combine_schema_file_path must be string; {value!r} is invalid")
        if self._combine_schema_file_path is not None:
            raise RuntimeError(f"can not reset combine_schema_file_path {self._combine_schema_file_path!r} to {value!r}")
        if value is None:
            self._combine_schema_source = None
            self._combine_schema_file_path = None
        else:
            if is_url(value):
                value = use_s3(value)
                if not file_exists(value):
                    raise RuntimeError(f"combine schema file {value!r} not found")
                combine_schema_data = _metaspore.stream_read_all(value)
                self._combine_schema_source = combine_schema_data.decode('utf-8')
                self._combine_schema_file_path = value
            else:
                self._combine_schema_source = value
                self._combine_schema_file_path = '<string>'

    @torch.jit.unused
    def _checked_get_combine_schema_file_path(self):
        if self._combine_schema_file_path is None:
            raise RuntimeError("combine_schema_file_path is not set")
        return self._combine_schema_file_path

    @property
    @torch.jit.unused
    def feature_count(self):
        extractor = self._feature_extractor
        if extractor is None:
            raise RuntimeError(f"combine schema is not loaded; can not get feature count")
        count = extractor.feature_count
        return count

    @property
    @torch.jit.unused
    def dtype(self):
        return self._dtype

    @property
    @torch.jit.unused
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    @torch.jit.unused
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    @torch.jit.unused
    def updater(self):
        return self._updater

    @updater.setter
    @torch.jit.unused
    def updater(self, value):
        if value is not None:
            if not isinstance(value, TensorUpdater):
                raise TypeError(f"updater must be TensorUpdater; {value!r} is invalid")
        if self._updater is not None:
            raise RuntimeError(f"can not reset updater {self._updater!r} to {value!r}")
        self._updater = value

    @property
    @torch.jit.unused
    def initializer(self):
        return self._initializer

    @initializer.setter
    @torch.jit.unused
    def initializer(self, value):
        if value is not None:
            if not isinstance(value, TensorInitializer):
                raise TypeError(f"initializer must be TensorInitializer; {value!r} is invalid")
        if self._initializer is not None:
            raise RuntimeError(f"can not reset initializer {self._initializer!r} to {value!r}")
        self._initializer = value

    @property
    @torch.jit.unused
    def output_batchsize1_if_only_level0(self):
        return self._output_batchsize1_if_only_level0

    @output_batchsize1_if_only_level0.setter
    @torch.jit.unused
    def output_batchsize1_if_only_level0(self, value):
        self._output_batchsize1_if_only_level0 = value

    @property
    @torch.jit.unused
    def use_nan_fill(self):
        return self._use_nan_fill

    @use_nan_fill.setter
    @torch.jit.unused
    def use_nan_fill(self, value):
        self._use_nan_fill = value

    @property
    @torch.jit.unused
    def save_as_text(self):
        return self._save_as_text

    @save_as_text.setter
    def save_as_text(self, value):
        self._save_as_text = value

    @property
    @torch.jit.unused
    def embedding_bag_mode(self):
        return self._embedding_bag_mode

    @embedding_bag_mode.setter
    def embedding_bag_mode(self, value):
        self._embedding_bag_mode = value

    @property
    @torch.jit.unused
    def _is_clean(self):
        return (self._indices is None and
                self._indices_meta is None and
                self._keys is None and
                self._data is None)

    @torch.jit.unused
    def _clean(self):
        self._indices = None
        self._indices_meta = None
        self._keys = None
        self._data = None
        self._output = torch.tensor(0.0)

    @torch.jit.unused
    def _check_clean(self):
        if self._keys is None and not self._is_clean:
            raise RuntimeError(f"{self!r} is expected to be clean when keys is None")

    @torch.jit.unused
    def _check_dtype_and_shape(self, keys, data):
        if not isinstance(data, numpy.ndarray):
            raise TypeError(f"data must be numpy.ndarray; got {type(data)!r}")
        dtype = str(self.dtype).rpartition('.')[-1]
        if data.dtype.name != dtype:
            raise TypeError(f"data must be numpy.ndarray of {dtype}; got {data.dtype.name}")
        embedding_size = self._checked_get_embedding_size()
        shape = len(keys), embedding_size
        if data.shape != shape:
            raise RuntimeError(f"data shape mismatches with op; {data.shape} vs. {shape}")

    @property
    @torch.jit.unused
    def keys(self):
        self._check_clean()
        return self._keys

    @property
    @torch.jit.unused
    def data(self):
        self._check_clean()
        return self._data

    @property
    @torch.jit.unused
    def grad(self):
        data = self.data
        if data is None:
            return None
        return data.grad

    @property
    @torch.jit.unused
    def output(self):
        return self._output

    @property
    @torch.jit.unused
    def keys_and_data(self):
        self._check_clean()
        if self._keys is None and self._data is None:
            return None, None
        if self._keys is not None and self._data is not None:
            return self._keys, self._data
        raise RuntimeError(f"keys and data of {self!r} must be both None or both not None")

    @keys_and_data.setter
    @torch.jit.unused
    def keys_and_data(self, value):
        if value is None:
            self._clean()
            return
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"value must be None or a pair of keys and data; {value!r} is invalid")
        keys, data = value
        if keys is None and value is None:
            self._clean()
            return
        if keys is not None and data is not None:
            if isinstance(keys, torch.Tensor):
                keys = keys.numpy()
            if not isinstance(keys, numpy.ndarray) or len(keys.shape) != 1 or keys.dtype not in (numpy.int64, numpy.uint64):
                raise TypeError("keys must be 1-D array of int64 or uint64")
            keys = keys.view(numpy.uint64)
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            self._check_dtype_and_shape(keys, data)
            self._clean()
            self._keys = keys
            self._update_data(data)
            return
        raise RuntimeError(f"keys and data of {self!r} must be both None or both not None")

    @torch.jit.unused
    def _update_data(self, data):
        self._data = torch.from_numpy(data)
        self._data.requires_grad = self.training and self.requires_grad

    @torch.jit.unused
    def _combine_to_indices_and_offsets(self, minibatch, feature_offset):
        import pyarrow as pa
        batch = pa.RecordBatch.from_pandas(minibatch)
        indices, offsets = self._feature_extractor.extract(batch)
        if not feature_offset:
            offsets = offsets[::self.feature_count]
        return indices, offsets

    @torch.jit.unused
    def _do_combine(self, minibatch):
        raise NotImplementedError

    @torch.jit.unused
    def _uniquify_hash_codes(self, indices):
        keys = HashUniquifier.uniquify(indices)
        return keys

    @torch.jit.unused
    def _combine(self, minibatch):
        self._clean()
        self._ensure_combine_schema_loaded()
        self._indices, self._indices_meta = self._do_combine(minibatch)
        self._keys = self._uniquify_hash_codes(self._indices)

    @torch.jit.unused
    def _check_embedding_bag_mode(self, mode):
        if mode not in ('mean', 'sum', 'max'):
            raise ValueError(f"embedding bag mode must be one of: 'mean', 'sum', 'max'; {mode!r} is invalid")

    @torch.jit.unused
    def _compute_sum_concat(self):
        self._check_embedding_bag_mode(self.embedding_bag_mode)
        feature_count = self.feature_count
        minibatch_size = len(self._indices_meta) // feature_count
        embedding_size = self._checked_get_embedding_size()
        indices = self._indices
        indices_1d = torch.from_numpy(indices.view(numpy.int64))
        offsets = self._indices_meta
        offsets_1d = torch.from_numpy(offsets.view(numpy.int64))

        # we use the forward method to replace the torch.nn.functional.embedding_bag
        out = self.sparse_embedding_bag.forward(indices_1d, self._data, offsets_1d, minibatch_size)

        return out

    @torch.jit.unused
    def _compute_embedding_prepare(self):
        minibatch_size = self._indices_meta.shape[0]
        embedding_size = self._checked_get_embedding_size()
        indices = self._indices
        indices_1d = torch.from_numpy(indices.view(numpy.int64))
        offsets = self._indices_meta
        offsets_1d = torch.from_numpy(offsets.astype(numpy.int64))
        t = minibatch_size, embedding_size, indices_1d, offsets_1d
        return t

    @torch.jit.unused
    def _compute_range_sum(self):
        self._check_embedding_bag_mode(self.embedding_bag_mode)
        t = self._compute_embedding_prepare()
        minibatch_size, embedding_size, indices_1d, offsets_1d = t

        # we use the forward method to replace the torch.nn.functional.embedding_bag
        out = self.sparse_embedding_bag.forward(indices_1d, self._data, offsets_1d, minibatch_size)

        return out

    @torch.jit.unused
    def _compute_embedding_lookup(self):
        t = self._compute_embedding_prepare()
        minibatch_size, embedding_size, indices_1d, offsets_1d = t
        embs = torch.nn.functional.embedding(indices_1d, self._data)
        expected_shape = len(indices_1d), embedding_size
        if embs.shape != expected_shape:
            raise RuntimeError(f"embs has unexpected shape; expect {expected_shape}, found {embs.shape}")
        out = embs, offsets_1d
        return out

    @torch.jit.unused
    def _do_compute(self):
        raise NotImplementedError

    @torch.jit.unused
    def _compute(self):
        if self._is_clean:
            raise RuntimeError(f"{self!r} is clean; can not execute the 'compute' step")
        self._output = self._do_compute()

    def forward(self, x):
        return self._output

    @torch.jit.unused
    async def _sparse_tensor_clear(self):
        tensor = self._distributed_tensor
        await tensor._sparse_tensor_clear()

    @torch.jit.unused
    async def _sparse_tensor_import_from(self, meta_file_path, *,
                                         data_only=False, skip_existing=False,
                                         transform_key=False, feature_name=''):
        tensor = self._distributed_tensor
        await tensor._sparse_tensor_import_from(meta_file_path,
            data_only=data_only, skip_existing=skip_existing,
            transform_key=transform_key, feature_name=feature_name)

    @torch.jit.unused
    def clear(self):
        if self._distributed_tensor is None:
            raise RuntimeError(f"{self!r} is not properly initialized; attribute '_distributed_tensor' is None")
        tensor = self._distributed_tensor
        agent = tensor._handle.agent
        agent.barrier()
        if agent.rank == 0:
            asyncio.run(self._sparse_tensor_clear())
        agent.barrier()

    @torch.jit.unused
    def import_from(self, meta_file_path, *,
                    clear_existing=False,
                    data_only=False, skip_existing=False,
                    transform_key=False, feature_name=''):
        if self._distributed_tensor is None:
            raise RuntimeError(f"{self!r} is not properly initialized; attribute '_distributed_tensor' is None")
        if transform_key and not feature_name:
            raise ValueError("feature_name must be specified to transform key")
        if clear_existing:
            self.clear()
        tensor = self._distributed_tensor
        agent = tensor._handle.agent
        agent.barrier()
        asyncio.run(self._sparse_tensor_import_from(meta_file_path,
            data_only=data_only, skip_existing=skip_existing,
            transform_key=transform_key, feature_name=feature_name))
        agent.barrier()

class EmbeddingSumConcat(EmbeddingOperator):
    @torch.jit.unused
    def _do_combine(self, minibatch):
        return self._combine_to_indices_and_offsets(minibatch, True)

    @torch.jit.unused
    def _do_compute(self):
        return self._compute_sum_concat()

    @torch.jit.unused
    def _clean(self):
        super()._clean()
        self._output = torch.zeros((2, self.feature_count * self.embedding_size))

class EmbeddingRangeSum(EmbeddingOperator):
    @torch.jit.unused
    def _do_combine(self, minibatch):
        return self._combine_to_indices_and_offsets(minibatch, False)

    @torch.jit.unused
    def _do_compute(self):
        return self._compute_range_sum()

    @torch.jit.unused
    def _clean(self):
        super()._clean()
        self._output = torch.zeros((2, self.embedding_size))

class EmbeddingLookup(EmbeddingOperator):
    @torch.jit.unused
    def _do_combine(self, minibatch):
        return self._combine_to_indices_and_offsets(minibatch, True)

    @torch.jit.unused
    def _do_compute(self):
        return self._compute_embedding_lookup()

    @torch.jit.unused
    def _clean(self):
        super()._clean()
        self._output = torch.zeros((2, self.embedding_size)), torch.zeros((1,), dtype=torch.int64)
