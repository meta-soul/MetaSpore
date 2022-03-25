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
import torch
from ._metaspore import DenseTensor
from ._metaspore import SparseTensor
from .embedding import EmbeddingOperator
from .url_utils import use_s3

class DistributedTensor(object):
    def __init__(self, name, item, name_prefix):
        self.__name = name if name_prefix is None else name_prefix + name
        self.__item = item
        self.__handle = None

    @property
    def name(self):
        return self.__name

    @property
    def item(self):
        return self.__item

    @property
    def _handle(self):
        return self.__handle

    @property
    def is_dense(self):
        return isinstance(self.item, torch.Tensor)

    @property
    def is_dense_parameter(self):
        return isinstance(self.item, torch.nn.Parameter)

    @property
    def is_dense_buffer(self):
        return self.is_dense and not self.is_dense_parameter

    @property
    def is_sparse(self):
        return isinstance(self.item, EmbeddingOperator)

    @property
    def is_backing(self):
        return self.is_sparse and self.item.is_backing

    @property
    def is_exported(self):
        return self.is_sparse and self.item.is_exported

    def _zero_grad(self):
        if self.is_dense_parameter or self.is_sparse:
            if self.item.grad is not None:
                self.item.grad.detach_()
                self.item.grad.zero_()

    def _init_tensor(self, trainer):
        if self.is_dense:
            return self._init_dense_tensor(trainer)
        else:
            return self._init_sparse_tensor(trainer)

    def _init_tensor_log(self, x):
        if self.is_dense_parameter:
            string = "\033[38;5;046m"
        elif self.is_dense_buffer:
            string = "\033[38;5;051m"
        else:
            string = "\033[38;5;196m"
        if self.is_dense:
            string += f"init dense tensor {x.name} with shape {x.data_shape}, "
        else:
            string += f"init sparse tensor {x.name} with slice shape {x.slice_data_shape}, "
        string += f"updater {x.updater} and "
        string += f"initializer {x.initializer}"
        string += "\033[m"
        print(string)

    def _init_dense_tensor(self, trainer):
        x = DenseTensor()
        x.name = self.name
        x.data_type = trainer._get_dtype_name(self)
        x.data_shape = trainer._get_dense_data_shape(self)
        x.state_shape = trainer._get_dense_state_shape(self)
        x.initializer = trainer._get_dense_initializer(self)
        x.updater = trainer._get_dense_updater(self)
        x.partition_count = trainer.agent.server_count
        x.agent = trainer.agent._cxx_agent
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def init_dense_tensor_done():
            self.__handle = x
            loop.call_soon_threadsafe(future.set_result, None)
        self._init_tensor_log(x)
        x.init(init_dense_tensor_done)
        return future

    def _init_sparse_tensor(self, trainer):
        x = SparseTensor()
        x.name = self.name
        x.data_type = trainer._get_dtype_name(self)
        x.slice_data_shape = trainer._get_sparse_slice_data_shape(self)
        x.slice_state_shape = trainer._get_sparse_slice_state_shape(self)
        x.initializer = trainer._get_sparse_initializer(self)
        x.updater = trainer._get_sparse_updater(self)
        x.partition_count = trainer.agent.server_count
        x.agent = trainer.agent._cxx_agent
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def init_sparse_tensor_done():
            self.__handle = x
            loop.call_soon_threadsafe(future.set_result, None)
        self._init_tensor_log(x)
        x.init(init_sparse_tensor_done)
        return future

    def _pull_tensor(self):
        if self.is_dense:
            return self._pull_dense_tensor()
        else:
            return self._pull_sparse_tensor()

    def _pull_dense_tensor(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def pull_dense_tensor_done(data):
            data = torch.from_numpy(data)
            data = data.view(self.item.shape)
            self.item.data.copy_(data)
            loop.call_soon_threadsafe(future.set_result, None)
        self._handle.pull(pull_dense_tensor_done, False)
        return future

    async def _pull_sparse_tensor(self):
        op = self.item
        keys = op.keys
        if keys is None:
            return
        read_only = not op.training or not op.requires_grad
        nan_fill = read_only and op.use_nan_fill
        def pull_sparse_tensor():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            def pull_sparse_tensor_done(data):
                op._check_dtype_and_shape(keys, data)
                op._update_data(data)
                loop.call_soon_threadsafe(future.set_result, None)
            self._handle.pull(keys, pull_sparse_tensor_done, read_only, nan_fill)
            return future
        await pull_sparse_tensor()

    def _push_tensor(self, *, is_value=False, skip_no_grad=True):
        if self.is_dense:
            return self._push_dense_tensor(is_value=is_value, skip_no_grad=skip_no_grad)
        else:
            return self._push_sparse_tensor(is_value=is_value, skip_no_grad=skip_no_grad)

    async def _push_dense_tensor(self, *, is_value=False, skip_no_grad=True):
        data = self.item
        if self.is_dense_parameter:
            if not is_value and data.grad is None:
                if skip_no_grad:
                    return
                raise RuntimeError(f"the gradient of parameter {self.name!r} is not available")
        # For dense buffers, use .data to fake gradients.
        # But we still need to pass is_value=False, otherwise updaters on server won't be called.
        data = data.data.numpy() if self.is_dense_buffer or is_value else data.grad.data.numpy()
        def push_dense_tensor():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            def push_dense_tensor_done():
                loop.call_soon_threadsafe(future.set_result, None)
            self._handle.push(data, push_dense_tensor_done, is_value, False)
            return future
        await push_dense_tensor()

    async def _push_sparse_tensor(self, *, is_value=False, skip_no_grad=True):
        op = self.item
        keys, data = op.keys_and_data
        if keys is None:
            return
        if not is_value and data.grad is None:
            if skip_no_grad:
                return
            raise RuntimeError(f"the gradient of operator {op!r} is not available")
        data = data.data.numpy() if is_value else data.grad.data.numpy()
        op._check_dtype_and_shape(keys, data)
        def push_sparse_tensor():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            def push_sparse_tensor_done():
                loop.call_soon_threadsafe(future.set_result, None)
            self._handle.push(keys, data, push_sparse_tensor_done, is_value)
            return future
        await push_sparse_tensor()

    def _load_tensor(self, dir_path, *, keep_meta=False):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def load_tensor_done():
            loop.call_soon_threadsafe(future.set_result, None)
        dir_path = use_s3(dir_path)
        self._handle.load(dir_path, load_tensor_done, keep_meta)
        return future

    def _save_tensor(self, dir_path):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def save_tensor_done():
            loop.call_soon_threadsafe(future.set_result, None)
        dir_path = use_s3(dir_path)
        if self.is_sparse:
            text_mode = self.item.save_as_text
            self._handle.save(dir_path, save_tensor_done, text_mode)
        else:
            self._handle.save(dir_path, save_tensor_done)
        return future

    def _sparse_tensor_clear(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def sparse_tensor_clear_done():
            loop.call_soon_threadsafe(future.set_result, None)
        self._handle.clear(sparse_tensor_clear_done)
        return future

    def _sparse_tensor_export(self, dir_path):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def sparse_tensor_export_done():
            loop.call_soon_threadsafe(future.set_result, None)
        dir_path = use_s3(dir_path)
        self._handle.export(dir_path, sparse_tensor_export_done)
        return future

    def _sparse_tensor_import_from(self, meta_file_path, *,
                                   data_only=False, skip_existing=False,
                                   transform_key=False, feature_name=''):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def sparse_tensor_import_from_done():
            loop.call_soon_threadsafe(future.set_result, None)
        meta_file_path = use_s3(meta_file_path)
        self._handle.import_from(meta_file_path, sparse_tensor_import_from_done,
                                 data_only, skip_existing,
                                 transform_key, feature_name)
        return future

    def _sparse_tensor_prune_small(self, epsilon):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def sparse_tensor_prune_small_done():
            loop.call_soon_threadsafe(future.set_result, None)
        self._handle.prune_small(epsilon, sparse_tensor_prune_small_done)
        return future

    def _sparse_tensor_prune_old(self, max_age):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        def sparse_tensor_prune_old_done():
            loop.call_soon_threadsafe(future.set_result, None)
        self._handle.prune_old(max_age, sparse_tensor_prune_old_done)
        return future
