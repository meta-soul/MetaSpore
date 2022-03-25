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
from .updater import TensorUpdater
from .updater import SGDTensorUpdater
from .initializer import TensorInitializer
from .initializer import DefaultTensorInitializer
from .model import Model

class DistributedTrainer(object):
    def __init__(self, model, updater=None, initializer=None):
        if not isinstance(model, Model):
            raise TypeError(f"model must be Model; {model!r} is invalid")
        if updater is None:
            updater = SGDTensorUpdater(0.1)
        if not isinstance(updater, TensorUpdater):
            raise TypeError(f"updater must be TensorUpdater; {updater!r} is invalid")
        if initializer is None:
            initializer = DefaultTensorInitializer()
        if not isinstance(initializer, TensorInitializer):
            raise TypeError(f"initializer must be TensorInitializer; {initializer!r} is invalid")
        self._model = model
        self._updater = updater
        self._initializer = initializer
        self._skip_no_grad = True

    @property
    def model(self):
        return self._model

    @property
    def agent(self):
        return self._model.agent

    @property
    def updater(self):
        return self._updater

    @property
    def initializer(self):
        return self._initializer

    @property
    def skip_no_grad(self):
        return self._skip_no_grad

    @skip_no_grad.setter
    def skip_no_grad(self, value):
        self._skip_no_grad = value

    def _get_dtype_name(self, tensor):
        return str(tensor.item.dtype).rpartition('.')[-1]

    def _get_dense_initializer(self, tensor):
        initializer = getattr(tensor.item, 'initializer', None)
        if initializer is None:
            initializer = self.initializer
        if not isinstance(initializer, TensorInitializer):
            message = "initializer must be an instance of TensorInitializer; "
            message += f"{initializer!r} is invalid"
            raise TypeError(message)
        return initializer

    def _get_dense_updater(self, tensor):
        updater = getattr(tensor.item, 'updater', None)
        if updater is None:
            updater = self.updater
        if not isinstance(updater, TensorUpdater):
            message = "updater must be an instance of TensorUpdater; "
            message += f"{updater!r} is invalid"
            raise TypeError(message)
        return updater

    def _get_sparse_initializer(self, tensor):
        initializer = getattr(tensor.item, 'initializer', None)
        if initializer is None:
            initializer = self.initializer
        if not isinstance(initializer, TensorInitializer):
            message = "initializer must be an instance of TensorInitializer; "
            message += f"{initializer!r} is invalid"
            raise TypeError(message)
        return initializer

    def _get_sparse_updater(self, tensor):
        updater = getattr(tensor.item, 'updater', None)
        if updater is None:
            updater = self.updater
        if not isinstance(updater, TensorUpdater):
            message = "updater must be an instance of TensorUpdater; "
            message += f"{updater!r} is invalid"
            raise TypeError(message)
        return updater

    def _get_dense_data_shape(self, tensor):
        updater = self._get_dense_updater(tensor)
        result = updater.get_dense_data_shape(tensor)
        return result

    def _get_dense_state_shape(self, tensor):
        updater = self._get_dense_updater(tensor)
        result = updater.get_dense_state_shape(tensor)
        return result or ()

    def _get_sparse_slice_data_shape(self, tensor):
        updater = self._get_sparse_updater(tensor)
        result = updater.get_sparse_slice_data_shape(tensor)
        return result

    def _get_sparse_slice_state_shape(self, tensor):
        updater = self._get_sparse_updater(tensor)
        result = updater.get_sparse_slice_state_shape(tensor)
        return result or ()

    def initialize(self):
        self.agent.barrier()
        self.model._configure_batch_norms()
        self.model._collect_tensors()
        asyncio.run(self.model._init_tensors(self))
        # We now always use local initialize mode since this is more natural.
        # Dense tensors will first be initialized by dense initializers,
        # then their values will be overridden by those from the PyTorch model.
        # Sparse tensors are still initialized by sparse initializers block by block.
        self.agent.barrier()
        if self.agent.rank == 0:
            asyncio.run(self.model._push_tensors(is_value=True))
        self.agent.barrier()
        asyncio.run(self.model._pull_tensors(force_mode=True))
        self.agent.barrier()

    def load(self, dir_path, *, keep_meta=False):
        # When spare tensors are repartitioned, we need to make
        # sure sparse tensors are cleared, as ``import_from``
        # won't clear or override existing keys. Make sure this
        # in C++ is a bit complicated, so we do it here.
        self.agent.barrier()
        if self.agent.rank == 0:
            asyncio.run(self.model._clear_tensors())
        # When spare tensors are repartitioned, they must be loaded
        # and pushed to servers later by all workers, so we do not
        # use ``if self.agent.rank == 0:`` here, instead this will
        # be checked in C++ code when necessary.
        self.agent.barrier()
        asyncio.run(self.model._load_tensors(dir_path, keep_meta=keep_meta))
        self.agent.barrier()
        asyncio.run(self.model._pull_tensors(force_mode=True))
        self.agent.barrier()

    def save(self, dir_path):
        self.agent.barrier()
        if self.agent.rank == 0:
            asyncio.run(self.model._save_tensors(dir_path))
        self.agent.barrier()

    def train(self, loss):
        if not self.model.training:
            message = "model is in evaluation mode, can not train it; "
            message += "call the 'train' method to set it in training mode explicitly"
            raise RuntimeError(message)
        self.model._zero_grad()
        loss.backward()
        asyncio.run(self.model._push_tensors(skip_no_grad=self.skip_no_grad))
