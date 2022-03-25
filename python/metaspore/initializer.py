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

import abc
import torch
import numpy

class TensorInitializer(abc.ABC):
    @abc.abstractmethod
    def __repr__(self):
        return '%s()' % self.__class__.__name__

    @abc.abstractmethod
    def initialize_dense(self, name, data):
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_sparse(self, name, data, keys):
        raise NotImplementedError

    def __call__(self, name, data, keys):
        if data.size == 0:
            return
        data = torch.from_numpy(data)
        if keys is None:
            self.initialize_dense(name=name, data=data)
        else:
            keys = torch.from_numpy(keys.view(numpy.int64))
            self.initialize_sparse(name=name, data=data, keys=keys)

    def _log_initialized(self, name, data):
        string = f'name: \033[31m{name}\033[m, '
        string += f'dtype: {data.dtype}, '
        string += f'shape: {data.shape} '
        string += '\033[32minitialized\033[m'
        print(string)

class DefaultTensorInitializer(TensorInitializer):
    def __repr__(self):
        return super().__repr__()

    def initialize_dense(self, name, data):
        self.initialize_tensor(name, data)

    def initialize_sparse(self, name, data, keys):
        self.initialize_tensor(name, data)

    def initialize_tensor(self, name, data):
        #self._log_initialized(name, data)
        if name.endswith('bias'):
            data.fill_(0)
        else:
            torch.nn.init.normal_(data)

class ZeroTensorInitializer(TensorInitializer):
    def __repr__(self):
        return super().__repr__()

    def initialize_dense(self, name, data):
        self.initialize_tensor(name, data)

    def initialize_sparse(self, name, data, keys):
        self.initialize_tensor(name, data)

    def initialize_tensor(self, name, data):
        data.fill_(0)

class OneTensorInitializer(TensorInitializer):
    def __repr__(self):
        return super().__repr__()

    def initialize_dense(self, name, data):
        self.initialize_tensor(name, data)

    def initialize_sparse(self, name, data, keys):
        self.initialize_tensor(name, data)

    def initialize_tensor(self, name, data):
        data.fill_(1)

class NormalTensorInitializer(TensorInitializer):
    def __init__(self, mean=0.0, var=1.0):
        if not isinstance(mean, float):
            message = "mean must be float; "
            message += "%r is invalid" % var
            raise ValueError(message)
        if not isinstance(var, float) or var <= 0.0:
            message = "var must be positive float; "
            message += "%r is invalid" % var
            raise ValueError(message)
        self._mean = mean
        self._var = var

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self._mean,
                               self._var)

    def initialize_dense(self, name, data):
        self.initialize_tensor(name, data)

    def initialize_sparse(self, name, data, keys):
        self.initialize_tensor(name, data)

    def initialize_tensor(self, name, data):
        if name.endswith('bias'):
            data.fill_(0)
        else:
            torch.nn.init.normal_(data, self._mean, self._var)

class XavierTensorInitializer(TensorInitializer):
    def __init__(self, activation_type='relu', distribution_type='uniform'):
        if distribution_type not in ('uniform', 'normal'):
            message = "distribution_type of XavierTensorInitializer "
            message += "must be one of 'uniform', 'normal'; "
            message += "%r is invalid" % distribution_type
            raise ValueError(message)
        self._activation_type = activation_type
        self._distribution_type = distribution_type
        self._gain = torch.nn.init.calculate_gain(self._activation_type)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self._distribution_type,
                               self._activation_type)

    def initialize_dense(self, name, data):
        self.initialize_tensor(name, data)

    def initialize_sparse(self, name, data, keys):
        self.initialize_tensor(name, data)

    def initialize_tensor(self, name, data):
        if self._distribution_type == 'uniform':
            torch.nn.init.xavier_uniform_(data, gain=self._gain)
        elif self._distribution_type == 'normal':
            torch.nn.init.xavier_normal_(data, gain=self._gain)
        else:
            assert False
